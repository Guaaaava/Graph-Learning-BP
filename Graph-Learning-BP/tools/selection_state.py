"""Auditable job cache for explicitly requested selection stages."""
from __future__ import annotations
import json
from pathlib import Path
import shutil
import tempfile
import os

from gib4cl.config import REPO_ROOT, canonical_json, fingerprint
from gib4cl.runtime import atomic_json, sha256, source_manifest, runtime_info, _io_path


def read_json(path):
    def pairs(items):
        result = {}
        for key,value in items:
            if key in result:
                raise ValueError('Duplicate JSON key: '+key)
            result[key] = value
        return result
    def reject(value):
        raise ValueError('Nonfinite JSON: '+value)
    return json.loads(_io_path(path).read_text(encoding='utf-8'),object_pairs_hook=pairs,parse_constant=reject)


def sources():
    return {**source_manifest(), **{p.relative_to(REPO_ROOT).as_posix():sha256(p)
            for p in sorted((REPO_ROOT/'tools').glob('selection_*.py'))}}


def freeze_copy(source,destination):
    final_path = Path(destination).resolve()
    source,destination = _io_path(source),_io_path(destination)
    expected = sha256(source)
    if destination.exists():
        if sha256(destination) != expected:
            raise ValueError('Immutable selection artifact differs: '+str(destination))
        return final_path
    destination.parent.mkdir(parents=True,exist_ok=True)
    fd,name = tempfile.mkstemp(dir=destination.parent,prefix='.copy-')
    os.close(fd)
    try:
        shutil.copyfile(source,name)
        os.replace(name,destination)
    finally:
        if Path(name).exists():
            Path(name).unlink()
    if sha256(destination) != expected:
        raise ValueError('Snapshot copy checksum mismatch')
    return final_path


class WorkLimit(Exception):
    pass


class Workspace:
    def __init__(self,root,*,kind,config,protocol,device,resume=False,retry_failed=False,max_jobs=None,
                 control=None,max_training_epochs=None):
        if max_jobs is not None and (isinstance(max_jobs,bool) or max_jobs < 1):
            raise ValueError('max_jobs must be positive')
        self.root = Path(root).resolve()
        self.limit,self.executed,self.retry = max_jobs,0,retry_failed
        if max_training_epochs is not None and (isinstance(max_training_epochs,bool) or
                not isinstance(max_training_epochs,int) or max_training_epochs<1):
            raise ValueError('max_training_epochs must be a positive integer')
        self.control,self.max_training_epochs = control,max_training_epochs
        identity = {'version':'gib4cl.selection-workspace.v1','kind':kind,
                    'config':config.to_dict(),'protocol':protocol,'sources':sources(),
                    'runtime':runtime_info(device)}
        self.identity = json.loads(canonical_json(identity))
        self.path = self.root/'manifest.json'
        if resume:
            self.manifest = read_json(self.path)
            if self.manifest['identity'] != self.identity:
                raise ValueError('Selection resume requires identical config, protocol, sources and runtime')
        else:
            self.root.mkdir(parents=True,exist_ok=False)
            self.manifest = {'identity':self.identity,'jobs':{},'reports':{}}
            self.save()

    def save(self):
        atomic_json(self.path,self.manifest)

    def should_stop(self):
        return self.control is not None and self.control.stop_requested()

    def artifact(self,path):
        path = Path(path).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError('Selection artifact escapes workspace')
        return {'path':path.relative_to(self.root).as_posix(),'sha256':sha256(_io_path(path))}

    def verify(self,artifact):
        path = (self.root/artifact['path']).resolve()
        if not path.is_relative_to(self.root) or not _io_path(path).is_file() or sha256(_io_path(path)) != artifact['sha256']:
            raise ValueError('Selection artifact missing or changed: '+artifact['path'])
        return path

    def report(self,name):
        if name not in self.manifest['reports']:
            raise ValueError('Required completed selection stage/report missing: '+name)
        return read_json(self.verify(self.manifest['reports'][name]))

    def bind(self,name,settings):
        settings = json.loads(canonical_json(settings))
        requests = self.manifest.setdefault('requests',{})
        if name in requests and requests[name] != settings:
            raise ValueError('Requested stage settings differ on resume: '+name)
        requests[name] = settings
        self.save()

    def publish(self,name,value):
        if sources() != self.identity['sources']:
            raise ValueError('Sources changed while selection work was running')
        path = self.root/'reports'/(name+'.json')
        atomic_json(path,value)
        self.manifest['reports'][name] = self.artifact(path)
        self.save()
        return value

    def job(self,key,function):
        key = json.loads(canonical_json(key))
        job_id = fingerprint(key)
        path = self.root/'jobs'/(job_id+'.json')
        old = None
        if _io_path(path).exists():
            old = read_json(path)
            content = {k:v for k,v in old.items() if k != 'payload_sha256'}
            if old.get('payload_sha256') != fingerprint(content) or old.get('key') != key:
                raise ValueError('Selection job record changed: '+job_id)
            registered = self.manifest['jobs'].get(job_id)
            if registered is not None and registered != self.artifact(path):
                raise ValueError('Selection job checksum differs from manifest')
            for artifact in old.get('artifacts',[]):
                self.verify(artifact)
            # Adopt a complete atomic record if a crash preceded manifest update.
            self.manifest['jobs'][job_id] = self.artifact(path)
            self.save()
            if old['status'] == 'complete':
                return old['value']
            if old['status'] not in ('failed','paused'):
                raise ValueError('Unknown selection job status')
            if old['status']=='failed' and not self.retry:
                raise ValueError('Failed job retained; use explicit retry_failed: '+job_id)
        if self.limit is not None and self.executed >= self.limit:
            raise WorkLimit()
        if self.should_stop():
            raise WorkLimit()
        self.executed += 1
        if self.control is not None:
            self.control.job_started(key)
        try:
            value,artifacts = function()
            row = {'key':key,'status':'complete','value':value,
                   'artifacts':[self.artifact(p) for p in artifacts]}
        except WorkLimit:
            # A committed epoch slice is resumable progress, not a failed trial.
            row = {'key':key,'status':'paused','artifacts':[]}
            row['payload_sha256'] = fingerprint(row)
            atomic_json(path,row)
            self.manifest['jobs'][job_id] = self.artifact(path)
            self.save()
            raise
        except Exception as error:
            row = {'key':key,'status':'failed','error':repr(error),'artifacts':[]}
            attempt = 1
            failure = self.root/'failures'/f'{job_id}-{attempt:03d}.json'
            while _io_path(failure).exists():
                attempt += 1
                failure = self.root/'failures'/f'{job_id}-{attempt:03d}.json'
            atomic_json(failure,row)
            row['payload_sha256'] = fingerprint(row)
            atomic_json(path,row)
            self.manifest['jobs'][job_id] = self.artifact(path)
            self.save()
            raise
        row['payload_sha256'] = fingerprint(row)
        atomic_json(path,row)
        self.manifest['jobs'][job_id] = self.artifact(path)
        self.save()
        return value
