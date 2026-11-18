"""Build and verify the self-contained source package; no training dependencies required."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path,PurePosixPath
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 'gib4cl.source-package.v1'
PREFIX = 'Graph-Learning-BP'
MANIFEST = 'PACKAGE-MANIFEST.json'
SOURCE_DIRS = ('gib4cl','configs','tests','tools','docs')
ROOT_FILES = ('README.md','requirements.txt','pytest.ini','.gitignore','.gitattributes')
REQUIRED_FILES = {'gib4cl/__main__.py','gib4cl/config.py','gib4cl/experiments/cli.py',
                  'gib4cl/learning/train.py','gib4cl/evaluation/evaluate.py',
                  'configs/experiment.json','configs/smoke-pipeline.json','configs/suites/smoke-seeds.json',
                  'tests/fixtures/provenance.json','tests/audits/checkpoints.py','tests/audits/metrics.py',
                  'tools/accept_cpu.py','tools/provenance.py','tools/package_project.py'}
FORBIDDEN_ROOTS = {'legacy','references','GNN_learning','BP','NBP_ST','NEBP','results'}
SUFFIXES = {'.py','.json','.md','.txt','.ini','.yaml','.yml','.toml','.png','.svg','.pdf'}
CACHE_NAMES = {'__pycache__','.pytest_cache'}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode('utf-8')


def _json(data):
    def pairs(items):
        value = {}
        for key,item in items:
            if key in value:
                raise ValueError('Duplicate manifest key: '+key)
            value[key] = item
        return value
    return json.loads(data,object_pairs_hook=pairs)


def _relative(name):
    path = PurePosixPath(name)
    if not isinstance(name,str) or '\\' in name or ':' in name or path.is_absolute() or str(path)!=name:
        raise ValueError('Invalid package path: '+str(name))
    if any(part in ('.','..') for part in path.parts) or not path.parts:
        raise ValueError('Invalid package path: '+name)
    if path.parts[0] in FORBIDDEN_ROOTS:
        raise ValueError('Historical code cannot be in a release package')
    if name not in ROOT_FILES and (path.parts[0] not in SOURCE_DIRS or len(path.parts)<2):
        raise ValueError('Path is outside the source allowlist: '+name)
    return path


def collect_sources(root=ROOT):
    root = Path(root).resolve()
    paths = []
    for directory in SOURCE_DIRS:
        base = root/directory
        if not base.is_dir() or base.is_symlink():
            raise ValueError('Required source directory missing or linked: '+directory)
        for path in base.rglob('*'):
            if any(part in CACHE_NAMES for part in path.relative_to(root).parts):
                continue
            if path.is_symlink():
                raise ValueError('Linked source is not packaged: '+str(path))
            if path.is_file():
                if path.suffix.lower() not in SUFFIXES:
                    raise ValueError('Unexpected artifact inside source directory: '+str(path))
                paths.append(path)
    for name in ROOT_FILES:
        path = root/name
        if not path.is_file() or path.is_symlink():
            raise ValueError('Required root file missing or linked: '+name)
        paths.append(path)
    files = {}
    for path in sorted(paths):
        if not path.resolve().is_relative_to(root):
            raise ValueError('Source escapes checkout')
        name = path.relative_to(root).as_posix()
        _relative(name)
        data = path.read_bytes()
        files[name] = {'sha256':digest(data),'bytes':len(data)}
    return files


def _manifest(files):
    return {'schema_version':SCHEMA,'root_directory':PREFIX,'files':files,
            'source_fingerprint':digest(canonical(files)),
            'excluded':'legacy, local environments, Git/IDE metadata, caches, data, outputs and dist'}


def _validate_manifest(value):
    if not isinstance(value,dict) or value.get('schema_version')!=SCHEMA or value.get('root_directory')!=PREFIX:
        raise ValueError('Unsupported source package manifest')
    files = value.get('files')
    if not isinstance(files,dict) or not files:
        raise ValueError('Empty package manifest')
    for name,entry in files.items():
        _relative(name)
        if not isinstance(entry,dict) or set(entry)!={'sha256','bytes'}:
            raise ValueError('Malformed package file entry')
        if not isinstance(entry['sha256'],str) or len(entry['sha256'])!=64 or any(c not in '0123456789abcdef' for c in entry['sha256']):
            raise ValueError('Malformed SHA-256')
        if isinstance(entry['bytes'],bool) or not isinstance(entry['bytes'],int) or entry['bytes']<0:
            raise ValueError('Malformed source byte count')
    if not (set(ROOT_FILES)|REQUIRED_FILES)<=set(files) or any(not any(p.startswith(d+'/') for p in files) for d in SOURCE_DIRS):
        raise ValueError('Incomplete source package')
    if digest(canonical(files))!=value.get('source_fingerprint'):
        raise ValueError('Package manifest fingerprint mismatch')
    return files


def verify_archive(path):
    path = Path(path).resolve()
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names)!=len(set(names)):
            raise ValueError('Duplicate ZIP entries')
        manifest_name = PREFIX+'/'+MANIFEST
        if manifest_name not in names:
            raise ValueError('ZIP is missing the package manifest')
        manifest = _json(archive.read(manifest_name))
        files = _validate_manifest(manifest)
        expected = {PREFIX+'/'+name for name in files}|{manifest_name}
        if set(names)!=expected:
            raise ValueError('ZIP contains missing or unexpected entries')
        for name,entry in files.items():
            data = archive.read(PREFIX+'/'+name)
            if digest(data)!=entry['sha256'] or len(data)!=entry['bytes']:
                raise ValueError('ZIP source checksum mismatch: '+name)
    return {'schema_version':SCHEMA,'status':'verified','files':len(files),
            'source_fingerprint':manifest['source_fingerprint'],'archive_sha256':digest(path.read_bytes())}


def verify_tree(root=ROOT):
    root = Path(root).resolve()
    manifest_path = root/MANIFEST
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise ValueError('Unpacked package manifest is missing')
    manifest = _json(manifest_path.read_bytes())
    files = _validate_manifest(manifest)
    for path in root.iterdir():
        name = path.name
        generated = name in {'data','outputs','dist','.git','.vscode',*CACHE_NAMES} or name.startswith('.venv')
        if name in FORBIDDEN_ROOTS or (name not in {*SOURCE_DIRS,*ROOT_FILES,MANIFEST} and not generated):
            raise ValueError('Unexpected root entry in unpacked package: '+name)
    actual = collect_sources(root)
    if actual!=files:
        raise ValueError('Unpacked source checksum/file inventory mismatch')
    return {'schema_version':SCHEMA,'status':'verified','files':len(files),
            'source_fingerprint':manifest['source_fingerprint']}


def build_package(output_root,*,source_root=ROOT):
    source_root = Path(source_root).resolve()
    files = collect_sources(source_root)
    manifest = _manifest(files)
    _validate_manifest(manifest)
    output_root = Path(output_root).resolve()
    if any(output_root.is_relative_to(source_root/d) for d in SOURCE_DIRS):
        raise ValueError('Build output must not be inside a source directory')
    output_root.mkdir(parents=True,exist_ok=False)
    payload = (json.dumps(manifest,ensure_ascii=False,sort_keys=True,indent=2)+'\n').encode('utf-8')
    path = output_root/(PREFIX+'.zip')
    with zipfile.ZipFile(path,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for name in [*files,MANIFEST]:
            data = payload if name==MANIFEST else (source_root/name).read_bytes()
            info = zipfile.ZipInfo(PREFIX+'/'+name,date_time=(1980,1,1,0,0,0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info,data,compress_type=zipfile.ZIP_DEFLATED,compresslevel=6)
    if collect_sources(source_root)!=files:
        raise ValueError('Source changed during packaging; use a fresh build directory')
    checked = verify_archive(path)
    (output_root/MANIFEST).write_bytes(payload)
    (output_root/(path.name+'.sha256')).write_text(checked['archive_sha256']+'  '+path.name+'\n',encoding='ascii')
    return dict(checked,archive=str(path),manifest=str(output_root/MANIFEST))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command',required=True)
    build = commands.add_parser('build',help='Build a source-only ZIP without any legacy code')
    build.add_argument('--output-root',required=True)
    verify = commands.add_parser('verify',help='Verify archive inventory and every source checksum')
    target = verify.add_mutually_exclusive_group(required=True)
    target.add_argument('--archive')
    target.add_argument('--directory')
    args = parser.parse_args(argv)
    result = build_package(args.output_root) if args.command=='build' else (
        verify_archive(args.archive) if args.archive else verify_tree(args.directory))
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return result


if __name__=='__main__':
    main()
