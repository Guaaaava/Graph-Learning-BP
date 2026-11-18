"""Nested job logs and checkpoints must remain writable past Win32 MAX_PATH."""
import json
import torch
from gib4cl.runtime import atomic_json, atomic_torch, _io_path


def test_atomic_persistence_handles_long_nested_job_paths(tmp_path):
    directory=tmp_path.joinpath(*(['nested-job-component']*12))
    target=directory/'batch-00001.json'
    assert len(str(target.absolute()))>260
    atomic_json(target,{'batch':1,'status':'complete'})
    assert json.loads(_io_path(target).read_text(encoding='utf-8'))=={'batch':1,'status':'complete'}
    atomic_json(target,{'batch':2,'status':'complete'})
    assert json.loads(_io_path(target).read_text(encoding='utf-8'))['batch']==2
    checkpoint=directory/'last_checkpoint.pt'
    atomic_torch(checkpoint,{'state':torch.tensor([1.,2.])})
    value=torch.load(_io_path(checkpoint),map_location='cpu',weights_only=True)
    assert torch.equal(value['state'],torch.tensor([1.,2.]))
    assert not list(_io_path(directory).glob('.tmp-*'))


def test_atomic_target_parent_components_are_normalized(tmp_path):
    target=tmp_path/'not-created'/'..'/'normalized.json'
    atomic_json(target,{'value':3})
    assert json.loads((tmp_path/'normalized.json').read_text(encoding='utf-8'))=={'value':3}
