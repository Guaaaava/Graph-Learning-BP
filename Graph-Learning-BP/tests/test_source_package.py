"""Source-only packaging, portable provenance and tamper detection."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import zipfile
import pytest
from gib4cl.config import REPO_ROOT
from tools.package_project import (build_package,verify_archive,verify_tree,collect_sources,
                                  _manifest,_validate_manifest,MANIFEST,PREFIX)
from tools.provenance import fixture


@pytest.fixture(scope='module')
def built(tmp_path_factory):
    root = tmp_path_factory.mktemp('source-package')
    result = build_package(root/'build')
    return root,Path(result['archive']),result


def unpack(archive,destination):
    with zipfile.ZipFile(archive) as handle:
        handle.extractall(destination)
    return destination/PREFIX


def test_package_contains_only_current_sources_and_has_no_execution_history(built):
    _,archive,result = built
    with zipfile.ZipFile(archive) as handle:
        names = handle.namelist()
        assert all(not any(part in {'legacy','GNN_learning','BP','NBP_ST','NEBP','references'}
                           for part in Path(name).parts[:2]) for name in names)
        assert PREFIX+'/tests/fixtures/provenance.json' in names
        assert PREFIX+'/tests/references/particle_posterior.py' in names
        assert not any('__pycache__' in name or '.venv' in name or name.endswith('.zip') for name in names)
        assert result['files']==len(names)-1
    assert verify_archive(archive)['source_fingerprint']==result['source_fingerprint']


def test_identical_sources_build_byte_identical_archives(built,tmp_path):
    _,archive,original = built
    repeated = build_package(tmp_path/'repeat')
    assert repeated['archive_sha256']==original['archive_sha256']
    assert Path(repeated['archive']).read_bytes()==archive.read_bytes()


def test_unpack_verifies_without_legacy_and_accepts_generated_outputs(built,tmp_path):
    _,archive,_ = built
    root = unpack(archive,tmp_path/'unpacked')
    assert not (root/'legacy').exists()
    before = verify_tree(root)
    (root/'outputs').mkdir()
    (root/'outputs'/'new-result.json').write_text('{}',encoding='utf-8')
    assert verify_tree(root)==before


def test_source_tampering_is_rejected(built,tmp_path):
    root = unpack(built[1],tmp_path/'unpacked')
    path = root/'gib4cl/config.py'
    path.write_text(path.read_text(encoding='utf-8')+'\n# modified\n',encoding='utf-8')
    with pytest.raises(ValueError,match='checksum'):
        verify_tree(root)


def test_missing_source_is_rejected(built,tmp_path):
    root = unpack(built[1],tmp_path/'unpacked')
    (root/'requirements.txt').unlink()
    with pytest.raises(ValueError,match='missing'):
        verify_tree(root)


def test_unlisted_source_is_rejected(built,tmp_path):
    root = unpack(built[1],tmp_path/'unpacked')
    (root/'gib4cl/unlisted.py').write_text('pass\n',encoding='utf-8')
    with pytest.raises(ValueError,match='inventory'):
        verify_tree(root)


def test_returned_history_in_release_root_is_rejected(built,tmp_path):
    root = unpack(built[1],tmp_path/'unpacked')
    (root/'legacy').mkdir()
    with pytest.raises(ValueError,match='Unexpected root'):
        verify_tree(root)


def test_unexpected_archive_entry_is_rejected(built,tmp_path):
    archive = tmp_path/'unexpected.zip'
    shutil.copyfile(built[1],archive)
    with zipfile.ZipFile(archive,'a') as handle:
        handle.writestr(PREFIX+'/legacy/old.py','pass')
    with pytest.raises(ValueError,match='unexpected'):
        verify_archive(archive)


@pytest.mark.parametrize('name',('../outside.py','/outside.py','gib4cl/../outside.py',
                                 'C:/outside.py','gib4cl\\outside.py','legacy/old.py'))
def test_manifest_rejects_paths_outside_source_allowlist(name):
    files = collect_sources(REPO_ROOT)
    files[name]={'sha256':'0'*64,'bytes':0}
    with pytest.raises(ValueError):
        _validate_manifest(_manifest(files))


def test_incomplete_manifest_rejected():
    with pytest.raises(ValueError,match='Incomplete'):
        _validate_manifest(_manifest({'README.md':{'sha256':'0'*64,'bytes':0}}))


def test_build_does_not_overwrite_an_existing_directory(built):
    with pytest.raises(FileExistsError):
        build_package(built[0]/'build')


def test_frozen_provenance_can_be_checked_without_any_live_history(built,tmp_path):
    root = unpack(built[1],tmp_path/'unpacked')
    # Exercise the extracted code, not the working-copy ROOT cached in this process.
    import subprocess,sys,os
    code = ("from tools.provenance import preserved_core,preserved_references,provenance_scope; "
            "assert len(preserved_core())==3 and len(preserved_references())==8; "
            "s=provenance_scope(); assert not s['live_reference_archive_checked']; "
            "assert not s['historical_source_snapshot_checked']; "
            "from gib4cl.data.generate import _source_version; assert _source_version()['git_head'] is None")
    process = subprocess.run([sys.executable,'-B','-c',code],cwd=root,
                             env=dict(os.environ,PYTHONPATH=str(root)),capture_output=True,text=True)
    assert process.returncode==0,process.stderr


def test_cpu_acceptance_prepares_the_data_parent_in_a_fresh_checkout(tmp_path):
    from tools.accept_cpu import _test_workspace
    assert not (tmp_path/'data').exists()
    workspace = _test_workspace(tmp_path)
    assert workspace.parent==tmp_path/'data' and workspace.parent.is_dir()
    assert not workspace.exists()  # pytest owns creation of its private base directory
