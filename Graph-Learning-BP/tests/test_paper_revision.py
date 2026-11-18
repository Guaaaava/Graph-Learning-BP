"""Reviewed manuscript registration preserves the old baseline and rejects drift."""
import json
import shutil
import pytest
from gib4cl.config import REPO_ROOT
from tools import provenance


@pytest.fixture
def copy_registration(tmp_path,monkeypatch):
    root=tmp_path/'checkout'
    reference=provenance.fixture()
    for name in ('tests/fixtures/provenance.json',reference['paper_current']['review_record']):
        target=root/name
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(REPO_ROOT/name,target)
    monkeypatch.setattr(provenance,'ROOT',root)
    return root,reference


def test_reviewed_registration_preserves_baseline_without_claiming_external_read(copy_registration):
    _,reference=copy_registration
    identity=provenance.paper_identity()
    assert identity['registered_sha256']=='03e3b31ae08fef1c050c35c63b00c34cc5e11a2f53645ac11140d85186a96266'
    assert identity['current_sha256']==reference['paper_current']['sha256']
    assert identity['external_manuscript_checked'] is False
    assert provenance.paper_sha256()==identity['current_sha256']


def test_unknown_external_manuscript_is_rejected(copy_registration):
    root,reference=copy_registration
    (root.parent/reference['paper']['filename']).write_text('unreviewed revision',encoding='utf-8')
    with pytest.raises(AssertionError,match='External manuscript'):
        provenance.paper_identity()


@pytest.mark.parametrize('change',['remove','tamper'])
def test_missing_or_modified_review_record_is_rejected(copy_registration,change):
    root,reference=copy_registration
    path=root/reference['paper_current']['review_record']
    if change=='remove':
        path.unlink()
    else:
        path.write_text('{}',encoding='utf-8')
    with pytest.raises(AssertionError,match='Paper review record'):
        provenance.paper_identity()


def test_original_only_fixture_still_enforces_original_draft(copy_registration):
    root,reference=copy_registration
    reference.pop('paper_current')
    (root/'tests/fixtures/provenance.json').write_text(json.dumps(reference),encoding='utf-8')
    assert provenance.paper_sha256()==reference['paper']['sha256']
    (root.parent/reference['paper']['filename']).write_text('not the registered original',encoding='utf-8')
    with pytest.raises(AssertionError,match='External manuscript'):
        provenance.paper_sha256()
