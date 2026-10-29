"""Metadata cleanup keeps full current-source checks and frozen historical evidence."""
import shutil
import pytest

from gib4cl.config import REPO_ROOT
from tools import provenance


def test_historical_projection_only_removes_algorithm_metadata():
    historical = '''class EdgePredictorGNN:
    architecture_version = 'historical'
    def get_extra_state(self):
        return {'architecture_version': self.architecture_version}
    def score_edges(self, a, b):
        return 0.5 * (a + b)
'''
    current = '''class EdgePredictorGNN:
    def score_edges(self, a, b):
        return 0.5 * (a + b)
'''
    projected = provenance.without_algorithm_tags(historical)
    assert provenance.semantic_digest(projected) == provenance.semantic_digest(current)
    changed = current.replace('0.5', '0.25')
    assert provenance.semantic_digest(projected) != provenance.semantic_digest(changed)


@pytest.mark.parametrize('change', ['calculation', 'metadata'])
def test_portable_core_check_does_not_normalize_current_source(tmp_path, monkeypatch, change):
    reference = provenance.fixture()
    for name in [*reference['core'], 'tests/fixtures/provenance.json']:
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO_ROOT/name, target)
    monkeypatch.setattr(provenance, 'ROOT', tmp_path)
    assert len(provenance.preserved_core()) == 3
    path = tmp_path/'gib4cl/learning/model.py'
    source = path.read_text(encoding='utf-8')
    if change == 'calculation':
        changed = source.replace('0.5 * (forward + backward)', '0.25 * (forward + backward)')
    else:
        changed = source.replace('class EdgePredictorGNN(nn.Module):',
                                 'class EdgePredictorGNN(nn.Module):\n    architecture_version = "injected"')
    assert changed != source
    path.write_text(changed, encoding='utf-8')
    with pytest.raises(AssertionError, match='Core changed'):
        provenance.preserved_core()
