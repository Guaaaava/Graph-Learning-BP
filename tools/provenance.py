"""Portable provenance checks; history is optional and excluded from release packages."""
import ast
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def fixture():
    value = json.loads((ROOT/'tests/fixtures/provenance.json').read_text(encoding='utf-8'))
    if value.get('schema_version')!='gib4cl.provenance-fixture.v1':
        raise ValueError('Unsupported provenance fixture')
    expected = {'gib4cl/inference/particle_bp.py','gib4cl/learning/model.py','gib4cl/learning/objectives.py'}
    if set(value['core'])!=expected or len(value['nebp_sources'])!=8:
        raise ValueError('Incomplete provenance fixture')
    return value


def layout():
    return json.loads((ROOT/'docs/source-layout.json').read_text(encoding='utf-8'))


def source_path(old):
    target = layout()['paths'].get(old,old)
    if old.startswith('NEBP/'):
        target = 'legacy/references/'+old
    path = (ROOT/target).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError('Source path escapes checkout')
    return path


def semantic_digest(source):
    class ImportsOnly(ast.NodeTransformer):
        def visit_Import(self,node):
            return None
        def visit_ImportFrom(self,node):
            return None
        def visit_If(self,node):
            self.generic_visit(node)
            return node if node.body or node.orelse else None
    tree = ImportsOnly().visit(ast.parse(source))
    return hashlib.sha256(ast.dump(tree,include_attributes=False).encode()).hexdigest()


def without_algorithm_tags(source):
    """Project only trusted historical sources through the recorded metadata cleanup.

    Current sources are always checked by their full semantic digest. This
    projection is used only after verifying the immutable historical source.
    """
    tree = ast.parse(source)
    for cls in (node for node in tree.body if isinstance(node, ast.ClassDef)):
        if cls.name not in ('EdgePredictorGNN', 'ParticleBP'):
            continue
        attribute = 'architecture_version' if cls.name == 'EdgePredictorGNN' else 'algorithm_version'
        cls.body = [node for node in cls.body if not (
            isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == attribute for target in node.targets))]
        if cls.name == 'EdgePredictorGNN':
            cls.body = [node for node in cls.body if not (
                isinstance(node, ast.FunctionDef) and node.name in
                ('get_extra_state', 'set_extra_state', 'load_state_dict'))]
        else:
            for method in (node for node in cls.body if isinstance(node, ast.FunctionDef)):
                if method.name == 'from_config':
                    method.body = [node for node in method.body if not (
                        isinstance(node, ast.If) and any(
                            isinstance(item, ast.Attribute) and item.attr == attribute
                            for item in ast.walk(node.test)))]
                elif method.name == 'infer':
                    for node in ast.walk(method):
                        if isinstance(node, ast.Dict):
                            items = [(key, value) for key, value in zip(node.keys, node.values)
                                     if not (isinstance(key, ast.Constant) and key.value == attribute)]
                            node.keys = [key for key, _ in items]
                            node.values = [value for _, value in items]
    return ast.unparse(tree)


def preserved_core():
    reference = fixture()
    snapshot = ROOT/reference['snapshots']['pre_layout']['path']
    originals = {}
    if snapshot.is_file():
        assert hashlib.sha256(snapshot.read_bytes()).hexdigest()==reference['snapshots']['pre_layout']['sha256']
        with zipfile.ZipFile(snapshot) as archive:
            originals = {name:archive.read('repository/'+info['original_path']) for name,info in reference['core'].items()}
    result = {}
    for name,info in reference['core'].items():
        actual = semantic_digest((ROOT/name).read_text(encoding='utf-8'))
        expected = info.get('current_semantic_sha256', info['semantic_sha256'])
        assert actual==expected,'Core changed beyond recorded metadata cleanup: '+name
        if originals:
            assert hashlib.sha256(originals[name]).hexdigest()==info['original_sha256']
            original = originals[name].decode('utf-8')
            assert semantic_digest(original)==info['semantic_sha256']
            cleaned = without_algorithm_tags(original) if 'current_semantic_sha256' in info else original
            assert semantic_digest(cleaned)==actual
        result[name] = dict(info)
    return result


def preserved_references():
    """Return frozen reference hashes; additionally verify a local archive when present.

    A source-only release intentionally contains no NEBP execution files. Its
    provenance is the packaged fixture, protected by PACKAGE-MANIFEST.json.
    """
    reference = fixture()
    directory = ROOT/'legacy/references/NEBP'
    if directory.exists():
        for name,expected in reference['nebp_sources'].items():
            path = ROOT/name
            assert path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest()==expected,'Reference changed: '+name
    snapshot = ROOT/reference['snapshots']['initial']['path']
    if snapshot.is_file():
        assert hashlib.sha256(snapshot.read_bytes()).hexdigest()==reference['snapshots']['initial']['sha256']
    return dict(reference['nebp_sources'])


def provenance_scope():
    reference = fixture()
    return {'fixture':'tests/fixtures/provenance.json',
            'fixture_sha256':hashlib.sha256((ROOT/'tests/fixtures/provenance.json').read_bytes()).hexdigest(),
            'live_reference_archive_checked':(ROOT/'legacy/references/NEBP').exists(),
            'historical_source_snapshot_checked':(ROOT/reference['snapshots']['pre_layout']['path']).is_file(),
            'release_scope':'frozen provenance hashes and executable mathematical references; no dependency on legacy code'}


def paper_sha256():
    expected = fixture()['paper']['sha256']
    path = ROOT.parent/fixture()['paper']['filename']
    if path.is_file():
        assert hashlib.sha256(path.read_bytes()).hexdigest()==expected,'External manuscript differs from the verified draft'
    return expected


def record_path(path):
    path = Path(path).resolve()
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)
