"""Compatibility import only. Production implementation: gib4cl.learning.model."""
from pathlib import Path
import sys
if not __package__:
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gib4cl.compat import export_module
export_module('gib4cl.learning.model',globals())
