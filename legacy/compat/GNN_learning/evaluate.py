"""Compatibility import only. Production implementation: gib4cl.evaluation.evaluate."""
from pathlib import Path
import sys
if not __package__:
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gib4cl.compat import export_module
export_module('gib4cl.evaluation.evaluate',globals())

if __name__ == '__main__':
    main()
