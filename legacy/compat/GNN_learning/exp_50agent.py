"""Compatibility launcher for the unified CLI.

The historical FIM/lambda experiment is archived in the step-1 snapshot.
This module performs no work on import. Pass generate/train/evaluate/run explicitly.
"""
from pathlib import Path
import sys

if not __package__:
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def main():
    from GNN_learning.experiment import main as unified_main
    return unified_main()


if __name__ == '__main__':
    main()
