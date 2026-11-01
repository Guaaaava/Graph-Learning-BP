"""Run the Chapter 2 particle-BP diagnostic without GNN training."""
from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_settings
from .experiment import render_outputs, run_experiment


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for command in ('pilot', 'run'):
        item = commands.add_parser(command, help='Numerical budget sweep' if command == 'pilot' else 'Fixed-budget paired study')
        item.add_argument('--config', type=Path, required=True)
        item.add_argument('--output-dir', type=Path, required=True, help='New result directory; use --resume to continue it')
        item.add_argument('--resume', action='store_true')
        item.add_argument('--stop-after-cases', type=int)
        item.add_argument('--no-plots', action='store_true')
    item = commands.add_parser('plot', help='Verify persisted trials and regenerate plots/tables without inference')
    item.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == 'plot':
            result = render_outputs(args.output_dir)
        else:
            result = run_experiment(load_settings(args.config), args.output_dir,
                                    phase='pilot' if args.command == 'pilot' else 'study', resume=args.resume,
                                    stop_after_cases=args.stop_after_cases, make_plots=not args.no_plots)
    except (ValueError, AssertionError, FloatingPointError, FileExistsError, FileNotFoundError) as error:
        parser.exit(1, str(error)+'\n')
    print(f"{result['status']}: {result['completed_cases']}/{result['expected_cases']} independent trials")
    print('Results: '+str(args.output_dir.resolve()))
    return result


if __name__ == '__main__':
    main()
