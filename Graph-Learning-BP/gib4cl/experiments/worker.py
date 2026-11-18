"""Internal isolated suite worker; the public user interface is python -m gib4cl."""
import argparse
from gib4cl.config import load_config
from gib4cl.learning.train import train_gnn
from gib4cl.evaluation.evaluate import evaluate


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('kind',choices=('train','evaluate'))
    parser.add_argument('--config',required=True)
    parser.add_argument('--destination',required=True)
    parser.add_argument('--data-root',required=True)
    parser.add_argument('--device',choices=('cpu','cuda'),required=True)
    parser.add_argument('--threads',type=int,required=True)
    parser.add_argument('--checkpoint')
    parser.add_argument('--split',default='test')
    parser.add_argument('--max-graphs',type=int)
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    if args.kind=='train':
        return train_gnn(settings=cfg,data_root=args.data_root,run_directory=args.destination,
                         device=args.device,threads=args.threads)
    if args.checkpoint is None:
        parser.error('evaluate requires --checkpoint')
    return evaluate(checkpoint=args.checkpoint,settings=cfg,data_root=args.data_root,
                    evaluation_directory=args.destination,split=args.split,max_graphs=args.max_graphs,
                    device=args.device,threads=args.threads)


if __name__=='__main__':
    main()
