"""Entry point."""
import os

import torch

import data
import config
import utils
import trainer

import numpy
import random

logger = utils.get_logger()


def main(args):  # pylint:disable=redefined-outer-name
    """main: Entry point."""
    utils.prepare_dirs(args)

    torch.manual_seed(args.random_seed)
    # Add this for the random seed
    numpy.random.seed(args.random_seed)
    random.seed(args.random_seed)
    torch.backends.cudnn.deterministic = True

    if args.num_gpu > 0:
        torch.cuda.manual_seed(args.random_seed)

    if args.network_type == 'rnn':
        dataset = data.text.Corpus(args.data_path)
        trnr = trainer.Trainer(args, dataset)
    elif 'cnn' in args.network_type:
        dataset = data.image.Image(args)
        trnr = trainer.CNNTrainer(args, dataset)
    else:
        raise NotImplementedError(f"{args.dataset} is not supported")

    if args.mode == 'train':
        utils.save_args(args)
        trnr.train()
    elif args.mode == 'derive':
        assert args.load_path != "", ("`--load_path` should be given in "
                                      "`derive` mode")
        trnr.derive()
    else:
        if not args.load_path:
            raise Exception("[!] You should specify `load_path` to load a "
                            "pretrained model")
        trnr.test()


if __name__ == "__main__":
    args, unparsed = config.get_args()
    main(args)
