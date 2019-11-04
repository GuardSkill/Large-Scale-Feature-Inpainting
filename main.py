import os
import cv2
import random
import numpy as np
import torch
import argparse
from shutil import copyfile

from tune_parameters import randomTune
from src.config import Config
from src.process import CLFNet
from src.utils import *
import os


def main(mode=None):
    r"""starts the model

    Args:
        mode (int): 1: train, 2: test, 3: eval, reads from config file if not specified
    """

    config = load_config(mode)
    # set cuda visble devices from config file
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(e) for e in config.GPU)
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # init device
    if torch.cuda.is_available():
        config.DEVICE = torch.device("cuda")
        torch.backends.cudnn.benchmark = True  # cudnn auto-tuner
    else:
        config.DEVICE = torch.device("cpu")
    # set cv2 running threads to 1 (prevents deadlocks with pytorch dataloader)
    cv2.setNumThreads(0)
    # initialize random seed
    torch.manual_seed(config.SEED)
    torch.cuda.manual_seed_all(config.SEED)
    np.random.seed(config.SEED)
    random.seed(config.SEED)

    # tune parameters
    # randomT0une(config)

    # build the model and initialize
    model = CLFNet(config)
    model.load()

    # model training
    if config.MODE == 1:
        # config.print()
        print('\nstart training...\n')
        model.train()

    # model test
    elif config.MODE == 2:
        print('\nstart testing...\n')
        model.test()

    # eval mode
    else:
        print('\nstart eval...\n')
        model.eval(0)


def load_config(mode=None):
    r"""loads model config

    Args:
        mode (int): 1: train, 2: test, 3: eval, reads from config file if not specified
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--path', '--checkpoints', type=str, default='./checkpoints',
                        help='model checkpoints path (default: ./checkpoints)')
    parser.add_argument('--output', type=str, default='./output', help='path to the output directory')

    # test mode
    if mode == 2:
        parser.add_argument('--input', type=str, help='path to the input images directory or an input image')
        parser.add_argument('--mask', type=str, help='path to the masks directory or a mask file')

    args = parser.parse_args()
    config_path = os.path.join(args.path, 'config.yml')

    # create checkpoints path if does't exist
    create_dir(args.path)

    # copy config template if does't exist
    if not os.path.exists(config_path):
        copyfile('./config.yml.example', config_path)

    # load config file
    config = Config(config_path)

    # train mode
    if mode == 1:
        config.MODE = 1

    # test mode
    elif mode == 2:
        config.MODE = 2
        # config.INPUT_SIZE = 0         Set to 0 for one to one mapping

        if args.input is not None:
            config.TEST_FLIST = args.input

        if args.mask is not None:
            config.TEST_MASK_FLIST = args.mask

        if args.output is not None:
            config.RESULTS = args.output

    # eval mode
    elif mode == 3:
        config.MODE = 3

    return config


if __name__ == "__main__":
    main()
