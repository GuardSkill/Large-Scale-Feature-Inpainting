from src.config import Config
from src.utils import create_dir
import numpy as np
import os
# do your thing with the hyper-parameters
from src.CLFModel import CLFNet

TRAIN_LOSS=True
def randomTune(config):
    # 'LR': 0.0001,                   # learning rate
    # 'D2G_LR': 0.1,                  # discriminator/generator learning rate ratio
    # 'BETA1': 0.0,                   # adam optimizer beta1
    # 'BETA2': 0.9,                   # adam optimizer beta2
    # 'L1_LOSS_WEIGHT': 1,            # l1 loss weight
    # 'FM_LOSS_WEIGHT': 10,           # feature-matching loss weight
    config.MAX_STEPS = 1500
    config.EVAL_INTERVAL = 80
    config.MAX_EPOCHES = 1
    # config.MAX_STEPS = 3
    experiments = 50
    for i in range(experiments):
        # sample from a Uniform distribution on a log-scale
        # config.LR = 10 ** np.random.uniform(-3, -5)  # Sample learning rate candidates in the range (0.001 to 0.00001)
        # config.D2G_LR = 10 ** np.random.uniform(-2,
        #                                         0)  # Sample regularization candidates in the range (0.01 to 0.0001)
        # config.LR = 0.0001
        # config.D2G_LR =0.1
        # # config.LR=0.0001

        # config.PATH = './checkpoints/places2_tune_%d_%f%f_' % (i, config.LR, config.D2G_LR)
        # logdir= config.PATH+('/log_%s_%s' % (config.LR , config.D2G_LR))
        create_dir(config.PATH)
        if TRAIN_LOSS:
            # if config.MODEL == 1:
                # config.L1_LOSS_WEIGHT = 10 ** np.random.uniform(-1,
                #                                                 1)  # Sample regularization candidates in the range (1 to 200)
                # config.FM_LOSS_WEIGHT = 10 ** np.random.uniform(-1,
                #                                                 1.5)  # Sample regularization candidates in the range (1 to 200)
                # config.ADV_LOSS_WEIGHT = 10 ** np.random.uniform(-1,
                #                                                  1)  # Sample regularization candidates in the range (1 to 200)
            if config.MODEL != 1:
                # config.L1_LOSS_WEIGHT = 10 ** np.random.uniform(-1,
                #                                                 1)  # Sample regularization candidates in the range (1 to 200)
                # config.FM_LOSS_WEIGHT = 10 ** np.random.uniform(-1,
                #                                                 1.5)  # Sample regularization candidates in the range (1 to 200)
                config.STYLE_LOSS_WEIGHT =np.random.uniform(10,400)  # Sample regularization candidates in the range (1 to 200)
                # config.CONTENT_LOSS_WEIGHT = 2 * 10 ** np.random.uniform(0,
                #                                                          2)  # Sample regularization candidates in the range (1 to 200)
                # config.INPAINT_ADV_LOSS_WEIGHT = 10 ** np.random.uniform(-1,
                #                                                  1)  # Sample regularization candidates in the range (1 to 200)
        model = EdgeConnect(config)
        model.load()
        # config.print()
        print('\nEx %d: learning_rate:%f  D_Learning_rate: %f:' % (i, config.LR, config.D2G_LR))
        if TRAIN_LOSS:
            if config.MODEL == 1:
                print('Ex %d: L1:%f  FM: %f  ADV: %f:' % (
                    i, config.L1_LOSS_WEIGHT, config.FM_LOSS_WEIGHT, config.ADV_LOSS_WEIGHT))
            if config.MODEL != 1:
                print('Ex %d: L1:%f  FM: %f  STYLE: %f CONTENT: %f ADV: %f:' % (i, config.L1_LOSS_WEIGHT,
                                                                                config.FM_LOSS_WEIGHT,
                                                                                config.STYLE_LOSS_WEIGHT,
                                                                                config.CONTENT_LOSS_WEIGHT,
                                                                                config.INPAINT_ADV_LOSS_WEIGHT))

        model.train()

    os._exit(0)
