from src.config import Config
from src.utils import create_dir
import numpy as np
import os
# do your thing with the hyper-parameters
from src.process import CLFNet
import math

TRAIN_LOSS = True


def randomTune(config):
    # 'LR': 0.0001,                   # learning rate
    # 'D2G_LR': 0.1,                  # discriminator/generator learning rate ratio
    # 'BETA1': 0.0,                   # adam optimizer beta1
    # 'BETA2': 0.9,                   # adam optimizer beta2
    # 'L1_LOSS_WEIGHT': 1,            # l1 loss weight
    # 'FM_LOSS_WEIGHT': 10,           # feature-matching loss weight
    config.MAX_STEPS = 3200
    config.EVAL_INTERVAL = 80
    config.MAX_EPOCHES = 10
    # config.MAX_STEPS = 3
    config.BATCH_SIZE = 16
    experiments = 50
    for i in range(experiments):
        # sample from a Uniform distribution on a log-scale
        # config.LR = 10 ** np.random.uniform(-3, -5)  # Sample learning rate candidates in the range (0.001 to 0.00001)
        # config.D2G_LR = 10 ** np.random.uniform(-2,
        #                                         0)  # Sample regularization candidates in the range (0.01 to 0.0001)
        # config.LR = 0.0001
        # config.D2G_LR =0.1
        # # config.LR=0.0001

        # config.PATH = './checkpoints/tune_parameters/places2_tune_%d_%f%f_' % (i, config.LR, config.D2G_LR)
        # logdir= config.PATH+('/log_%s_%s' % (config.LR , config.D2G_LR))
        if TRAIN_LOSS:
            # if config.MODEL == 1:
            # config.L1_LOSS_WEIGHT = 10 ** np.random.uniform(-1,1)
            # config.FM_LOSS_WEIGHT = 10 ** np.random.uniform(-1,1.5)
            # config.ADV_LOSS_WEIGHT = 10 ** np.random.uniform(-1,1)
            # config.STYLE_LOSS_WEIGHT = np.random.uniform(0, 300)
            # config.CONTENT_LOSS_WEIGHT = 2 * 10 ** np.random.uniform(0, 2)
            # config.INPAINT_ADV_LOSS_WEIGHT = 10 ** np.random.uniform(-1, 1)
            # if config.MODEL != 1:
                #  Sample regularization candidates in the range (1 to 200)
            max_number=math.log(300, 10)
            config.L1_LOSS_WEIGHT = 10 ** np.random.uniform(-1,max_number)
            config.FM_LOSS_WEIGHT = 10 ** np.random.uniform(-1,max_number)
            # config.GRADIENT_LOSS_WEIGHT= 10 ** np.random.uniform(-1,max_number)
            config.STYLE_LOSS_WEIGHT = 10 ** np.random.uniform(-1,max_number)
            config.CONTENT_LOSS_WEIGHT = 10 ** np.random.uniform(-1,max_number)
            config.INPAINT_ADV_LOSS_WEIGHT = 10 ** np.random.uniform(-1,max_number)

        config.PATH = './checkpoints/tune_parameters/ex%d_L1_%f_ADV_%f_Style_%f_Perc_%f_Grad_%f_FM_%f' % (i,
             config.L1_LOSS_WEIGHT,config.INPAINT_ADV_LOSS_WEIGHT,config.STYLE_LOSS_WEIGHT,
             config.CONTENT_LOSS_WEIGHT,config.GRADIENT_LOSS_WEIGHT,config.FM_LOSS_WEIGHT)
        create_dir(config.PATH)
        model = CLFNet(config)
        model.load()
        # config.print()
        # print('\nEx %d: learning_rate:%f  D_Learning_rate: %f:' % (i, config.LR, config.D2G_LR))
        if TRAIN_LOSS:
            print('Ex %d - L1:%f  FM: %f  STYLE: %f CONTENT: %f ADV: %f: GRAD: %f' % (i, config.L1_LOSS_WEIGHT,
                                                                                config.FM_LOSS_WEIGHT,
                                                                                config.STYLE_LOSS_WEIGHT,
                                                                                config.CONTENT_LOSS_WEIGHT,
                                                                                config.INPAINT_ADV_LOSS_WEIGHT,
                                                                                config.GRADIENT_LOSS_WEIGHT ))
            # if config.MODEL == 1:
            #     print('Ex %d: L1:%f  FM: %f  ADV: %f:' % (
            #         i, config.L1_LOSS_WEIGHT, config.FM_LOSS_WEIGHT, config.ADV_LOSS_WEIGHT))
            # if config.MODEL != 1:
            #     print('Ex %d: L1:%f  FM: %f  STYLE: %f CONTENT: %f ADV: %f:' % (i, config.L1_LOSS_WEIGHT,
            #                                                                     config.FM_LOSS_WEIGHT,
            #                                                                     config.STYLE_LOSS_WEIGHT,
            #                                                                     config.CONTENT_LOSS_WEIGHT,
            #                                                                     config.INPAINT_ADV_LOSS_WEIGHT))
        model.train()
    os._exit(0)
