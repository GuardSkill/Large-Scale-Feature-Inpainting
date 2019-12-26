import os
import argparse
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, help='path to the dataset')
parser.add_argument('--output', type=str, help='path to the file list')
parser.add_argument('--celeba', type=str, default=None, help='path to the file list')
args = parser.parse_args()

ext = {'.JPG', '.JPEG', '.PNG', '.TIF', 'TIFF'}
if args.celeba == None:
    images = []
    for root, dirs, files in os.walk(args.path):
        print('loading ' + root)
        for file in files:
            if os.path.splitext(file)[1].upper() in ext:
                images.append(os.path.join(root, file))

    images = sorted(images)
    np.savetxt(args.output, images, fmt='%s')
else:
    csv_file=str(args.celeba)
    root_dir=str(args.path)
    df = pd.read_csv(csv_file,sep =' ',header=None)
    traindf=df[df[1]==0]
    images=[]
    for train_file in traindf[0]:
        images.append(os.path.join(root_dir, train_file))

    np.savetxt('./datasets/celeba_train.flist', images, fmt='%s')

    images = []
    evaldf = df[df[1] == 1]
    for eval_file in evaldf[0]:
        images.append(os.path.join(root_dir, eval_file))

    np.savetxt('./datasets/celeba_eval.flist', images, fmt='%s')

    images = []
    traindf = df[df[1] == 2]
    for train_file in traindf[0]:
        images.append(os.path.join(root_dir, train_file))

    np.savetxt('./datasets/celeba_test.flist', images, fmt='%s')