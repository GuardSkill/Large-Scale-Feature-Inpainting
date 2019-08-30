import cv2
import random
import os
from PIL import Image, ImageOps
from glob import glob
import numpy as np
from src.utils import create_dir
import scipy.signal

img_dir = 'E:\PyProjects\edge-connect-master\examples\places2\images'
mask_dir = 'E:\PyProjects\edge-connect-master\examples\places2\masks'
ext = {'.JPG', '.JPEG', '.PNG', '.TIF', 'TIFF'}
img_paths = []
mask_paths = []

i = 1

for file in glob('{:s}/*'.format(img_dir)):
    if os.path.splitext(file)[1].upper() in ext:
        img_paths.append(file)

for file in glob('{:s}/*'.format(mask_dir)):
    if os.path.splitext(file)[1].upper() in ext:
        mask_paths.append(file)

N_img = len(img_paths)
N_mask = len(mask_paths)
N_mini = N_mask if N_img > N_mask else N_img

ipn_kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])  # kernel for inpaint_nans
def inpaint_nans(img):
    nans =img==0
    while np.sum(nans) > 0:
        img[nans] = 0
        vNeighbors = scipy.signal.convolve2d((nans == False), ipn_kernel, mode='same', boundary='fill')
        im2 = scipy.signal.convolve2d(img, ipn_kernel, mode='same', boundary='fill')
        im2[vNeighbors > 0] = im2[vNeighbors > 0] / vNeighbors[vNeighbors > 0]
        im2[vNeighbors == 0] = np.nan
        im2[(nans == False)] = img[(nans == False)]
        im = im2
        nans = np.isnan(img)
    return img


for i in range(N_mini):
    img = cv2.imread(img_paths[i])
    mask = cv2.imread(mask_paths[i])
    # if img.shape[2]==3:
    #     m=cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    # dst = cv2.inpaint(img, m, 3, cv2.INPAINT_TELEA)
    img[mask != 0] = 0

    inpaint_nans(img)
    cv2.imshow('inpaint_first', img)

    cv2.imshow('inpaint_first', dst)
    dst = cv2.medianBlur(dst, 5)
    cv2.imshow('median', dst)
    dst = dst * mask + (255 - mask) * img
    cv2.imshow('final', dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite(img_paths[i], dst)


