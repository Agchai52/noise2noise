import numpy as np
import math
import cv2
import os
import matplotlib.pyplot as plt

from skimage.measure import compare_ssim as ssim


def psnr(img1, img2):
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


deblu_root = './test' #_all_deblurred'
sharp_root = './datasets/test_single' #_all'

deblu_list = os.listdir(deblu_root)
sharp_list = os.listdir(sharp_root)
sharp_list = sorted(sharp_list, key=str.lower)
print(sharp_list)

num_imgs = len(deblu_list)
PSNR_all = []
SSIM_all = []

for n, item in enumerate(sharp_list):
    if not item.startswith('.'):

        name_sharp = sharp_list[n]
        name_deblu = 'test_' + name_sharp

        path_deblu = os.path.join(deblu_root, name_deblu)
        path_sharp = os.path.join(sharp_root, name_sharp)

        img_deblu = cv2.imread(path_deblu, cv2.IMREAD_COLOR).astype(np.float)
        img_sharp = cv2.imread(path_sharp, cv2.IMREAD_COLOR).astype(np.float)

        _, ws, _ = img_sharp.shape
        img_sharp = img_sharp[:, int(ws/2):ws]

        img_sharp = img_sharp[:, :, [2, 1, 0]]
        img_deblu = img_deblu[:, :, [2, 1, 0]]

        plt.figure()
        plt.imshow(img_sharp/255)
        plt.title('sharp')

        plt.figure()
        plt.imshow(img_deblu/255)
        plt.title('denoise')
        plt.show()

        psnr_n = psnr(img_deblu, img_sharp)
        ssim_n = ssim(img_deblu/255, img_sharp/255, gaussian_weights=True, multichannel=True, use_sample_covariance=False, sigma=1.5)

        PSNR_all.append(psnr_n)
        SSIM_all.append(ssim_n)
    else:
        continue

PSNR = np.mean(PSNR_all)
SSIM = np.mean(SSIM_all)
#print(PSNR_all)
#print(SSIM_all)
print(PSNR)
print(SSIM)

