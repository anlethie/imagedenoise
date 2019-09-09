import numpy as np
import math
import dataset
import matplotlib.pyplot as plt


def show_sample_image(sampl_n, sampl_t):
    # Parameters are tensor type
    
    orig_noise = dataset.ToImage(sampl_n)
    orig_truth = dataset.ToImage(sampl_t)
    substract_1 = dataset.ToImage(sampl_n-sampl_t)

    plt.figure(figsize=(24, 24))

    ax = plt.subplot(2, 2, 1)
    plt.imshow(orig_noise)
    ax.set_title('Original Noisy Image.')

    ax = plt.subplot(2, 2, 2)
    plt.imshow(orig_truth)
    ax.set_title('Original Clean Image')

    ax = plt.subplot(2, 2, 3)
    plt.imshow(substract_1)
    ax.set_title('Noise Map')

    plt.show()


def show_image_comparision(sampl_n, sampl_t, sampl_o):
    # Parameters are tensor type
    
    orig_noise = dataset.ToImage(sampl_n)
    denoised = dataset.ToImage(sampl_o)
    orig_truth = dataset.ToImage(sampl_t)
    substract_1 = dataset.ToImage(sampl_n-sampl_t)
    substract_2 = dataset.ToImage(sampl_n-sampl_o)

    plt.figure(figsize=(24, 24))

    ax = plt.subplot(3, 2, 1)
    plt.imshow(orig_noise)
    ax.set_title('Original Noisy Image')

    ax = plt.subplot(3, 2, 2)
    plt.imshow(denoised)
    ax.set_title('Denoised Image')

    ax = plt.subplot(3, 2, 3)
    plt.imshow(orig_truth)
    ax.set_title('Original Clean Image')

    ax = plt.subplot(3, 2, 4)
    plt.imshow(denoised)
    ax.set_title('Denoised Image')

    ax = plt.subplot(3, 2, 5)
    plt.imshow(substract_1)
    ax.set_title('Original Noisy Image subtract Original Clean Image')

    ax = plt.subplot(3, 2, 6)
    plt.imshow(substract_2)
    ax.set_title('Original Noisy Image subtract Denoised Image')

    plt.show()
    


# Code from https://github.com/aizvorski/video-quality/blob/master/psnr.py
def psnr(img1, img2):
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
