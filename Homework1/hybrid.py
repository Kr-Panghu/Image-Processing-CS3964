# import sys
# import cv2
# import numpy as np

# def cross_correlation_2d(img, kernel):
#     '''Given a kernel of arbitrary m x n dimensions, with both m and n being
#     odd, compute the cross correlation of the given image with the given
#     kernel, such that the output is of the same dimensions as the image and that
#     you assume the pixels out of the bounds of the image to be zero. Note that
#     you need to apply the kernel to each channel separately, if the given image
#     is an RGB image.

#     Inputs:
#         img:    Either an RGB image (height x width x 3) or a grayscale image
#                 (height x width) as a numpy array.
#         kernel: A 2D numpy array (m x n), with m and n both odd (but may not be
#                 equal).

#     Output:
#         Return an image of the same dimensions as the input image (same width,
#         height and the number of color channels)
#     '''
    
#     (m, n) = kernel.shape # Kernel是(m, n)维矩阵
#     (k1, k2) = (m-1)/2, (n-1)/2
#     g = img
#     w, h, nchan = img.shape
    
#     for i in range(w):
#         for j in range(h): 
#             # G[i, j] = Sigma Sigma H[u,v]F[i+u,j+v]   
#             for u in range(-k1, k1+1):
#                 for v in range(-k2, k2+1):
#                     g[i,j] += kernel[u,v] * img[i+u, j+v]
    
    
#     # TODO-BLOCK-BEGIN
#     raise Exception("TODO in hybrid.py not implemented")
#     # TODO-BLOCK-END

# def convolve_2d(img, kernel):
#     '''Use cross_correlation_2d() to carry out a 2D convolution.

#     Inputs:
#         img:    Either an RGB image (height x width x 3) or a grayscale image
#                 (height x width) as a numpy array.
#         kernel: A 2D numpy array (m x n), with m and n both odd (but may not be
#                 equal).

#     Output:
#         Return an image of the same dimensions as the input image (same width,
#         height and the number of color channels)
#     '''
#     (m, n) = kernel.shape # Kernel是(m, n)维矩阵
#     (k1, k2) = (m-1)/2, (n-1)/2
#     g = img
#     w, h, nchan = img.shape
    
#     for i in range(w):
#         for j in range(h): 
#             # G[i, j] = Sigma Sigma H[u,v]F[i+u,j+v]   
#             for u in range(-k1, k1+1):
#                 for v in range(-k2, k2+1):
#                     g[i,j] += kernel[u,v] * img[i+u, j+v]
    
#     # TODO-BLOCK-BEGIN
#     raise Exception("TODO in hybrid.py not implemented")
#     # TODO-BLOCK-END

# def gaussian_blur_kernel_2d(sigma, height, width):
#     '''Return a Gaussian blur kernel of the given dimensions and with the given
#     sigma. Note that width and height are different.

#     Input:
#         sigma:  The parameter that controls the radius of the Gaussian blur.
#                 Note that, in our case, it is a circular Gaussian (symmetric
#                 across height and width).
#         width:  The width of the kernel.
#         height: The height of the kernel.

#     Output:
#         Return a kernel of dimensions height x width such that convolving it
#         with an image results in a Gaussian-blurred image.
#     '''
#     # TODO-BLOCK-BEGIN
#     raise Exception("TODO in hybrid.py not implemented")
#     # TODO-BLOCK-END

# def low_pass(img, sigma, size):
#     '''Filter the image as if its filtered with a low pass filter of the given
#     sigma and a square kernel of the given size. A low pass filter supresses
#     the higher frequency components (finer details) of the image.

#     Output:
#         Return an image of the same dimensions as the input image (same width,
#         height and the number of color channels)
#     '''
#     # TODO-BLOCK-BEGIN
#     raise Exception("TODO in hybrid.py not implemented")
#     # TODO-BLOCK-END

# def high_pass(img, sigma, size):
#     '''Filter the image as if its filtered with a high pass filter of the given
#     sigma and a square kernel of the given size. A high pass filter suppresses
#     the lower frequency components (coarse details) of the image.

#     Output:
#         Return an image of the same dimensions as the input image (same width,
#         height and the number of color channels)
#     '''
#     # TODO-BLOCK-BEGIN
#     raise Exception("TODO in hybrid.py not implemented")
#     # TODO-BLOCK-END

# def create_hybrid_image(img1, img2, sigma1, size1, high_low1, sigma2, size2,
#         high_low2, mixin_ratio, scale_factor):
#     '''This function adds two images to create a hybrid image, based on
#     parameters specified by the user.'''
#     high_low1 = high_low1.lower()
#     high_low2 = high_low2.lower()

#     if img1.dtype == np.uint8:
#         img1 = img1.astype(np.float32) / 255.0
#         img2 = img2.astype(np.float32) / 255.0

#     if high_low1 == 'low':
#         img1 = low_pass(img1, sigma1, size1)
#     else:
#         img1 = high_pass(img1, sigma1, size1)

#     if high_low2 == 'low':
#         img2 = low_pass(img2, sigma2, size2)
#     else:
#         img2 = high_pass(img2, sigma2, size2)

#     img1 *=  (1 - mixin_ratio)
#     img2 *= mixin_ratio
#     hybrid_img = (img1 + img2) * scale_factor
#     return (hybrid_img * 255).clip(0, 255).astype(np.uint8)





# =====================================






import sys
import cv2
import numpy as np
import os

# def cross_correlation_2d(img, kernel):
#     '''Given a kernel of arbitrary m x n dimensions, with both m and n being
#     odd, compute the cross correlation of the given image with the given
#     kernel, such that the output is of the same dimensions as the image and that
#     you assume the pixels out of the bounds of the image to be zero. Note that
#     you need to apply the kernel to each channel separately, if the given image
#     is an RGB image.

#     Inputs:
#         img:    Either an RGB image (height x width x 3) or a grayscale image
#                 (height x width) as a numpy array.
#         kernel: A 2D numpy array (m x n), with m and n both odd (but may not be
#                 equal).'''
    
#     u,v = kernel.shape
#     new_img = np.zeros(img.shape)

#      #define padding size
#     u_pad = (u - 1) / 2
#     v_pad = (v - 1) / 2

#     # if picture is rgb
#     if len(img.shape) > 2:
#         x, y, colors = img.shape

#         # create padded image
#         padded_img = np.pad(img, pad_width=((u_pad, u_pad,), (v_pad, v_pad), (0,0)), mode = 'constant', constant_values = 0)

#         #loop
#         for i in range(x):
#             for j in range(y):
#                 for color in range(colors):
#                     new_img[i, j, color] = np.sum(kernel * padded_img[i:i + u, j:j + v, color])

#     else:
#         x,y = img.shape

#         # created padded img
#         padded_img = np.pad(img, pad_width=((u_pad, u_pad,), (v_pad, v_pad)), mode = 'constant', constant_values = 0)

#         #loop
#         for i in range(x):
#             for j in range(y):
#                 new_img[i, j] = np.sum(kernel * padded_img[i:i + u, j:j + v])

#     return new_img

#     '''Output:
#         Return an image of the same dimensions as the input image (same width,
#         height and the number of color channels)'''

def cross_correlation_2d(img, kernel):
    '''Given a kernel of arbitrary m x n dimensions, with both m and n being
    odd, compute the cross correlation of the given image with the given
    kernel, such that the output is of the same dimensions as the image and that
    you assume the pixels out of the bounds of the image to be zero. Note that
    you need to apply the kernel to each channel separately, if the given image
    is an RGB image.

    Inputs:
        img:    Either an RGB image (height x width x 3) or a grayscale image
                (height x width) as a numpy array.
        kernel: A 2D numpy array (m x n), with m and n both odd (but may not be
                equal).

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''
    m, n = kernel.shape
    
    print(kernel.shape)
    
    km, kn = m // 2, n // 2
    mn = m * n
    kernel = np.ravel(kernel) # ravel函数将kernel矩阵按照行优先展平, 同时空间复用

    newImg = np.empty(img.shape)

    # print("Test for variable type", img.shape)  # ==> (361, 400, 3)

    if img.ndim == 3:
        h, w, chans = img.shape
    else: 
        # Make img always end up with 3 dimensions
        chans = 1
        h, w = img.shape
        img = img[:, :, np.newaxis]

    paddedWorkspace = np.zeros((h + m - 1, w + n - 1, chans), dtype=img.dtype)
    # Pad the original image in our reusable workspace
    paddedWorkspace[km : km + h, kn : kn + w] = img

    for x in range(w):
        for y in range(h):
            # Extract the area from the workspace we are cross-correlating
            # and compute dot product
            sliced = np.reshape(paddedWorkspace[y : y + m, x : x + n], (mn, chans))
            newImg[y, x] = np.dot(kernel, sliced)

    return newImg

def convolve_2d(img, kernel):
    '''Use cross_correlation_2d() to carry out a 2D convolution.

    Inputs:
        img:    Either an RGB image (height x width x 3) or a grayscale image
                (height x width) as a numpy array.
        kernel: A 2D numpy array (m x n), with m and n both odd (but may not be
                equal).

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''
    flipped_kernel = np.flip(kernel)

    conv_image = cross_correlation_2d(img, flipped_kernel)

    return conv_image


def gaussian_blur_kernel_2d(sigma, height, width):
    '''Return a Gaussian blur kernel of the given dimensions and with the given
    sigma. Note that width and height are different.

    Input:
        sigma:  The parameter that controls the radius of the Gaussian blur.
                Note that, in our case, it is a circular Gaussian (symmetric
                across height and width).
        width:  The width of the kernel.
        height: The height of the kernel.

    Output:
        Return a kernel of dimensions height x width such that convolving it
        with an image results in a Gaussian-blurred image.
    '''

    gauss_kernel = np.zeros((height, width))
    h = np.linspace(-height / 2 + 1, height / 2, height)
    w = np.linspace(-width / 2 + 1, width / 2, width)

    for x, x1 in enumerate(h):
        for y, y1 in enumerate(w):
            gauss_kernel[x,y] = 1 / (2 * np.pi * (sigma ** 2)) * np.exp(-(x1 ** 2 + y1 ** 2)/(2 * (sigma ** 2)))
    
    norm_gauss = gauss_kernel * 1 / np.sum(gauss_kernel)
    
    return norm_gauss


def low_pass(img, sigma, size):
    '''Filter the image as if its filtered with a low pass filter of the given
    sigma and a square kernel of the given size. A low pass filter supresses
    the higher frequency components (finer details) of the image.

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''

    return convolve_2d(img, gaussian_blur_kernel_2d(sigma, size, size))


def high_pass(img, sigma, size):
    '''Filter the image as if its filtered with a high pass filter of the given
    sigma and a square kernel of the given size. A high pass filter suppresses
    the lower frequency components (coarse details) of the image.

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''

    return img - low_pass(img, sigma, size)


def create_hybrid_image(img1, img2, sigma1, size1, high_low1, sigma2, size2,
        high_low2, mixin_ratio, scale_factor):
    '''This function adds two images to create a hybrid image, based on
    parameters specified by the user.'''
    high_low1 = high_low1.lower()
    high_low2 = high_low2.lower()

    if img1.dtype == np.uint8:
        img1 = img1.astype(np.float32) / 255.0
        img2 = img2.astype(np.float32) / 255.0

    if high_low1 == 'low':
        img1 = low_pass(img1, sigma1, size1)
    else:
        img1 = high_pass(img1, sigma1, size1)

    if high_low2 == 'low':
        img2 = low_pass(img2, sigma2, size2)
    else:
        img2 = high_pass(img2, sigma2, size2)

    img1 *=  (1 - mixin_ratio)
    img2 *= mixin_ratio
    hybrid_img = (img1 + img2) * scale_factor
    return (hybrid_img * 255).clip(0, 255).astype(np.uint8)