import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import cv2
import os
import numpy as np
import skimage
from skimage import io, color, filters, transform, exposure, morphology
from scipy import signal, sparse
import random

def gen_stroke_map(img, kernel_size, num_of_directions=8):
    height = img.shape[0]
    width = img.shape[1]
      
    smooth_im	= cv2.GaussianBlur(img,(3,3),0)
    sobelx = cv2.Sobel(smooth_im,cv2.CV_64F,1,0,ksize=5)
    sobely = cv2.Sobel(smooth_im,cv2.CV_64F,0,1,ksize=5)
    G = np.sqrt(np.square(sobelx) + np.square(sobely))

    basic_ker = np.zeros((kernel_size * 2 + 1, kernel_size * 2 + 1))
    basic_ker[kernel_size + 1,:] = 1

    res_map = np.zeros((height, width, num_of_directions))
    for d in range(num_of_directions):
        ker = transform.rotate(basic_ker, (d * 180) / num_of_directions)
        res_map[:,:, d] = signal.convolve2d(G, ker, mode='same')
        
    max_pixel_indices_map = np.argmax(res_map, axis=2)
    C = np.zeros_like(res_map)
    for d in range(num_of_directions):
        C[:,:,d] = G * (max_pixel_indices_map == d) 

    S_tag_sep = np.zeros_like(C)
    for d in range(num_of_directions):
        ker = transform.rotate(basic_ker, (d * 180) / num_of_directions)
        S_tag_sep[:,:,d] = signal.convolve2d(C[:,:,d], ker, mode='same')
    S_tag = np.sum(S_tag_sep, axis=2)

    S_tag_normalized = (S_tag - np.min(S_tag.ravel())) / (np.max(S_tag.ravel()) - np.min(S_tag.ravel()))
    S = 1 - S_tag_normalized
    # norm_image = cv2.normalize(S, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
    # norm_image = norm_image.astype(np.uint8)
    # th, S = cv2.threshold(norm_image, 0, 255, cv2.THRESH_OTSU)
    return S


def gen_tone_map(img, w_group=0):

    w_mat = np.array([[11, 37, 52],
                     [29, 29, 42],
                     [2, 22, 76]])
    w = w_mat[w_group,:]

    u_b = 225
    u_a = 105
    sigma_b = 9
    mu_d = 90
    sigma_d = 11
    
    num_pixel_vals = 256
    p = np.zeros(num_pixel_vals)
    for v in range(num_pixel_vals):
        p1 = (1 / sigma_b) * np.exp(-(255 - v) / sigma_b)
        if (u_a <= v <= u_b):
            p2 = 1 / (u_b - u_a)
        else:
            p2 = 0
        p3 = (1 / np.sqrt(2 * np.pi * sigma_d)) * np.exp( (-np.square(v - mu_d)) / (2 * np.square(sigma_d)) )
        p[v] = w[0] * p1 + w[1] * p2 + w[2] * p3 * 0.01
    # normalize the histogram:
    p_normalized = p / np.sum(p)
    # calculate the CDF of the desired histogram:
    P = np.cumsum(p_normalized)
    # calculate the original histogram:
    h = exposure.histogram(img, nbins=256)
    # CDF of original:
    H = np.cumsum(h / np.sum(h))
    # histogram matching:
    lut = np.zeros_like(p)
    for v in range(num_pixel_vals):
        # find the closest value:
        dist = np.abs(P - H[v])
        argmin_dist = np.argmin(dist)
        lut[v] = argmin_dist
    lut_normalized = lut / num_pixel_vals
    J = lut_normalized[(255 * img).astype(np.int)]
    # smooth:
    J_smoothed = filters.gaussian(J, sigma=np.sqrt(2))
    return J_smoothed

def gen_pencil_texture(img, H, J):
    # define the regularization parameter:
    lamda = 0.2
    height = img.shape[0]
    width = img.shape[1]

    H_res = cv2.resize(H, (width, height), interpolation=cv2.INTER_CUBIC)
    H_res_reshaped = np.reshape(H_res, (height * width, 1))
    logH = np.log(H_res_reshaped)
    
    J_res = cv2.resize(J, (width, height), interpolation=cv2.INTER_CUBIC)
    J_res_reshaped = np.reshape(J_res, (height * width, 1))
    logJ = np.log(J_res_reshaped)
    
    # In order to use Conjugate Gradient method we need to prepare some sparse matrices:
    logH_sparse = sparse.spdiags(logH.ravel(), 0, height*width, height*width) # 0 - from main diagonal
    e = np.ones((height * width, 1))
    ee = np.concatenate((-e,e), axis=1)
    diags_x = [0, height*width]
    diags_y = [0, 1]
    dx = sparse.spdiags(ee.T, diags_x, height*width, height*width)
    dy = sparse.spdiags(ee.T, diags_y, height*width, height*width)
    
    # Compute matrix X and b: (to solve Ax = b)
    A = lamda * ((dx @ dx.T) + (dy @ dy.T)) + logH_sparse.T @ logH_sparse
    b = logH_sparse.T @ logJ
    
    # Conjugate Gradient
    beta = sparse.linalg.cg(A, b, tol=1e-6, maxiter=60)
    beta_reshaped = np.reshape(beta[0], (height, width))
    T = np.power(H_res, beta_reshaped)
    return T

def get_sketched_image(img):
    #img = cv2.imread(image_path,1)
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    img_y_ch = img_yuv[:,:,0]
    pencil_file = "./pencil_styles/" + random.choice(os.listdir("./pencil_styles"))
    pencil_tex = io.imread(pencil_file, as_gray=True)
    img_stroke_map = gen_stroke_map(img_y_ch, 3)
    img_tone_map = gen_tone_map(img_y_ch, w_group=0)
    img_tex_map = gen_pencil_texture(img_y_ch, pencil_tex, img_tone_map)
    sketched_image = np.multiply(img_stroke_map, img_tex_map)
    sketched_image = cv2.normalize(sketched_image, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
    sketched_image = sketched_image.astype(np.uint8)
    return sketched_image