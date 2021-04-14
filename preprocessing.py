from skimage import io, color, filters, transform, exposure, morphology
from scipy import signal, sparse
from matplotlib import pyplot as plt
import cv2
import os
import skimage
import numpy as np
import matplotlib
import random


def showImage(path, opt=1):
    img=cv2.imread(path,opt) # BGR colors
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # to RGB
    plt.imshow(rgb,interpolation='nearest')
    plt.show()
    
def find_edges(img):
    im = np.array(img) 
    im = cv2.GaussianBlur(im,(3,3),0)
    im = cv2.Canny(im,100,200)
    img = Image.fromarray(im)
    return img.point(lambda p: p > 128 and 255)  

def sobel (img):
    Imgx = cv2.Sobel(img,cv2.CV_8U,0,1,ksize=3)
    Imgy = cv2.Sobel(img,cv2.CV_8U,1,0,ksize=3)
    return cv2.bitwise_or(Imgx, Imgy)

def sketch(img):	
    #Premiere fonction de sketch qui applique uniquement une detection
    #de contours sur une image en niveau de gris uniformisée.
    #Cette detection se fait par combinaison de sobel.
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img	= cv2.GaussianBlur(img,(3,3),0)
    invImg = 255-img
    edgImg = sobel(img)
    edgImgInv	= sobel(invImg)
    edgImg = cv2.addWeighted(edgImg,1,edgImgInv,1,0)	
    sketchImg	= 255-edgImg
    return sketchImg

def sketch_binarized(img):	
    #fonction de sketch qui applique uniquement une detection
    #de contours sur une image en niveau de gris uniformisée et ensuite binarisée.
    #Cette detection se fait par combinaison de sobel.
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img	= cv2.GaussianBlur(img,(3,3),0)
    invImg = 255-img
    edgImg = sobel(img)
    edgImgInv	= sobel(invImg)
    edgImg = cv2.addWeighted(edgImg,1,edgImgInv,1,0)	
    sketchImg	= 255-edgImg
    th, im_gray_th_otsu = cv2.threshold(sketchImg, 0, 255, cv2.THRESH_OTSU)
    return im_gray_th_otsu

def sketch_denoized_binarized(img):	
    #Cette fonction se base sur la fonction sketch_binarized mais propose en plus
    #d'appliquer une opération de débruitage de opencv
    img = cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img	= cv2.GaussianBlur(img,(3,3),0)
    invImg = 255-img
    edgImg = sobel(img)
    edgImgInv	= sobel(invImg)
    edgImg = cv2.addWeighted(edgImg,1,edgImgInv,1,0)	
    sketchImg	= 255-edgImg
    th, im_gray_th_otsu = cv2.threshold(sketchImg, 0, 255, cv2.THRESH_OTSU)
    #test to remove noise from the flower background
    #remove_background(im_gray_th_otsu)
    return im_gray_th_otsu

def sketch_denoized_binarized_unified(img):	
    #Cette fonction se base sur la fonction sketch_denoized_binarized mais propose en plus
    #d'appliquer une opération de suprression de zones de petite taille considérée comme du 
    #bruit après detection de contours.
    img = cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img	= cv2.GaussianBlur(img,(3,3),0)
    invImg = 255-img
    edgImg = sobel(img)
    edgImgInv	= sobel(invImg)
    edgImg = cv2.addWeighted(edgImg,1,edgImgInv,1,0)	
    cv2_imshow(edgImg)
    thresh = cv2.threshold(edgImg, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    cnts = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 5500:
            cv2.drawContours(thresh, [c], -1, (0,0,0), -1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    close = 255 - cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    return close

def sketch_canny(img):
    #Cett fonction permet de produire des images sous format de sketch
    #par application de canny edge detection.
    ratio = 3
    kernel_size = 3
    low_threshold = 70
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur	= cv2.GaussianBlur(img,(3,3),0)
    detected_edges = cv2.Canny(img_blur, low_threshold, low_threshold*ratio, kernel_size)
    new_img = 255-detected_edges
    return new_img

def sketch_canny_denoised(img):
    #Cette fonction se base sur sketch_canny mais appliqué après réduction du bruit 
    #grace a une fonction opencv
    img = cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)
    ratio = 3
    kernel_size = 3
    low_threshold = 70
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur	= cv2.GaussianBlur(img,(3,3),0)
    detected_edges = cv2.Canny(img_blur, low_threshold, low_threshold*ratio, kernel_size)
    new_img = 255-detected_edges
    return new_img

def sketch_canny_denoised_small_removed(img):
    #Cette fonction se base sur sketch_canny_denoised mais propose 
    #de supprimer les petits objets détéctés une fois la detection
    #de contours réalisée
    img = cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)
    ratio = 3
    kernel_size = 3
    low_threshold = 70
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur	= cv2.GaussianBlur(img,(3,3),0)
    detected_edges = cv2.Canny(img_blur, low_threshold, low_threshold*ratio, kernel_size)
    new_img = 255-detected_edges
    processed = morphology.remove_small_objects(new_img.astype(bool), min_size=1, connectivity=5).astype(int)
    mask_x, mask_y = np.where(processed == 0)
    new_img[mask_x, mask_y] = 255
    return new_img

def separate_for_back(img):
  #Cette fonction permet de supprimer le fond de l'image grace à la fonction 
  #grabcut d'opencv
  mask = np.zeros(img.shape[:2],np.uint8)
  bgdModel = np.zeros((1,65),np.float64)
  fgdModel = np.zeros((1,65),np.float64)
  rect = (0,0,img.shape[0],img.shape[1])
  cv2.grabCut(img,mask,rect,bgdModel,fgdModel,2,cv2.GC_INIT_WITH_RECT)
  mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
  img = img*mask2[:,:,np.newaxis]
  return img

def get_histograms(img):
  #Fonction utilitaire
  color = ('b','g','r')
  hists = []
  for i,col in enumerate(color):
      histr = cv2.calcHist([img],[i],None,[256],[0,256])
      hists.append(histr)
      plt.plot(histr,color = col)
      plt.xlim([0,256])
  plt.show()
  return hists

def split_image(img):
  #Fonction utilitaire
  b,g,r = cv2.split(img)
  return b,g,r

def get_hsv_histogram(img):
  #Fonction utilitaire
  hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
  hist = cv2.calcHist( [hsv], [0, 1], None, [180, 256], [0, 180, 0, 256] )
  plt.imshow(hist,interpolation = 'nearest')
  plt.show()
    
def test_simple_functions(image_path):
    #Fonction pour tester les fonctions de base. A utiliser sur google colab en
    #important from google.colab.patches import cv2_imshow
    #pour pouvoir utiliser la fonction cv2_imshow()
    img = cv2.imread(image_path,1)
    cv2_imshow(img)
    print("original image")
    foreground = separate_for_back(img)
    cv2_imshow(foreground)
    print("background separated from foreground")
    sketchImg	= sketch(img)	
    cv2_imshow(sketchImg)
    print("sketched image with sobel")
    sketchImgBin	= sketch_binarized(img)	
    cv2_imshow(sketchImgBin)
    print("sketched image with sobel binarized")
    sketchImgDenoiBin	= sketch_denoized_binarized(img)	
    cv2_imshow(sketchImgDenoiBin)
    print("sketched image with sobel binarized and denoized")
    sketchImgDenoiBinUnif	= sketch_denoized_binarized_unified(img)	
    cv2_imshow(sketchImgDenoiBinUnif)
    print("sketched image with sobel binarized and denoized and shape unified")
    sketchCanny	= sketch_canny(img)	
    cv2_imshow(sketchCanny)
    print("sketched image with canny edge detector")
    sketchCannyDenoi = sketch_canny_denoised(img)	
    cv2_imshow(sketchCannyDenoi)
    print("sketched image with canny edge detector denoized")
    sketchCannyDenoiForeg = sketch_canny_denoised(foreground)
    cv2_imshow(sketchCannyDenoiForeg)
    print("sketched image with canny edge detector denoized background separated from foreground")
    sketchCannyDenoiSmallRm	= sketch_canny_denoised_small_removed(img)	
    cv2_imshow(sketchCannyDenoiSmallRm)
    print("sketched image with canny edge detector denoized and small objects removed")
    
def gen_stroke_map(img, kernel_size, num_of_directions=8):
    #Permet de créer une stroke map de l'image
    #la taille du noyau et le nombre de directions 
    #permet de définir les elements de convolution

    #Tout d'abord on applique les opérations de base pour obtenir 
    #une image uniformisée
    height = img.shape[0]
    width = img.shape[1]
    
    smooth_im	= cv2.GaussianBlur(img,(3,3),0)
    sobelx = cv2.Sobel(smooth_im,cv2.CV_64F,1,0,ksize=5)
    sobely = cv2.Sobel(smooth_im,cv2.CV_64F,0,1,ksize=5)
    G = np.sqrt(np.square(sobelx) + np.square(sobely))

    #Nous créeons un segment de base horizontal au centre du noyau
    basic_ker = np.zeros((kernel_size * 2 + 1, kernel_size * 2 + 1))
    basic_ker[kernel_size + 1,:] = 1

    #On définit un ensemble de directions "de base" par des segments
    #On crée ensuite un mapping des réponses selon chaque direction 
    #Puis on sélectionne la direction ou la réponse est la plus forte
    res_map = np.zeros((height, width, num_of_directions))
    for d in range(num_of_directions):
        ker = transform.rotate(basic_ker, (d * 180) / num_of_directions)
        res_map[:,:, d] = signal.convolve2d(G, ker, mode='same')
    max_pixel_indices_map = np.argmax(res_map, axis=2)
    
    #On calcule ici la classification map
    C = np.zeros_like(res_map)
    for d in range(num_of_directions):
        C[:,:,d] = G * (max_pixel_indices_map == d) 

    #On crée ici les lignes selon notre classification à chaque pixel par une
    #opération de convolution
    S_tag_sep = np.zeros_like(C)
    for d in range(num_of_directions):
        ker = transform.rotate(basic_ker, (d * 180) / num_of_directions)
        S_tag_sep[:,:,d] = signal.convolve2d(C[:,:,d], ker, mode='same')
    S_tag = np.sum(S_tag_sep, axis=2)

    #On s'assure que les valeurs sont dans [0,1]
    S_tag_normalized = (S_tag - np.min(S_tag.ravel())) / (np.max(S_tag.ravel()) - np.min(S_tag.ravel()))
    S = 1 - S_tag_normalized
    # norm_image = cv2.normalize(S, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
    # norm_image = norm_image.astype(np.uint8)
    # th, S = cv2.threshold(norm_image, 0, 255, cv2.THRESH_OTSU)
    return S


def gen_tone_map(img, w_group=0):
    #Cette fonction permet de créer une tone map celon une distribution
    #cible de ton. Nous utilisons ici les valeurs recommandées dans l'article
    w_mat = np.array([[11, 37, 52],
                     [29, 29, 42],
                     [2, 22, 76]])
    w = w_mat[w_group,:]
    #Nous définissons ici les paramètres issus de l'article pour
    #assigner à chaque pixel un ton
    u_b = 225
    u_a = 105
    sigma_b = 9
    mu_d = 90
    sigma_d = 11
    num_pixel_vals = 256

    #Nouq allons maintenant céer le nouvel histogramme
    p = np.zeros(num_pixel_vals)
    for v in range(num_pixel_vals):
        p1 = (1 / sigma_b) * np.exp(-(255 - v) / sigma_b)
        if (u_a <= v <= u_b):
            p2 = 1 / (u_b - u_a)
        else:
            p2 = 0
        p3 = (1 / np.sqrt(2 * np.pi * sigma_d)) * np.exp( (-np.square(v - mu_d)) / (2 * np.square(sigma_d)) )
        p[v] = w[0] * p1 + w[1] * p2 + w[2] * p3 * 0.01
    
    #Nous normalisons l'histogramme
    p_normalized = p / np.sum(p)

    #On calcule le CDF de l'histogramme voulu
    P = np.cumsum(p_normalized)

    #On calcule l'histogramme réel de notre image d'origine
    h, bins = np.histogram(img.ravel(),256,[0,256])

    #On calcule le CDF de l'histogramme d'origine
    H = np.cumsum(h / np.sum(h))

    #Nous allons maintenant associer à chaque valeur de l'histogramme
    lut = np.zeros_like(p)
    for v in range(num_pixel_vals):
        #On trouve la valeur la plus proche et on l'assigne a notre nouvel histogramme
        dist = np.abs(P - H[v])
        argmin_dist = np.argmin(dist)
        lut[v] = argmin_dist

    lut_normalized = lut / num_pixel_vals
    J = lut_normalized[(255 * img).astype(np.int)]
    #On uniformise l'image grace à un filtre gaussien
    J_smoothed = filters.gaussian(J, sigma=np.sqrt(2))
    return J_smoothed

def gen_pencil_texture(img, H, J):
    #On définit les paramètres de regularisation
    lamda = 0.2
    height = img.shape[0]
    width = img.shape[1]

    #On ajuste la taille de notre image de style de dessin pour correspondre à notre
    #image d'origine
    H_res = cv2.resize(H, (width, height), interpolation=cv2.INTER_CUBIC)
    H_res_reshaped = np.reshape(H_res, (height * width, 1))
    logH = np.log(H_res_reshaped)

    #On s'assure que notre tone map est au bon format
    J_res = cv2.resize(J, (width, height), interpolation=cv2.INTER_CUBIC)
    J_res_reshaped = np.reshape(J_res, (height * width, 1))
    logJ = np.log(J_res_reshaped)
    
    #On initialise les matrices pour effecter le calcul du gradient
    logH_sparse = sparse.spdiags(logH.ravel(), 0, height*width, height*width)
    e = np.ones((height * width, 1))
    ee = np.concatenate((-e,e), axis=1)
    diags_x = [0, height*width]
    diags_y = [0, 1]
    dx = sparse.spdiags(ee.T, diags_x, height*width, height*width)
    dy = sparse.spdiags(ee.T, diags_y, height*width, height*width)
    
    #On effectue le caclule de la matrice X et b pour résoudre Ax=b
    A = lamda * ((dx @ dx.T) + (dy @ dy.T)) + logH_sparse.T @ logH_sparse
    b = logH_sparse.T @ logJ
    
    #On conjuge le gradient avant d'ajuster le resultat et de retourner
    #l'image finale texturée
    beta = sparse.linalg.cg(A, b, tol=1e-6, maxiter=60)
    beta_reshaped = np.reshape(beta[0], (height, width))
    T = np.power(H_res, beta_reshaped)
    return T


def get_sketched_image(img):
    # img = cv2.imread(image_path,1)
    #On reecupere l'image et on transforme en format YUV
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    #Dans la première fonction gen_stroke_map, on travaille uniquement sur la brightness
    img_y_ch = img_yuv[:, :, 0]
    #On selectionne aléatoirement un type de style de dessin dans le dossier associé
    pencil_file = "./pencil_styles/" + random.choice(os.listdir("./pencil_styles"))
    pencil_tex = io.imread(pencil_file, as_gray=True)
    #on applique successivement les 3 étapes avant de combiner le résultat
    img_stroke_map = gen_stroke_map(img_y_ch, 3)
    img_tone_map = gen_tone_map(img_y_ch, w_group=0)
    img_tex_map = gen_pencil_texture(img_y_ch, pencil_tex, img_tone_map)
    sketched_image = np.multiply(img_stroke_map, img_tex_map)
    sketched_image = cv2.normalize(sketched_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    sketched_image = sketched_image.astype(np.uint8)
    return sketched_image
