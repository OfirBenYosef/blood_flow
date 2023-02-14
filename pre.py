from socket import INADDR_BROADCAST
from skimage import data
from skimage import color
import os
from vidstab import VidStab
from skimage.filters import meijering, sato, frangi, hessian, threshold_otsu
from skimage.filters import gaussian, laplace, sobel, roberts
from skimage.segmentation import clear_border
import matplotlib.pyplot as plt
import numpy as np
import cv2
from skimage import exposure
from skimage.measure import label, regionprops, regionprops_table
from skimage.segmentation import active_contour
from skimage.color import rgb2gray
from skimage import filters
from PIL import Image
from skimage.morphology import disk, flood_fill
from scipy import ndimage as ndi
import plotly
import plotly.express as px
import time
import plotly.graph_objects as go
from skimage.color import label2rgb
from skimage import (
    color, feature, filters, measure, morphology, segmentation, util
)
from skimage.draw import polygon

import temp 


def F_frangi(img_adapteq):
    """
    Applies Frangi filter to an image after adaptive histogram equalization.
    :param img_adapteq: input image after adaptive histogram equalization
    :return: filtered image

    """
    image_f = 10000*frangi(np.array(img_adapteq), sigmas=range(1, 3, 1), scale_range=None, scale_step=None, alpha=0.25, beta=0.85, gamma=15, black_ridges=True, mode='reflect', cval=0)
    return image_f

def threshold(image):
    """
    Applies thresholding to the input image.

    Args:
        image (numpy.ndarray): The input image.

    Returns:
        numpy.ndarray: The binary mask of the thresholded image.

    """
    # Calculate Otsu threshold
    thresh = threshold_otsu(image)
    
    # Convert image to binary based on threshold
    binary = np.array(image > thresh, dtype=bool)
    
    # Perform morphological opening to remove small objects
    selem = disk(2.7)
    opened_image = morphology.opening(binary, selem)

    # Display the input image, thresholded image, and opened image
    # fig, ax = plt.subplots(ncols=3, figsize=(10, 5))
    # ax[0].imshow(image,cmap='gray')
    # ax[0].set_title('Frangi input image')
    # ax[0].axis('off')

    # ax[1].imshow(binary,cmap='gray')
    # ax[1].set_title('Thresholding')
    # ax[1].axis('off')

    # ax[2].imshow(opened_image,cmap='gray')
    # ax[2].set_title('Morphologic opening')
    # ax[2].axis('off')
    # plt.show()
    
    return opened_image


def load_frame(path):
     # load the frame from path and return the equalize green channel
     img = cv2.imread(path)[:, :, 1]
     img_adapteq = exposure.equalize_adapthist(img, clip_limit=0.03)
     return img_adapteq

def video_stab():

    # Using defaults
    #stabilizer = VidStab()
    #stabilizer.stabilize(input_path='good.mov', output_path='stable_vid_15sec.avi')

    # Using a specific keypoint detector
    #stabilizer = VidStab(kp_method='ORB')
    #stabilizer.stabilize(input_path='good.mov', output_path='stable_vid_15sec1.avi')

    # Using a specific keypoint detector and customizing keypoint parameters
    stabilizer = VidStab(kp_method='FAST', threshold=42, nonmaxSuppression=False)
    stabilizer.stabilize(input_path='13_sec_us.mov', output_path='13_sec_s.avi')

def colon_seg(img):
    """
    This function takes an input image and performs segmentation of the colon using the active contour model. 
    It returns a binary mask with the colon region marked as 1 and the background marked as 0.

    Parameters:
    img (numpy.ndarray): Input image

    Returns:
    numpy.ndarray: Binary mask with the colon region marked as 1 and the background marked as 0

    """

    # Initialize an array of points to form the initial contour
    s = np.linspace(0, 2*np.pi, 400)
    r = 240 + 310*np.sin(s)
    c = 370 + 370*np.cos(s)
    init = np.array([r, c]).T
    
    # Apply Gaussian filter to the input image
    f_img = gaussian(img, 0.003, preserve_range=False)
    
    # Apply active contour model to segment the colon
    snake = active_contour(f_img, init, alpha=0.2, beta=30, gamma=0.01)
    
    # Create a binary mask from the segmented contour
    T = create_mask(img, np.array(snake))

    return T




def find_vessels(image):
    label_img = label(image)
    regions = regionprops(label_img)

    orientations = np.empty(regions.__len__())
    i = 0
    for prop in regions:
        orientations[i] = prop.orientation
        i += 1
    print('...')
    hist, bin_edges = np.histogram(orientations, density=True)
    _ = plt.hist(orientations)  # arguments are passed to np.histogram


def keep_largest_connected_components(mask):
    """
    Removes all connected components in a binary image except the largest one.

    Parameters:
    image (ndarray): A binary input image where 0 represents the background and 1 represents the foreground.

    Returns:
    ndarray: A binary image where all connected components except the largest one have been removed.

    """
    out_img = np.zeros(mask.shape, dtype=np.uint8)

    for struc_id in [1, 2, 3]:

        binary_img = mask == struc_id
        blobs = measure.label(binary_img, connectivity=1)

        props = measure.regionprops(blobs)

        if not props:
            continue

        area = [ele.area for ele in props]
        largest_blob_ind = np.argmax(area)
        largest_blob_label = props[largest_blob_ind].label

        out_img[blobs == largest_blob_label] = struc_id

    return out_img

def remove_background(image):
    """
    Segments the breast tissue in a given grayscale image using Multi-Otsu thresholding and morphological operations.

    Parameters:
    image (ndarray): A grayscale input image.

    Returns:
    ndarray: A binary image where the breast tissue is set to 1 and the remaining tissue is set to 0.

    """
    # Apply Multi-Otsu thresholding to segment the breast tissue
    thresholds = filters.threshold_multiotsu(image, classes=2)
    regions = np.digitize(image, bins=thresholds)

    # Apply morphological operations to remove noise and small holes
    selem = disk(4)
    tmp = morphology.erosion(regions, selem)
    selem = disk(2.5)
    tmp = morphology.closing(tmp, selem)

    # Keep only the largest connected component
    I = keep_largest_connected_components(tmp)

    # Remove small objects and holes
    mask = morphology.remove_small_objects(I, 1000)
    mask = morphology.remove_small_holes(mask, 1200)

    # Apply morphological closing operation to fill remaining holes
    selem = disk(21)
    tmp = morphology.closing(mask, selem)
    I = keep_largest_connected_components(tmp)

    return I






#def try_2(image):   
#    thresholds = filters.threshold_multiotsu(image, classes=2)
#    regions = np.digitize(image, bins=thresholds)
#   # plt.imshow(regions, cmap='gray'),plt.show()
#
#    selem = disk(4)
#    tmp = morphology.erosion(regions,selem)
#    selem = disk(2.5)
#    tmp = morphology.closing(regions,selem)
#    # plt.imshow(tmp, cmap='gray'),plt.show()
#
#    I = keep_largest_connected_components(tmp)
#    # plt.imshow(I, cmap='gray'),plt.show()
#
#    mask = morphology.remove_small_objects(I, 1000)
#    mask = morphology.remove_small_holes(mask, 1200)
#   # plt.imshow(mask, cmap='gray'),plt.show() 
#
#    selem = disk(21)
#    tmp = morphology.closing(mask,selem)
#    I = keep_largest_connected_components(tmp)
#
#    
#    # fig, ax = plt.subplots(ncols=2, figsize=(10, 5))
#    # ax[0].imshow(regions)
#    # ax[0].set_title('Multi-Otsu thresholding')
#    # ax[0].axis('off')
#    # ax[1].imshow(I)
#    # ax[1].set_title('Morphologic closing')
#    # ax[1].axis('off')
#    # plt.show()
#    
#    return I
#

#
def grab_cut():
    img_o = cv2.imread('/Users/ofirbenyosef/hello/frames/MicrosoftTeams-image (10).png')
    img = img_o
    mask = np.zeros(img.shape[:2],np.uint8)
    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)
    rect = (10,10,650,590)
    # (start_x, start_y, width, height).
    cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    img = img*mask2[:,:,np.newaxis]
    plt.imshow(img),plt.colorbar(),plt.show()
    # newmask is the mask image I manually labelled
    newmask = remove_background(img_o)
    plt.imshow(newmask),plt.colorbar(),plt.show()
    # wherever it is marked white (sure foreground), change mask=1
    # wherever it is marked black (sure background), change mask=0
    mask[newmask == 0] = 0
    mask[newmask == 255] = 1
    mask, bgdModel, fgdModel = cv2.grabCut(img,mask,None,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_MASK)
    mask = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    img = img*mask[:,:,np.newaxis]
    plt.imshow(img.astype('uint8')),plt.colorbar(),plt.show()
#
#


def segment_images(path):
    for filename in os.listdir(path):
            frame = load_frame(os.path.join(path, filename))
            binary_frame = threshold(F_frangi(frame))
            #find_vessels(binary_frame)
def find_frame():
    img = cv2.imread('/Users/ofirbenyosef/hello/frames/PHOTO-2022-09-19-19-50-08.jpg')[:,:,1]
    mask = np.zeros(img.shape[:2],np.uint8)
    mask[img < 100] = 1
    mask[img > 100] = 0
    #print(np.shape(mask)[1])
    plt.imshow(mask),plt.colorbar(),plt.show()
    len = range(50 ,np.shape(mask)[0])
    i_start = 0
    for i in len:
        vec = mask[i,:]
        if vec.sum() < 1 :
            i_start = i
            print(i)
            break

    j_start = 0
    len = range(50 ,np.shape(mask)[1])
    for j in len:
        vec = mask[:,j]
        #print(vec)
        #print('\n')
        s = vec.sum()
        if s == 0 :
            j_start = j
            print(j)
            break
  
def create_mask(image, points):
    """
    Creates a binary mask for a region of interest (ROI) specified by a list of points.
    
    Args:
    image: a 2-dimensional numpy array representing the input image.
    points: a list of points that define the ROI.
    
    Returns:
    mask: a binary mask with the same shape as the input image, where the pixels inside the ROI are set to 1, and the
          pixels outside the ROI are set to 0.
    """
    
    # Create an empty mask with the same shape as the input image
    mask = np.zeros_like(image)
    
    # Convert the list of points to integer coordinates
    points = np.array(points, dtype=int)
    
    # Fill the mask with the contour defined by the points using the polygon function from the skimage library
    rr, cc = polygon(points[:, 0], points[:, 1])
    mask[rr, cc] = 1
    
    # Return the binary mask
    return mask


def contrast_enhancement(img):
    """
    Enhances the contrast of the input image by applying Contrast Limited Adaptive Histogram Equalization (CLAHE).
    
    Args:
    img: a 3-dimensional numpy array representing the input image in BGR format.
    
    Returns:
    final_img: a 2-dimensional numpy array representing the enhanced grayscale image.
    """
    
    # Convert the image to RGB color space and take the green channel
    image_bw = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)[:,:,1]
    
    # Create a CLAHE object with a threshold for contrast limiting
    clahe = cv2.createCLAHE(clipLimit = 5)
    
    # Apply CLAHE to the grayscale image
    final_img = clahe.apply(image_bw)
    
    # Return the enhanced grayscale image
    return final_img


def apply_mask(video, mask):
    """
    Applies a binary mask to each frame of an input video, keeping only the pixels within the mask.

    Parameters:
    - video: a 4D numpy array representing the input video, with dimensions (num_frames, height, width, num_channels)
    - mask: a 2D numpy array representing the binary mask to apply to each frame of the input video, with dimensions (height, width)

    Returns:
    - masked_vid: a 4D numpy array representing the masked video, with the same dimensions as the input video
    """
    masked_vid = []
    for i in range(0, video.shape[0]):
        # Apply the mask to each channel of the current frame
        r = mask * video[i, :, :, 0]
        g = mask * video[i, :, :, 1]
        b = mask * video[i, :, :, 2]
        # Stack the masked channels to form the final masked frame
        tmp = np.dstack([r, g, b])
        # Append the masked frame to the output list
        masked_vid.append(tmp)
    # Convert the output list to a numpy array
    masked_vid = np.array(masked_vid)
    return masked_vid


def remove_fat(image):
    """
    Removes the fat tissue from a given BGR image using morphological operations.

    Parameters:
    image (ndarray): A BGR input image.

    Returns:
    ndarray: A binary image where the fat tissue is set to 0 and the remaining tissue is set to 1.

    """

    # Convert image to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Extract hue channel
    hue = hsv_image[:, :, 0]
    
    # Threshold the hue channel to segment the fat tissue
    fat_mask = (hue < 55).astype(np.uint8)
    
    # Remove small connected components
    fat_mask = keep_largest_connected_components(fat_mask)
    
    # Apply opening operation to remove small noise
    selem = disk(21)
    fat_mask = opening(fat_mask, selem)
    
    # Remove small connected components again
    fat_mask = keep_largest_connected_components(fat_mask)
    
    # Apply closing operation to fill small holes
    selem = disk(15)
    fat_mask = closing(fat_mask, selem)
    
    # Invert the mask and return as binary image
    return np.logical_not(fat_mask).astype(np.uint8)





if __name__ == '__main__':
 
    # start = time.time()
    # path = r'frames/MicrosoftTeams-image (10).jpeg'
    # frame = load_frame(path)
    # plt.imshow(cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY),cmap='gray'),plt.show()
    # in_im = contrast_enhancement(cv2.imread(path))
    # 
    # fig = plt.figure(figsize=(10, 8)) 
    # ax = fig.add_subplot(1, 2,1)   
    # ax.imshow(cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY),cmap='gray') 
# 
    # ax.set_title('Orginal')
    # ax = fig.add_subplot(1, 2,2)   
    # ax.imshow(in_im,cmap='gray') 
    # ax.set_title('Contrast enhancement')
    # plt.show()
# 
# 
    # I = try_2(in_im)
    # plt.imshow(in_im*I, cmap="gray")
    # plt.title('Remove backround')
    # plt.show()
    # #binary_frame = threshold(F_frangi(frame*I))
    # binary_frame = threshold(F_frangi(in_im*I))
  # 
    # ##load_video(path)
    # #
    # K = colon_seg(binary_frame)
## 
    # K = np.abs(1-K)
    # selem = disk(5)
## 
    # fig = plt.figure(figsize=(10, 8)) 
    # ax = fig.add_subplot(1, 3 ,1)
    # ax.imshow(frame*K,cmap = 'gray') 
    # ax = fig.add_subplot(1, 3 ,2)
    # K_image = morphology.closing(K, selem)
    # #
    # ax.imshow(frame*K_image*I,cmap = 'gray') 
    # ax = fig.add_subplot(1, 3 ,3)
    # ax.imshow(binary_frame,cmap = 'gray') 
    # plt.show()
# 
# 
    # fig, ax = plt.subplots(ncols=3, figsize=(10, 5))
    # ax[0].imshow(frame,cmap='gray')
    # ax[0].set_title('The first frame of the video')
    # ax[0].axis('off')
# 
    # ax[1].imshow(frame*K,cmap='gray')
    # ax[1].set_title('Remove the conating tissue')
    # ax[1].axis('off')
    # mask = K_image*I
    # ax[2].imshow(mask,cmap='gray')
    # ax[2].set_title('Remove the backround')
    # ax[2].axis('off')
    # plt.show()
    # ------------- #
   # video_stab()
    selem = disk(5)
    video ,video_fc, fs = temp.load_video('good_stable.avi')
    frame = video_fc[120]
    img_wy = remove_fat(frame)
    in_im = contrast_enhancement(frame)
    I = remove_background(in_im)

    binary_frame = threshold(F_frangi(in_im*I))
   # K = colon_seg(binary_frame)
    K = binary_frame
    K = np.abs(1-K)
    K_image = morphology.closing(K, selem)
    mask = K_image*I
    mask_all = np.logical_and(mask,img_wy).astype(int)
    selem = disk(35)
    mask_all = morphology.closing(mask_all, selem)
    mask_all = keep_largest_connected_components(mask_all)
    # ------------- #

    m_vid = apply_mask(video_fc,mask_all)
    temp.showing(m_vid)
   # temp.save_video(m_vid,fs)
 
#






