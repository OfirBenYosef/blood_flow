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



def F_frangi(img_adapteq):
    #cv2.imshow('image', img_adapteq)
    image_f = 10000*frangi(np.array(img_adapteq), sigmas=range(1, 3, 1), scale_range=None, scale_step=None, alpha=0.25, beta=0.85, gamma=15, black_ridges=True, mode='reflect', cval=0)
    #image_m = meijering(np.array(img_adapteq))
    #image_s = sato(np.array(img_adapteq),black_ridges=True, mode='reflect')
    #cv2.imshow('image', image_f)
    #cv2.waitKey(0)
    #plt.imshow(image_f, cmap="gray")
    #plt.show()
    return image_f
def threshold(image):
    thresh = threshold_otsu(image)
    print(thresh)
    binary = np.array(image > thresh, dtype=bool)
    plt.imshow(binary, cmap="gray")
    plt.show()
    selem = disk(2.5)
    opened_image = morphology.opening(binary, selem)
    plt.imshow(opened_image, cmap="gray")
    plt.show()
    return opened_image

def load_frame(path):
     # load the frame from path and return the equalize green channel
     img = cv2.imread(path)[:, :, 1]
     img_adapteq = exposure.equalize_adapthist(img, clip_limit=0.03)
     return img_adapteq

def video_stab():

    # Using defaults
    stabilizer = VidStab()
    stabilizer.stabilize(input_path='data_vid_15sec.mov', output_path='stable_vid_15sec.avi')

    # Using a specific keypoint detector
    stabilizer = VidStab(kp_method='ORB')
    stabilizer.stabilize(input_path='data_vid_15sec.mov', output_path='stable_vid_15sec1.avi')

    # Using a specific keypoint detector and customizing keypoint parameters
    stabilizer = VidStab(kp_method='FAST', threshold=42, nonmaxSuppression=False)
    stabilizer.stabilize(input_path='data_vid_15sec.mov', output_path='stable_vid_15sec2.avi')

def colon_seg(img):
    #img = data.astronaut()
    #image = cv2.imread('/Users/ofirbenyosef/hello/frames/MicrosoftTeams-image (4).png')[1,:,:]
    #img = try_2(image)
    

    s = np.linspace(0, 2*np.pi, 400)
    r = 240 + 310*np.sin(s)
    c = 370 + 370*np.cos(s)
    init = np.array([r, c]).T
    #f_img = roberts(sobel(img))
    #f_img = gaussian(sobel(100*img), 0.03, preserve_range=False)
    f_img = gaussian(img, 0.003, preserve_range=False)
    print('s',time.time())
    snake = active_contour(f_img, init, alpha=0.2, beta=30, gamma=0.01)
    print('e',time.time())
    #fig, ax = plt.subplots(figsize=(7, 7))
    #ax.imshow(img, cmap=plt.cm.gray)
    #ax.plot(init[:, 1], init[:, 0], '--r', lw=3)
    #ax.plot(snake[:, 1], snake[:, 0], '-b', lw=3)
    #ax.set_xticks([]), ax.set_yticks([])
    #ax.axis([0, img.shape[1], img.shape[0], 0])
    #plt.show()
    T = create_mask(img, np.array(snake))
    print('p',time.time())

    return T

def load_video(path):
    # load the video from path
    cap = cv2.VideoCapture('/Users/ofirbenyosef/hello/cut2.mov')
    while (cap.isOpened()):
        ret, frame = cap.read()
        #b_frame = np.uint8(1000*threshold(F_frangi(exposure.equalize_adapthist(frame[:, :, 1], clip_limit=0.03))))
        b_frame = try_2(exposure.equalize_adapthist(frame, clip_limit=0.03))
        cv2.imshow('frame', b_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()

def color_hist():
    file0 = 'frames/MicrosoftTeams-image (2).png'
    img_1 = cv2.imread(file0)
    color = ('b','g','r')
    plt.figure()
    for i,col in enumerate(color):
        histr_1 = cv2.calcHist([img_1],[i],None,[256],[0,256])
        plt.plot(histr_1,color = col)
        plt.xlim([0,256])
    plt.show()
    file1 = 'frames/MicrosoftTeams-image (10).png'
    img_2 = cv2.imread(file1)
    color = ('b','g','r')
    plt.figure()
    for i,col in enumerate(color):
        histr_2 = cv2.calcHist([img_2],[i],None,[256],[0,256])
        plt.plot(histr_2,color = col)
        plt.xlim([0,256])
    plt.show()
    for i,col in enumerate(color):
        histr_1 = cv2.calcHist([img_1],[i],None,[256],[0,256])
        histr_2 = cv2.calcHist([img_2],[i],None,[256],[0,256])
        plt.plot(np.divide(histr_2,histr_1),color = col)
        plt.xlim([0,256])
    plt.show()
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
    #plt.title("Histogram of orientation")
    #plt.show()

def r_prop(image_b,image):
    img = image
# Binary image, post-process the binary mask and compute labels

    labels = measure.label(img)

    fig = px.imshow(image_b)
    fig.update_traces(hoverinfo='skip') # hover is only for label info

    props = measure.regionprops(labels, img)
    properties = ['area', 'eccentricity', 'perimeter', 'intensity_mean']

    # For each label, add a filled scatter trace for its contour,
    # and display the properties of the label in the hover of this trace.
    for index in range(1, labels.max()):
        label_i = props[index].label
        contour = measure.find_contours(labels == label_i, 0.5)[0]
        y, x = contour.T
        hoverinfo = ''
        for prop_name in properties:
            hoverinfo += f'<b>{prop_name}: {getattr(props[index], prop_name):.2f}</b><br>'
        fig.add_trace(go.Scatter(
            x=x, y=y, name=label_i,
            mode='lines', fill='toself', showlegend=False,
            hovertemplate=hoverinfo, hoveron='points+fills'))

    plotly.io.show(fig)
def keep_largest_connected_components(mask):
    '''
    Keeps only the largest connected components of each label for a segmentation mask.
    '''

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
def try_2(image):   
   #image = human_mitosis()
   #image = cv2.imread('/Users/ofirbenyosef/hello/frames/MicrosoftTeams-image (8).png')
   #img = rgb2gray(image)
   #img = rgb2gray(img_vid)
   #fig, ax = plt.subplots()
   #ax.imshow(img, cmap='gray')
   #ax.set_title('Microscopy image of human cells stained for nuclear DNA')
   #plt.show()
   #fig, ax = plt.subplots(figsize=(5, 5))
   #qcs = ax.contour(img, origin='image')
   #ax.set_title('Contour plot of the same raw image')
   #plt.show()
   #print(qcs.levels)

    thresholds = filters.threshold_multiotsu(image, classes=2)
    regions = np.digitize(image, bins=thresholds)
    plt.imshow(regions, cmap='gray'),plt.show()

    selem = disk(1.5)
    tmp = morphology.closing(regions,selem)
    I = keep_largest_connected_components(tmp)
    
    mask = morphology.remove_small_objects(I, 50)
    mask = morphology.remove_small_holes(mask, 500)

    selem = disk(15)
    tmp = morphology.closing(mask,selem)
    I = keep_largest_connected_components(tmp)

    
    fig, ax = plt.subplots(ncols=2, figsize=(10, 5))
    ax[0].imshow(regions)
    ax[0].set_title('Original')
    ax[0].axis('off')
    ax[1].imshow(I)
    ax[1].set_title('Multi-Otsu thresholding')
    ax[1].axis('off')
    plt.show()
    
    return I
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
    newmask = try_2(img_o)
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
def ploting():
    path = r'/Users/ofirbenyosef/Desktop/OneDrive - Technion/מסמכים/מסמכים/סמסטר 8/פרוייקט ב/temp/frame00038.png'
    # Using cv2.imread() method
    img = cv2.imread(path)
    # Displaying the image
    cv2.imshow('image', img)
    cv2.waitKey(0)

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
    # Create an empty mask
    mask = np.zeros_like(image)
    # Convert the points to integer coordinates
    points = np.array(points, dtype=int)
    # Fill the mask with the contour
    rr, cc = polygon(points[:, 0], points[:, 1])
    mask[rr, cc] = 1
    #plt.imshow(mask),plt.show()
    return mask

def contrast_enhancement(img):
    # take the green channel
    image_bw = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)[:,:,1]
    # The declaration of CLAHE
    # clipLimit -> Threshold for contrast limiting
    clahe = cv2.createCLAHE(clipLimit = 5)
    final_img = clahe.apply(image_bw) + 30
    return final_img



if __name__ == '__main__':
    start = time.time()
    path = r'frames/MicrosoftTeams-image (6).png'
    frame = load_frame(path)
    #plt.imshow(20*np.log10(abs(np.fft.fft2(frame)))),plt.colorbar(),plt.show()
    in_im = contrast_enhancement(cv2.imread(path))
    plt.imshow(in_im,cmap='gray'),plt.show()
    #frame_s = frame + binary_frame
    #plt.imshow(frame_s,cmap = 'gray'),plt.show()
    # find_vessels(binary_frame)
    #r_prop(frame,binary_frame)
    #video_stab()
    #segment_images(path)
    I = try_2(in_im)
    binary_frame = threshold(F_frangi(frame*I))
    
    ##load_video(path)
    #
    K = colon_seg(binary_frame)
#
    K = np.abs(1-K)
    selem = disk(5)
#
    fig = plt.figure(figsize=(10, 8)) 
    ax = fig.add_subplot(1, 3 ,1)
    ax.imshow(frame*K,cmap = 'gray') 
    ax = fig.add_subplot(1, 3 ,2)
    K_image = morphology.closing(K, selem)
    #
    ax.imshow(frame*K_image*I,cmap = 'gray') 
    ax = fig.add_subplot(1, 3 ,3)
    ax.imshow(binary_frame,cmap = 'gray') 
    plt.show()
    #end = time.time()
    #print(str(end-start))
    ##grab_cut()
    ##find_frame()
    ##color_hist()
#