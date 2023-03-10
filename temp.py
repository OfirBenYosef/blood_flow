
import os
from skimage.filters import gaussian, laplace, sobel, roberts,gabor,gabor_kernel
from skimage import feature
from skimage.restoration import denoise_tv_chambolle
import matplotlib.pyplot as plt
import numpy as np
import cv2
from skimage import exposure
from scipy import signal
from scipy.fft import fftshift
from numpy.fft import fft ,ifft
import time
import classify

def load_frame(path):
    """
    Loads a single frame from the specified file path and returns the green channel
    of the frame after histogram equalization.
    """
    frame = cv2.imread(path)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = exposure.equalize_adapthist(frame, clip_limit=0.03)
    return frame

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from skimage import exposure

def load_frame(path):
    """
    Loads a single frame from the specified file path and returns the green channel
    of the frame after histogram equalization.
    """
    frame = cv2.imread(path)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = exposure.equalize_adapthist(frame, clip_limit=0.03)
    return frame


def load_video(path='stable_video_2.avi'): 
    """
    Load video from a file and return the processed video frames, full color frames, and number of frames

    Args:
    path (str): path to the video file. Default is 'stable_video_2.avi'

    Returns:
    tuple: processed video frames, full color frames, number of frames

    """
    cap = cv2.VideoCapture(path)
    #cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
    num_frames = int(cap.get(5))
    width = int(cap.get(3))
    height = int(cap.get(4))
    print(num_frames, width, height)
    
    video = []
    video_full_color = []
    while cap.isOpened():
        print('hii')
        ret, frame = cap.read()
        if ret:
            video.append(frame[:, :, 1])
            video_full_color.append(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
            # key = cv2.waitKey(10)
            # if key == ord('q'):
            #     break
        else:
            break
    # cv2.destroyAllWindows()
    cap.release()
    video = np.array(video)
    video_full_color = np.array(video_full_color)
    return video, video_full_color, num_frames


def stft_2d(video):
    """
    Calculates the 2D STFT of a given video.
    
    Parameters:
    video (np.ndarray): the video to be transformed
    
    Returns:
    None
    
    """
    fs = 24 # sample rate in Hz
    
    for i in range(video.shape[1]): # loop over each pixel in the width of the video
        for j in range(video.shape[2]): # loop over each pixel in the height of the video
            x = video[:,i,j] # get the 1D time series of a single pixel
            f, t, Zxx = signal.stft(x, fs, nperseg=128, window='boxcar')
            # plot the STFT magnitude of the time series using a color mesh plot
            plt.pcolormesh(t, f, np.abs(Zxx), vmin=0, vmax=5, shading='gouraud')
            plt.title('STFT Magnitude')
            plt.ylabel('Frequency [Hz]')
            plt.xlabel('Time [sec]')
            plt.colorbar()
            plt.show()

def crop_good(vid):
    """
    Crop the video to only contain the area with good blood flow and play the cropped video.
    
    Parameters:
    vid (np.ndarray): the video to be cropped
    
    Returns:
    None
    
    """
    left = 40
    top = 110
    right = 110
    bottom = 390
    good = vid[:, top:bottom, left:right]
    play_vid(good)
    # stft_2d(good)
    print('good')
    shell(good, 'good')

def crop_bad(vid):
    """
    Crop the video to only contain the area with bad blood flow and play the cropped video.
    
    Parameters:
    vid (np.ndarray): the video to be cropped
    
    Returns:
    None
    
    """
    left = 270
    top = 150
    right = 520
    bottom = 200
    bad = vid[:, top:bottom, left:right]
    play_vid(bad)
    # stft_2d(bad)
    print('bad')
    shell(bad, 'bad')

def play_vid(vid):
    """
    Play the given video.
    
    Parameters:
    vid (np.ndarray): the video to be played
    
    Returns:
    None
    
    """
    try:
        dim = vid.shape[3]
        for fr in range(vid.shape[0]):
            cv2.imshow('video', cv2.cvtColor(vid[fr,:,:,:], cv2.COLOR_BGR2RGB))
            cv2.waitKey(1)
    except:
        for fr in range(vid.shape[0]):
            cv2.imshow('video', vid[fr,:,:])
            cv2.waitKey(1)
    cv2.destroyAllWindows()

def crop_box(x, y, video, len=20):
    """
    Crop a box of size `len x len` around the given pixel (x,y) in the video.
    
    Parameters:
    x (int): x-coordinate of the pixel
    y (int): y-coordinate of the pixel
    video (np.ndarray): the video to be cropped
    len (int, optional): size of the box to be cropped. Default is 20.
    
    Returns:
    np.ndarray: the cropped box of shape `[t, len, len]`
    
    """
    left = int(y - np.ceil(len/2))
    top = int(x - np.ceil(len/2))
    right = int(y + np.ceil(len/2))
    bottom = int(x + np.ceil(len/2))
    try:
        dim = vid.shape[3]
        box = video[:, top:bottom, left:right, :]
    except:
        box = video[:, top:bottom, left:right]
    return box


def mean_box(video):
    """
    Calculates the mean pixel intensity for each frame in a video and returns an array of size (t, 1) where t is the number of frames.

    Parameters:
    video (np.ndarray): A numpy array of shape (t, w, h) where t is the number of frames, w is the width of each frame, and h is the height of each frame.

    Returns:
    np.ndarray: A numpy array of shape (t, 1) where each element is the mean pixel intensity for each frame.
    """
    mean_box = np.mean(video, axis=(1, 2))
    return mean_box

def shell(video, label, idx=0):
    """
    Crops the video into boxes, computes the mean pixel intensity of each box, and applies a Short-Time Fourier Transform (STFT) on the resulting signal. The STFT magnitudes are plotted and saved as PNG images.

    Parameters:
    video (np.ndarray): A numpy array of shape (t, w, h) where t is the number of frames, w is the width of each frame, and h is the height of each frame.
    label (str): A label to use as a prefix for the saved PNG images.
    idx (int, optional): The starting index to use for the saved PNG images. Defaults to 0.

    Returns:
    int: The ending index used for the saved PNG images.
    """
    fs = 24
    Stride = 10
    num_x = int(np.floor(video.shape[1] / Stride)) - 1
    num_y = int(np.floor(video.shape[2] / Stride)) - 1

    for i in range(num_x):
        for j in range(num_y):
            X = Stride + Stride * i
            Y = Stride + Stride * j
            box = crop_box(X, Y, video)
            x = mean_box(box)

            X = signal.stft(x, fs, nperseg=128)
            X = np.fft.fftshift(np.fft.fft(x))
            sos = signal.butter(6, 0.4, 'low', fs=24, output='sos')
            filtered = signal.sosfilt(sos, X)
            iX = np.fft.ifft(filtered)
            f, t, Zxx = signal.stft(iX, fs=24, nperseg=72, noverlap=12, padded=False, window='hamming')

            plt.pcolormesh(t, f, np.abs(Zxx), vmin=0, vmax=5, shading='gouraud')
            plt.title('STFT Magnitude')
            plt.ylabel('Frequency [Hz]')
            plt.xlabel('Time [sec]')
            plt.colorbar()

            name = f'stft_out/{label}_{idx}.png'
            print(name)
            plt.savefig(name)
            idx += 1
            plt.close()

    return idx


def color_hist_1():
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

def color_hist(frame):

    """
    get np array [w,h,3]
    calculate the color (RGB) histogram
    return np array [256,3]
    
    """

    color_1 = ('h','s','v')
    color = ('r','g','b')
    histr = []
    for i,col in enumerate(color):
        histr.append(cv2.calcHist([frame],[i],None,[256],[0,256]))

    return histr

def hist_shall(video):
    
    Stride = 10
    num_x = int(np.floor(video.shape[1]/Stride) )- 1
    num_y = int(np.floor(video.shape[2]/Stride)) - 1
    idx = 0

    for i in range(0,num_x): 
        for j in range(0,num_y): 
                X = Stride + Stride*i
                Y = Stride +Stride *j
            
                hists = []
                chis = []
                
                box = crop_box(X,Y,video)
                for ii in range(0,video.shape[0]):
                    frame = box[ii,:,:,:]
                    hists.append(color_hist(frame))
                hists = np.array(hists)
                print('...')
                
                for ii in range(1,len(hists)):
                    chis.append(chi_square_hist(hists[0,:,:,0],hists[ii,:,:,0]))
                print('i = ' + str(i) + ' j = ' + str(j) )
                chis = np.array(chis).transpose()
                plot_chi(chis)
                
                #time.sleep(1)
            

# --------------------------------------------------#

def chi_square_hist(hist1,hist2):
    chi = []
    for channel in range(0,hist1.shape[0]):
        chi.insert(channel,0)
        temp = int(0)
        for i in range(0,hist1.shape[1]):
            if hist1[channel,i] == 0.0 and hist2[channel,i] == 0.0:
                temp += 0 
            else:
                temp += (np.square(hist1[channel,i] - hist2[channel,i])/(hist1[channel,i] + hist2[channel,i]))
        chi[channel] = temp 
    #print(chi)
    return chi


def plot_chi(chi):
    chi = np.array(chi)
    plt.figure()
    color = ('r','g','b')
    color_1 = ('h','s','v')
    for i,col in enumerate(color):
        plt.plot(chi[0,i,:],color = col)
        plt.legend(color_1)
        #plt.xlim([0,256])
    plt.pause(1)
    plt.close()
    
def showing(imgs):
    plt.figure()
    for i in imgs:
        plt.imshow(i,cmap=plt.cm.gray)
        plt.pause(0.05)
        plt.close()
    
    
def crop_vid(vid,left,top ,right,bottom ):
    """
    Crops a given video based on the specified coordinates.
    
    Parameters:
    vid (ndarray): The input video to be cropped.
    left (int): The left boundary of the crop.
    top (int): The top boundary of the crop.
    right (int): The right boundary of the crop.
    bottom (int): The bottom boundary of the crop.
    
    Returns:
    ndarray: The cropped video.
    """
    try:
        dim = vid.shape[3]
        crop_vid = vid[:,top:bottom,left:right,:]
    except:
        crop_vid = vid[:,top:bottom,left:right]
    #play_vid(crop_vid)
    return crop_vid

def time_crop(video,dur=5,start=0,fs=24):
    """
    Crops a given video based on a specified duration and start time.
    
    Parameters:
    video (ndarray): The input video to be cropped.
    dur (int, optional): The duration of the cropped video in seconds. Defaults to 5.
    start (int, optional): The start time of the cropped video in seconds. Defaults to 0.
    
    Returns:
    ndarray: The cropped video.
    int: The end time of the cropped video in seconds.
    """

    num_of_frames = fs*dur
    end = start+num_of_frames
    try:
        dim = video.shape[3]
        crop_vid = video[start:end,:,:,:]
    except:
         crop_vid = video[start:end,:,:]
    #play_vid(crop_vid)
    return crop_vid , end

def texture_ana(vid, label):
    '''
    This function performs texture analysis on a video and returns the filtered images obtained by applying a gabor filter with different frequency and orientation values.
    
    Parameters:
        vid (ndarray): The input video.
        label (int): A label to identify the video.
    
    Returns:
        tiles (list): A list of filtered images.
    '''
    tiles = []
    Stride = 20
    num_x = int(np.floor(vid.shape[1]/Stride) )- 1
    num_y = int(np.floor(vid.shape[2]/Stride)) - 1
    pi = np.pi

    for i in range(0,num_x): 
        for j in range(0,num_y):
            X = Stride + Stride*i
            Y = Stride +Stride *j
            img = crop_box(X,Y,vid,len=2*Stride)
            for t in range(img.shape[0]):
                for freq in (0.05, 0.15, 0.25, 0.3, 0.45):
                    for theta_ in (0, pi/4, pi/2, 3*pi/4, pi, 5*pi/4, 6*pi/4, 7*pi/4):
                        temp = np.resize(cv2.cvtColor(img[t], cv2.COLOR_RGB2GRAY), (128, 128))
                        filt_real, filt_imag = gabor(temp, frequency=freq, theta=theta_)
                        tiles.append(filt_real)
    return tiles


def show_filters(filters):
    # plot
    fig = plt.figure(figsize=(10, 8)) 
    i =1
    for filter in filters:
        ax = fig.add_subplot(8, 5 ,i) 
        ax.imshow(np.array(filter))
        i +=1
        if i > 40:
            break

def clc_gabor(frame):
    f = np.array(frame[0])
    pi = np.pi
    gabor_l = []
    for freq in (0.05,0.15,0.25,0.3,0.45):
        for theta_ in (0,pi/4,pi/2,3*pi/4,pi,5*pi/4,6*pi/4,7*pi/4):
            print(theta_)
            temp = np.resize(cv2.cvtColor(f,cv2.COLOR_RGB2GRAY),(256,256))
            filt_real, filt_imag = gabor(temp, frequency=freq,theta = theta_) 
            gabor_l.append(filt_real)
    return gabor_l

    
def good_hist_prep(video):
    Stride = 10
    num_x = int(np.floor(video.shape[1]/Stride) )- 1
    num_y = int(np.floor(video.shape[2]/Stride)) - 1

    idx = 0
    f_hists = []
    for ii in range(0,video.shape[0]):
        hists = []
        for i in range(0,num_y): 
            for j in range(0,num_x): 
                    X = Stride + Stride*i
                    Y = Stride +Stride *j
                    box = crop_box(X,Y,video)
                    hists.append(color_hist(box[ii]))
        hists = np.array(hists)
        f_hists.append(np.average(hists,axis=0).astype(np.int32))
    return np.array(f_hists)
 
def time_corp_shell(video,stride,dur,left,top ,right ,bottom ,label = 'bad'):
     Start = 0 
     fs =24
     end = video.shape[0]
     rep = int(np.floor(end/(dur*fs)))
     idx = 0
     for i in range(rep):
        croped,Start = time_crop(video,dur=5,start=Start)
        Start = Start - stride
        idx = shell(crop_vid(croped,left,top,right,bottom),label,idx)
        print(idx)


def hist_shall2(video,good_hist):
    
    Stride = 10
    num_x = int(np.floor(video.shape[1]/Stride) )- 1
    num_y = int(np.floor(video.shape[2]/Stride)) - 1
    idx = 0

    for i in range(0,num_x): 
        for j in range(0,num_y): 
                X = Stride + Stride*i
                Y = Stride +Stride *j
            
                hists = []
                chis = []
                
                box = crop_box(X,Y,video)
                for ii in range(0,video.shape[0]):
                    frame = box[ii,:,:,:]
                    hists.append(color_hist(frame))
                hists = np.array(hists)
                print('...')
                
                for ii in range(0,len(hists)):
                    chis.append(chi_square_hist(good_hist[ii],hists[ii,:,:,0]))
                print('i = ' + str(i) + ' j = ' + str(j) )
                tv_denoised = denoise_tv_chambolle(np.array(chis), weight=70)
                chis = np.array(chis).transpose()
                tv_denoised = np.array(tv_denoised).transpose()

                plot_chi(tv_denoised)

def save2csv(histograms):
    f_out = open('good_hists.csv',"w")
    num = 0
    c = 0
    for hist in histograms :
        #f_out.write(str(num)+',')
        num += 1
        c = 0
        for color in hist :
            f_out.write(str(num)+','+str(c)+',')
            c += 1
            for i in range(len(color)-1):
                #print(str(color[i]) + ',')
                f_out.write(str(color[i][0]) +',')
            f_out.write(str(color[i+1][0])+'\n')
    f_out.close()
# ---------------------------------#
def vid2gray(video):
    """
    Convert color video (t, X, Y, 3) to grayscale.
    
    Parameters
    ----------
    video: numpy.ndarray
        A 4-dimensional numpy array representing the video with shape (t, X, Y, 3).

    Returns
    -------
    numpy.ndarray
        A 3-dimensional numpy array representing the grayscale video with shape (t, X, Y).
    """
    gray_video = []
    for i in range(video.shape[0]):
        gray_video.append(cv2.cvtColor(video[i, :, :, :], cv2.COLOR_RGB2GRAY))
    gray_video = np.array(gray_video)
    return gray_video

def FFT_module(vid,fs):
    """
    Calculates the Fast Fourier Transform (FFT) of the mean intensity of the green channel of a video.

    less
    Copy code
    Parameters:
    vid (ndarray): Video frames in the shape (num_frames, height, width, channels)
    fs (int): Sampling rate of the video in Hz.

    Returns:
    ndarray: The FFT result in the shape (num_frames, height, width)
    """
    # Convert the video frame to grayscale
    gray_frame = mean_box(vid[:,:,:,1])
    
    # Perform FFT on the grayscale frame
    fft_result = fftshift(fft(gray_frame))
    
    # Prepare the frequency axis
    n = np.arange(len(fft_result)) - len(fft_result) / 2
    time = len(fft_result) / fs
    frequencies = n / time
    
    # Get the absolute values of the FFT result
    magnitude = np.abs(fft_result)
    
    # Set the central value to 0
    magnitude[int(len(gray_frame) / 2)] = 0
    
    # Filter out low magnitude values
    filtered_magnitude = magnitude
    filtered_magnitude[magnitude < magnitude.max() / 4] = 0
    
    return filtered_magnitude


def color_module(vid):
    """
    A function to extract color features from a video.
    Parameters:
    vid (ndarray): The video data.

    Returns:
    ndarray: A 2D array of color features, one row per video frame.
    """
    color_features = []
    for frame_index in range(vid.shape[0]):
        frame = vid[frame_index,:,:,:]
        color_histogram = color_hist(frame)
        flat_color_histogram = np.reshape(color_histogram,-1) 
        color_features.append(flat_color_histogram)
    color_features = np.array(color_features)
    return color_features

def texture_module(vid):
    """
    A function to extract texture features from a video.

    Parameters:
    vid (ndarray): The video data.
    
    Returns:
    ndarray: A 2D array of texture features, one row per video frame.
    """
    tiles = []
    pi = np.pi
    freqs = (0.05, 0.15, 0.25, 0.3)
    ang = (0, np.pi/4, np.pi/2, np.pi, 7*np.pi/4)
    idxs = choose_random_frames(vid.shape[0] - 1, 5)
    
    # loop over the selected frames
    for t in idxs:
        for freq in freqs:
            for theta in ang:
                # Convert the current frame to grayscale
                temp = cv2.cvtColor(vid[t], cv2.COLOR_RGB2GRAY)
                
                # Apply the gabor filter to extract texture features
                filt_real, filt_imag = gabor(temp, frequency=freq, theta=theta, sigma_x=3, sigma_y=3)
                
                # Flatten the texture features
                texture_features = np.reshape(filt_real, -1)
                
                # Append to the list of tiles
                tiles.append(texture_features)
                
    # Convert the list of tiles to a numpy array
    tiles = np.array(tiles)
    return tiles



def choose_random_frames(vid_len=120, num_of_frames=10):
    """
    This function returns an array of `num_of_frames` random integers between 1 and `vid_len`.
    The random integers represent the frame numbers.
    
    Parameters:
        vid_len (int): The length of the video in frames.
        num_of_frames (int): The number of random frame numbers to return.
    
    Returns:
        numpy.ndarray: An array of `num_of_frames` random integers between 1 and `vid_len`.
    """
    return np.random.randint(1, vid_len + 1, num_of_frames)


def analyze_shell(video, fs=24):
    """
    Analyzes the video by cropping it into multiple patches, computing the FFT, color, and texture features of each patch,
    and returning the results.
    
    Parameters:
        - video: numpy.ndarray, the input video
        - fs: int, the frame rate of the video (default: 24)
        
    Returns:
        - numpy.ndarray, an array of FFT, color, and texture features
    """
    stride = 10
    num_x = int(np.floor(video.shape[1] / stride)) - 1
    num_y = int(np.floor(video.shape[2] / stride)) - 1
    fft = []
    color = []
    texture = []

    for i in range(num_x):
        for j in range(num_y):
            x = stride + stride * i
            y = stride + stride * j
            P = crop_box(x, y, video)
            t = time.time()
            fft.append(FFT_module(P, fs))
            color.append(color_module(P))
            texture.append(texture_module(P))
            print('time = ', str(time.time() - t))

    return np.array([fft, color, texture])


def analyze_shell2(video,label, fs=24):
    """
    Analyzes the video by cropping it into multiple patches, computing the FFT, color, and texture features of each patch,
    and returning the results.
    
    Parameters:
        - video: numpy.ndarray, the input video
        - fs: int, the frame rate of the video (default: 24)
        
    Returns:
        - numpy.ndarray, an array of FFT, color, and texture features
    """
    stride = 20
    num_x = int(np.floor(video.shape[1] / stride))
    num_y = int(np.floor(video.shape[2] / stride))
    #fft = []
    #color = []
    #texture = []
    #
    classifier = classify.Classify('SVM_model.sav','TrainedNN.pth','TrainedGaborModel2.pth')

      
    img = video[0]
    plt.figure()
    for i in range(num_x):
        for j in range(num_y):

            x = int(stride/2 + stride * i)
            y = int(stride/2 + stride * j)
            P = crop_box(x, y, video)

            ## new
            if P[0].mean().mean() < 50:
                continue

            t = time.time()
            fft = FFT_module(P, fs)
            color = color_module(P)
            #texture = texture_module(P)
            print('time = ', str(time.time() - t))
            # put the feature vectors into the classifier and get estimation for each
            output = classifier([fft,color[-1],P[-1]])
            # classifier consider all features
            #label = int(np.logical_or(output[2],output[1]))
            label = int(output[0])
            #label = output.mean().round()
            # color the corspondent patch according to the label
            img = color_square_shell(img,x=x,y=y,step_size=20,label=label)
            pass
            #show the mask
            plt.imshow(img)
            #plt.pause(0.05)
            if x>150:
                pass
            if i == num_x-1 and j == num_y -1:
                plt.imshow(img)
                plt.show()
                break
            # plt.close()


    plt.imshow(img)
    plt.show()
    print("/?")
   # return np.array([fft, color, texture])

def time_corp_analyze_shell(video, stride, dur, left, top, right, bottom, label='bad', fs=24):
    """
    Analyzes the video by cropping it into multiple segments and running `analyze_shell` on each segment.
    
    Parameters:
        - video: numpy.ndarray, the input video
        - stride: int, the stride used to separate the segments
        - dur: int, the duration of each segment
        - left, top, right, bottom: int, the coordinates used to crop the video
        - label: str, a label for the video (default: 'bad')
        - fs: int, the frame rate of the video (default: 24)
        
    Returns:
        - list, a list of results of `analyze_shell`
    """
    start = 0
    end = video.shape[0]
    rep = int(np.floor(end / (dur * fs)))
    result = []
    for i in range(rep):
        croped, start = time_crop(video, dur=dur, start=start, fs=fs)
        start = start - stride
        result.append(analyze_shell2(crop_vid(croped, left, top, right, bottom), fs=fs,label=label))
    
    return np.array(result[0])



def draw_grid(img, step_size=20, color=(255, 0, 0), thickness=1):
    """
    Draws a grid on the image `img`.
    
    Parameters:
        - img: numpy.ndarray, the input image
        - step_size: int, the size of one step (default: 20)
        - color: tuple (B, G, R), the color of the grid lines (default: (255, 0, 0))
        - thickness: int, the thickness of the grid lines (default: 1)
        
    Returns:
        - numpy.ndarray, the output image with the grid drawn
    """
    x, y = img.shape[1], img.shape[0]
    # Draw horizontal lines
    for i in range(0, y, step_size):
        cv2.line(img, (0, i), (x, i), color, thickness)
    # Draw vertical lines
    for i in range(0, x, step_size):
        cv2.line(img, (i, 0), (i, y), color, thickness)
    return img

def color_grid_square(img, i, j, color, step_size=20,alpha =0.4):
    """
    Adds a colored square to the image `img` at grid position (i, j).
    
    Parameters:
        - img: numpy.ndarray, the input image
        - i: int, the row position of the square
        - j: int, the column position of the square
        - color: tuple (B, G, R), the color of the square
        - step_size: int, the size of one step (default: 20)
        - alpha: float, the opacity of the square (default: 0.4)
        
    Returns:
        - numpy.ndarray, the output image with the colored square added
    """
    # Create a copy of the input image
    overlay = img.copy()
    i = int(i)
    j = int(j)
    # Draw a filled rectangle on the overlay
    cv2.rectangle(overlay,
                  (j * step_size, i * step_size),  # top-left corner
                  ((j + 1) * step_size, (i + 1) * step_size),  # bottom-right corner
                  color,
                  -1)  # negative thickness means filled rectangle
    # Combine the overlay and the original image using a weighted sum
    image_new = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
    return image_new


def color_square_shell(img,x=0,y=0,step_size=20,label=1):
    """
    This function colors a square in an image based on the specified coordinates and label.
    The square is placed on a grid with square cells of a given step size.

    Parameters:
    - img (np.ndarray): the image to be modified
    - x (int): x-coordinate of the center of the square
    - y (int): y-coordinate of the center of the square
    - step_size (int): size of the grid cells
    - label (int): either 1 or 0, indicating whether to color the square green (good) or red (bad)

    Returns:
    - np.ndarray: the modified image with the colored square
    """
    i = np.floor(x/step_size)
    j = np.floor(y/step_size)
    # good is green
    good = (0,150,0)
    # bad is red
    bad = (150,52,219)
    alpha = 0.4
    color = good if label==0 else bad
    return color_grid_square(img, i, j, color, step_size=step_size,alpha=alpha)


if __name__ == '__main__':
    path = r'/Users/ofirbenyosef/hello/frames/MicrosoftTeams-image (9).png'
    save_path = '/home/stavb/BFE_final/np_files'
    #frame = load_frame(path)
    path = r'C:\Users\Administrator\PycharmProjects\blood_flow\human.avi'
    vid, full_color, fs = load_video(path)
    fs = 25
    # samples = np.array([],dtype=object)
    # vid, full_color, fs = load_video('sample.vid.avi')
    # paths = ['/home/stavb/BFE_final/good','/home/stavb/BFE_final/bad']
    # save_path = '/home/stavb/BFE_final/np_files'
    # for path in paths:
    #     for file in os.listdir(path):
    #         file_path = os.path.join(path,file)
    #         label_string = path.split('/')[-1]
    #         if label_string == 'good':
    #             label = 0
    #         else:
    #             label = 1
    #         vid, full_color, fs = load_video(file_path)
    #         features = time_corp_analyze_shell(full_color, stride=0, dur=4, left=0, top=0, right=720, bottom=576,
    #                                fs=fs,label = label )
    #
    #
    #         name = file.split("_")[0]+'.npy'
    #         np.save(os.path.join(save_path, name), features)
    # name = '.npy'
    # print('horrs')

    prefix = 'new_data'
    #classify.make_data_func2(save_path, save_path, prefix)
    #classify.train_classifiers(save_path,save_path,prefix = prefix,num_of_tries=1,show = True)

    #stft_2d(vid)
    #crop_good(vid)
    #crop_bad(vid)
    left = 270
    top = 150
    right = 520
    bottom = 200
    #hist_shall(crop_vid(full_color))
    #good_t = texture_ana(crop_vid(full_color[60:62,:,:,:],left = 40,top = 110,right = 110,bottom = 390),'good')
    #bad_t = texture_ana(crop_vid(full_color[60:62,:,:,:],left = 270,top = 150 ,right = 520,bottom = 200),'bad')
    #showing(bad_t)
    #np.save('good_t.npy',good_t)
    #np.save('bad.npy',bad_t)
    #show_filters(good_t)
    #show_filters(bad_t)

    #good_hist = good_hist_prep(crop_vid(full_color,left,top ,right,bottom))
    #hist_shall2(crop_vid(full_color,left = 40,top = 110,right = 110,bottom = 390),good_hist)
    #save2csv(good_hist)
    #showing(good_t)
    #time_corp_shell(vid,stride=0,dur=5,left=270,top=150,right=520,bottom=200,label = 'bad')
    #time_corp_shell(vid,stride=0,dur=5,left = 40,top = 110,right = 110,bottom = 390,label = 'good')
    #Bad = time_corp_analyze_shell(full_color,stride=0,dur=4,left=270,top=150,right=520,bottom=200,label = 'bad',fs = fs)
    #Good = time_corp_analyze_shell(full_color,stride=0,dur=4,left = 40,top = 110,right = 110,bottom = 390,label = 'good',fs = fs)

    #all = time_corp_analyze_shell(full_color,stride=0,dur=4,left = 0,top = 0,right = 720,bottom = 576,label = 'good',fs = fs)
    vid_shape = full_color.shape[1:]
    all2 = time_corp_analyze_shell(full_color, stride=0, dur=4, left=0, top=0, right=vid_shape[1], bottom=vid_shape[0], label='good',
                                  fs=fs)

    print('HHH')
    #img = cv2.cvtColor(cv2.imread("frames/MicrosoftTeams-image (9).png"),cv2.COLOR_BGR2RGB)
    img1 = draw_grid(img)
    img2 = color_grid_square(img1, 3, 26, (0,150,0),alpha =0.4)
    img2 = color_grid_square(img2, 7, 26, (0,150,0),alpha =0.4)
    img2 = color_grid_square(img2, 18, 26,  (0,150,0),alpha =0.4)
    plt.imshow(img2),plt.show()

    #good_t = np.load('good_t.npy')
    #bad_t  = np.load('bad.npy')
    #showing(good_t)
    #show_filters(clc_gabor(crop_vid(full_color[60:61,:,:,:],left = 270,top = 150 ,right = 520,bottom = 200)))
    #show_filters(clc_gabor(crop_vid(full_color[60:61,:,:,:],left = 40,top = 110,right = 110,bottom = 390)))

 



   
