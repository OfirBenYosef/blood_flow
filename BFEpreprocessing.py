from skimage import data
from skimage import color
import os
from vidstab import VidStab
from skimage.filters import meijering, sato, frangi, hessian, threshold_otsu, gaussian
import matplotlib.pyplot as plt
import numpy as np
import cv2
from skimage import exposure
from skimage.measure import label, regionprops, regionprops_table
from skimage.segmentation import active_contour
from skimage.color import rgb2gray
import plotly
import plotly.express as px
import plotly.graph_objects as go
from skimage import (
    color, feature, filters, measure, morphology, segmentation, util
)
import generate_mask as gm


class BFE_image:
    def __init__(self,img = 0, path = False, format = 'BGR',show=0,color = 'RGB', equalize = 0):
        """
        param img : image. default - cv2 BGR WxHx3 numpy ndarray
        param path :True if img contains a path to an image and not the image itself. default False
        :param format:
        :param show:
        :param color: (string) image's colorscale. possible values: 'RGB' (default), 'GRAY', 'GREEN', 'RED', 'BLUE',
            'HSV', 'H', 'S', 'V', 'B'(binary)
        :param equalize:
        """
        if path: self.rgb_img = cv2.cvtColor(cv2.imread(img, cv2.IMREAD_COLOR),  cv2.COLOR_BGR2RGB)
        else:
            self.rgb_img = cv2.cvtColor(img)
            ### TODO - add options for RGB, grayscale images
        self.color = 'RGB'
        self.img = self.rgb_img
        self.shape = self.img.shape
        self.dim = self.shape[0]

        self.set_color(colorscale = color)
        if equalize:
            self.equalize(set_img = 1, color = 'GREEN', limit = 0.03)
        if show: self.show()

    def change_color(self, colorscale='RGB'):
        """
        return original RGB image in colorscale. doesn't change the image in class
        :param colorscale: possible values: 'RGB' (default), 'GRAY', 'GREEN', 'RED', 'BLUE',
            'HSV', 'H', 'S', 'V', 'B'(binary)
        :return: original image in colorscale
        """
        if colorscale == 'RGB':
            img = self.rgb_img
        elif colorscale == 'GRAY':
            img = cv2.cvtColor(self.rgb_img, cv2.COLOR_BGR2GRAY)
        elif colorscale == 'HSV':
            img = cv2.cvtColor(self.rgb_img, cv2.COLOR_BGR2HSV)
        elif colorscale == 'RED':
            img =  self.rgb_img[:,:,0]
        elif colorscale == 'GREEN':
            img =  self.rgb_img[:, :, 1]
        elif colorscale == 'BLUE':
            img =  self.rgb_img[:, :, 2]
        elif colorscale == 'H':
            img = cv2.cvtColor(self.rgb_img, cv2.COLOR_BGR2HSV)[:,:,0]
        elif colorscale == 'S':
            img = cv2.cvtColor(self.rgb_img, cv2.COLOR_BGR2HSV)[:,:,1]
        elif colorscale == 'V':
            img = cv2.cvtColor(self.rgb_img, cv2.COLOR_BGR2HSV)[:,:,2]
        elif colorscale == 'B':
            img = threshold(cv2.cvtColor(self.rgb_img, cv2.COLOR_BGR2GRAY))
        else:
            print("not a valid value")
        return img

    def set_color(self,colorscale,return_img = 0):
        """

        :param return_img:
        :param colorscale:
        :return:
        """
        if self.is_valid_color(colorscale):
            if colorscale.lower() == self.color:
                pass
            else:
                self.color = colorscale.upper()
                self.img = self.change_color(self.color)
                self.update_dim()
            if return_img:
                return self.img
        else:
            print("not a valid value")

    def is_valid_color(self,colorscale):
        """
        :param colorscale:
        :return:
        """
        valid_colors = ['RGB', 'GRAY', 'GREEN', 'RED', 'BLUE',
            'HSV', 'H', 'S', 'V', 'B']
        return colorscale.upper() in valid_colors

    def update_dim(self):
        self.shape = self.img.shape
        self.dim = self.img_dim(self.shape)

    def img_dim(self,shape):
        try:
            return shapelabel_list[2]
        except:
            return 1

    def equalize(self, set_img = 1, color = 'GREEN', limit = 0.03):
        """
        returns
        :param set_img:change image to equalized
        :param color: default GREEN. which image to equalize
        :param limit:
        :return:
        """

        if self.is_valid_color(color):
            img = self.change_color(color)
        else:
            img = self.change_color('GREEN')
        dim = self.img_dim(img.shape)
        if dim == 1:
            img= exposure.equalize_adapthist(img, clip_limit=limit)
        else:
            for i in range(dim):
                img[:,:,i] = exposure.equalize_adapthist(img[:,:,i], clip_limit=limit)
        if set:
            self.img = img
        return img

    def F_frangi(self, sigmas=range(1, 3, 1), alpha=0.5, beta=0.5, gamma=15, show=0):
        img_adapteq = self.equalize()
        if show: cv2.imshow('image', img_adapteq)
        image_f = 10000 * frangi(np.array(img_adapteq), sigmas=sigmas, scale_range=None, scale_step=None, alpha=alpha,
                                 beta=beta, gamma=gamma, black_ridges=True, mode='reflect', cval=0)
        # image_m = meijering(np.array(img_adapteq))
        # image_s = sato(np.array(img_adapteq),black_ridges=True, mode='reflect')
        if show:
            cv2.imshow('image', image_f, cmap="gray")
            cv2.waitKey(label_list0)
        return image_f

    def segment_images(self):
        '''
        orientation histogram
        :return:  nothing
        '''
        binary_frame = threshold(self.F_frangi())
        self.find_vessels(binary_frame)

    def find_vessels(self, binary_frame, show = True):
        label_img = label(binary_frame)
        regions = regionprops(label_img)

        orientations = np.empty(regions.__len__())
        i = 0
        for prop in regions:
            orientations[i] = prop.orientation
            i += 1
        print('...')
        hist, bin_edges = np.histogram(orientations, density=True)
        _ = plt.hist(orientations)  # arguments are passed to np.histogram
        plt.title("Histogram of orientation")
        # plt.show()

    def colon_seg(self, ax=0):
        '''
        segmentation by active_contour. not good
        :param ax:
        :return: grayscale image, with two active contours "snakes" on it
        '''
        img = self.change_color('GRAY')
        s = np.linspace(0, 2 * np.pi, 400)
        r = 240 + 300 * np.sin(s)
        c = 370 + 350 * np.cos(s)
        init = np.array([r, c]).T
        f_img = sato(np.array(img), black_ridges=True, mode='reflect')
        # = gaussian(img, 0.03, preserve_range=False)

        snake = active_contour(img, init, alpha=0.015, beta=10, gamma=0.001)

        if not ax: fig, ax = plt.subplots(figsize=(7, 7))
        ax.imshow(img, cmap=plt.cm.gray)
        ax.plot(init[:, 1], init[:, 0], '--r', lw=3)
        ax.plot(snake[:, 1], snake[:, 0], '-b', lw=3)
        ax.set_xticks([]), ax.set_yticks([])
        ax.axis([0, img.shape[1], img.shape[0], 0])
        if not ax: plt.show()

    def r_prop(self):
        '''
        takes too much time, do not use
        TODO : see if we can make it more efficient
        :return: RGB image, with different elements colored
        '''

        img_b = threshold(self.F_frangi())
        # Binary image, post-process the binary mask and compute labels
        labels = measure.label(img_b)

        fig = px.imshow(self.rgb_img)
        fig.update_traces(hoverinfo='skip')  # hover is only for label info

        props = measure.regionprops(labels, self.rgb_img)
        properties = ['area', 'eccentricity', 'perimeter', 'intensity_mean']

        # For each label, add a filled scatter trace for its contour,
        # and display the properties of the label in the hover of this trace.
        for index in range(1, labels.max()):
            label_i = props[index].label
            contour = measure.find_contours(labels == label_i, 0.5)[0]
            y, x = contour.T
            hoverinfo = ''
            for prop_name in properties:
                try: hoverinfo += f'<b>{prop_name}: {getattr(props[index], prop_name):.2f}</b><br>'
                except: pass
            fig.add_trace(go.Scatter(
                x=x, y=y, name=label_i,
                mode='lines', fill='toself', showlegend=False,
                hovertemplate=hoverinfo, hoveron='points+fills'))

        plotly.io.show(fig)


    def grab_cut(self):
        img = self.rgb_img
        mask = np.zeros(img.shape[:2],np.uint8)
        bgdModel = np.zeros((1,65),np.float64)
        fgdModel = np.zeros((1,65),np.float64)
        rect = (10,10,650,590)
        # (start_x, start_y, width, height).
        cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
        mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
        img = img*mask2[:,:,np.newaxis]
        return img, mask2
        #plt.imshow(img),plt.colorbar(),plt.show()

    def color_hist(self, show=True, color='RGB'):
        rgb_colors = ('r', 'g', 'b')
        if color == 'HSV':
            img = self.change_color('HSV')
            channels = ('h', 's', 'v')
        else:
            img = self.rgb_img
            channels = rgb_colors
        hist = []
        if show: plt.figure()
        for i, col in enumerate(rgb_colors):
            histr_1 = cv2.calcHist([img], [i], None, [256], [0, 256])
            if show:
                plt.plot(histr_1, color=col)
                plt.xlim([0, 256])
            hist.append(histr_1)
        if show:
            plt.legend(channels)
            plt.show()
        return hist

    def tool_rec(self, model_path='', model_type='', show = True):
        if model_path == '':
            base_models_path = r'/home/stavb/robot-surgery-segmentation-master/data/models'
            model_path = os.path.join(base_models_path, 'unet16_binary_20', 'model_1.pt')
            model_type = 'UNet16'
        mask_array = gm.generate_mask(model_path, self.rgb_img, model_type, show=0, cut=[576, 704])
        mask = (mask_array > 0).astype(np.uint8)
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contour_mask = np.ones(mask.shape[:2], dtype="uint8") * 255
        for c in contours:
            if c.size < 150:
                cv2.drawContours(contour_mask, [c], -1, 0, -1)
        mask2 = cv2.bitwise_and(mask, mask, mask=contour_mask)
        mask_overlayed = gm.mask_overlay(self.rgb_img, mask2)
        if show : plt.imshow(mask_overlayed)
        return mask2


def threshold(image, show=0):
    thresh = threshold_otsu(image)
    print(thresh)
    binary = np.array(image > thresh, dtype=bool)
    if show:
        plt.imshow(binary, cmap="gray")
        plt.show()
    return binary







def video_stab(input_path,output_path):
    # Using defaults
    stabilizer = VidStab(kp_method='STAR')
    stabilizer.stabilize(input_path=input_path, output_path=output_path)

    # Using a specific keypoint detector
    #
    # stabilizer.stabilize(input_path='cut2.mp4', output_path='stable_video.avi')

    # Using a specific keypoint detector and customizing keypoint parameters
    # stabilizer = VidStab(kp_method='FAST', threshold=42, nonmaxSuppression=False)
    # stabilizer.stabilize(input_path='cut2.mov', output_path='stable_video.avi')


def play_video(path,show=True):
    # load the video from path
    cap = cv2.VideoCapture(path)
    while (cap.isOpened()):
        ret, frame = cap.read()
        b_frame = np.uint8(1000 * threshold(F_frangi(exposure.equalize_adapthist(frame[:, :, 1], clip_limit=0.03))))
        if show:
            cv2.imshow('frame', b_frame)
            plt.axis('off')
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()


def read_video(vid_path, show = False, debug = False):
    # load video and save it as array to future work, can be change to get video for camara.
    # save only the green channel <- change it !!!

    cap = cv2.VideoCapture(vid_path)

    n = int(cap.get(5))
    w = int(cap.get(3))
    h = int(cap.get(4))
    if debug : print(n, w, h)
    vid = []
    vid_fc = []
    i = 0
    while (cap.isOpened()):
        if debug:
            if i%100 ==0 : print("frame # ",i)
        # vid_capture.read() methods returns a tuple, first element is a bool
        # and the second is frame
        ret, frame = cap.read()
        if ret == True:
            if show : cv2.imshow('Frame', frame)
            vid.append(frame)
            vid_fc.append(frame)
            i = i + 1
            # 20 is in milliseconds, try to increase the value, say 50 and observe
            key = cv2.waitKey(10)

            if key == ord('q'):
                break
        else:
            break
    # Release the video capture object
    cv2.destroyAllWindows()
    cap.release()
    vid = np.array(vid)
    vid_fc = np.array(vid_fc)
    return vid, vid_fc


if __name__ == '__main__':
    path = r'Data/frames'

