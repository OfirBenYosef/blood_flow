import cv2
import os
import numpy as np
import BFEpreprocessing as bfe
import pandas as pd
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import os, os.path
import matplotlib.pyplot as plt
import random

class BFEDataset(Dataset):

    def __init__(self, dataset,dir,batch_size = 1,FileType = 'npy', transform=None):
        """
        Args:
            dataset (list): list of all images
            dir (string) : dir path to where the dataset is located
            if dir = False, than dataset contains the images as np matrices

            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.dataset = dataset
        self.transform = transform
        self.dir = dir
        if batch_size:
            self.batch_size = batch_size
        else:
            self.batch_size = len(dataset)
        self.indices = [*range(len(dataset))]
        random.shuffle(self.indices)

    def __len__(self):

        self.len = len(self.dataset)//self.batch_size

        return self.len

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        images = []
        labels = []
        for i in self.indices[idx*self.batch_size : (idx+1)*self.batch_size]:
            if self.dir:
                chosen_image = self.dataset[i]
                image = np.load(os.path.join(self.dir,chosen_image))
                label = chosen_image.split('_')[0]
                if label == 'Good': label = 0
                else: label = 1
            else:
                image = self.dataset[i][0]
                label = self.dataset[i][1]


            if self.transform:
                sample = self.transform(sample)
            images.append(image.transpose([2,0,1]))
            labels.append(label)
        return images,labels

def save_frames(vid_path, frame_list, debug = False):
    '''
    saves frames as images with frame numbers
    :param vid_path:
    :param frame_list: frames numbers to save
    :param debug: if True, prints progress in loading the video
    :return: nothing.
    '''
    vidcap = cv2.VideoCapture(vid_path)
    success, image = vidcap.read()
    count = 0
    frame_list = [int(i) for i in frame_list]
    while success:
        if count in frame_list:
            cv2.imwrite("frame_%06d.tiff" % count, image)  # save frame as JPEG file
            success, image = vidcap.read()
            print('Read a new frame: ', success)
        success, image = vidcap.read()
        count += 1
        if debug and count % 25 == 0:
            print(count)


def get_frame_num(path):
    files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)) and ('frame' in f)]
    frames = [(f.split('_')[1]).split('.')[0] for f in files]
    return files, frames

def get_label_list(path):
    good = []
    poor = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.lower().endswith('txt'):
                text_file = open(os.path.join(root, file), "r")
                data = text_file.read()
                frame_num = (root.split("/")[4].split("_")[1])
                if 'Poor' in data : poor.append(frame_num)
                if 'Good' in data: good.append(frame_num)
                text_file.close()
    return good, poor

def labelme_json_to_dataset(json_path):
    os.system("labelme_json_to_dataset "+json_path+" -o "+json_path.replace(".","_"))

# Press the green button in the gutter to run the script.

def save_frames_by_mask(annotate_path,save_dir):
    labels_dict = {}
    for root, dirs, files in os.walk(annotate_path):
        for file in files:
            if file.lower().endswith('txt'):
                labels = []
                text_file = open(os.path.join(root, file), "r")
                label_data = text_file.read()
                frame_num = (root.split("/")[4].split("_")[1])
                label_data_split = label_data.splitlines()
                for i in range(1,len(label_data_split)):
                    labels.append(label_data_split[i])
                labels_dict[frame_num] = labels
                text_file.close()

                img = bfe.BFE_image(os.path.join(root, 'img.png'), path=True)
                img_masked, mask2 = img.grab_cut()

                labels_img = bfe.BFE_image(os.path.join(root, 'label.png'), path=True)

                save_dir = r'/home/stavb/new_frames'
                for i, i_label in enumerate(labels):
                    mask3 = labels_img.change_color('RGB')[:, :, i] > 0
                    tmp = np.multiply(img_masked, mask3[:, :, np.newaxis])
                    os.chdir(os.path.join(save_dir,i_label))
                    cv2.imwrite(frame_num+'.jpg', cv2.cvtColor(tmp, cv2.COLOR_RGB2BGR))


def color_hist(frame):
    '''
    return color hist for HSV color space
    TODO : maybe need to change to numbers
    :param frame: cv2 RGB image
    :return:
    '''
    img_hsv = cv2.cvtColor(frame,cv2.COLOR_RGB2HSV)
    hsv = ('h','s','v')
    color = ('r','g','b')
    hist = []
    hist_hsv = []
    for i, col in enumerate(hsv):
        hist_hsv.append(cv2.calcHist([img_hsv],[i],None,[256],[0,256]))

    for i, col in enumerate(color):
        hist.append(cv2.calcHist([frame],[i],None,[256],[0,256]))

    return hist,hist_hsv


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
        plt.plot(chi[i,:],color = col)
        plt.legend(color_1)
        #plt.xlim([0,256])
    plt.pause(1)
    plt.close()

def crop_box(x,y,image,len = 20,):
    # crop box for analsys in a [len X len]
    # may change len
    left = int(y - np.ceil(len/2))
    top = int(x - np.ceil(len/2))
    right = int(y + np.ceil(len/2))
    bottom = int(x + np.ceil(len/2))
    box = image[top:bottom,left:right]
    return box

def get_histograms(path):
    label_list = os.listdir(path)
    histograms = {}
    channels = ['red', 'green', 'blue']
    for label in label_list:
        histograms[label] = {}
        label_path=os.path.join(path, label)
        image_files = [f for f in os.listdir(label_path) if f.endswith('.jpg')]
        for image in image_files:
            image_name = image.split('.')[0]
            histograms[label][image_name]={}
            im = os.path.join(label_path, image)
            im = bfe.BFE_image(im, path = True).set_color('RGB',return_img = 1)
            x = 10
            stride = 20
            thershold_value = 150
            box_num = 0
            for x in range (10,im.shape[0],stride):
                for y in range(10, im.shape[1], stride):
                    box = crop_box(x,y,im,stride)
                    gray_box = cv2.cvtColor(box,cv2.COLOR_RGB2HSV)
                    if gray_box.mean() < thershold_value:
                        continue
                    else:
                        histograms[label][image_name][box_num] = {}
                        hist, _ = color_hist(box)
                        for i,channel in enumerate(channels):
                            histograms[label][image_name][box_num][channel] = np.concatenate(hist[i], axis=0 )
                        box_num = box_num + 1
                        A='we'
    return histograms

def save_histograms(path):
    A = get_histograms(path)
    good = A['Good']


    poor = A['Poor']
    names = ['good','poor']
    for i,hist in enumerate([good,poor]):
        pandas_dict = pd.DataFrame.from_dict({(i, j): hist[i][j]
                                for i in hist.keys()
                                for j in hist[i].keys()},
                               orient='columns')
        name = names[i] + '.xlsx'
        new_path = os.path.join(path,name)
        pandas_dict.to_excel(new_path)



def flatten_dict(nested_dict):
    res = {}
    if isinstance(nested_dict, dict):
        for k in nested_dict:
            flattened_dict = flatten_dict(nested_dict[k])
            for key, val in flattened_dict.items():
                key = list(key)
                key.insert(0, k)
                res[tuple(key)] = val
    else:
        res[()] = nested_dict
    return res


def nested_dict_to_df(values_dict):
    flat_dict = flatten_dict(values_dict)
    df = pd.DataFrame.from_dict(flat_dict, orient="index")
    df.index = pd.MultiIndex.from_tuples(df.index)
    df = df.unstack(level=-1)
    df.columns = df.columns.map("{0[1]}".format)
    return df

def too_close(frame,list):
    if len(list)<1 :
        return False
    for num in list:
        if (int(frame) - int(num)) < (24*4-1):
            return True
    return False



if __name__ == '__main__':

    pass

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
