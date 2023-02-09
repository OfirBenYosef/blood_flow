import cv2
import os
import numpy as np
import BFEpreprocessing as bfe
import pandas as pd
import sklearn
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
import json
import random
from sklearn.neighbors import KNeighborsClassifier
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
import seaborn as sns
from sklearn.decomposition import PCA
import time
from BFEGaborNet import GaborNN
from torchvision import transforms
import utils
import pickle

def cuda(x):
    return x.cuda() if torch.cuda.is_available() else x
    # return x.cuda(non_blocking=True) if torch.cuda.is_available() else x


class SaveBestModel:
    """
    Class to save the best model while training. If the current epoch's
    validation loss is less than the previous least less, then save the
    model state as best_model_FreeText, where FreeText is an initiation input.
    :param FreeText (Optional) : (str)
    :param path (optional) : (str) path to saving location for models
    :return: nothing
    """

    def __init__(
            self, best_valid_loss=float('inf'), FreeText=False, path=False
    ):
        self.best_valid_loss = best_valid_loss
        self.FreeText = FreeText
        self.path = path

    def __call__(
            self, current_valid_loss,
            epoch, model, optimizer, criterion
    ):
        if current_valid_loss < self.best_valid_loss:
            self.best_valid_loss = current_valid_loss
            print(f"\nBest validation loss: {self.best_valid_loss}")
            print(f"\nSaving best model for epoch: {epoch + 1}\n")
            if self.FreeText:
                save_path = os.path.join(self.path, 'best_model_' + self.FreeText + '.pth')
            else:
                save_path = 'best_model.pth'
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
            }, save_path)


def TrainGabornet(dataset, dir, split=[0.8, 0.12, 0.08]):
    """
    Train Gabornet with labeled 20x20 patches.
    COMMENTS: we saved the patches in npy files, each contains X patches. this need to be changed for future
              implementations
    :param dataset : (list) list of all .npy files contains the data
    :param dir : (str) path to dir contains the .npy files
    :param split: (list) split ratios for [train set, test set,val set] in this order
    :return: train accuracy
    """

    train_set, test_set, val_set = torch.utils.data.random_split(dataset, split)

    # load the data using custom dataloader
    train_data = utils.BFEDataset(train_set, dir, batch_size=64 * 16)
    test_data = utils.BFEDataset(test_set, dir, batch_size=False)
    val_data = utils.BFEDataset(val_set, dir, batch_size=False)

    # set the model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = GaborNN().to(device)
    optimizer = torch.optim.AdamW(model.parameters())
    epochs = 70
    save_best_model = SaveBestModel(FreeText='GaborNet5', path='/home/stavb/PycharmProjects/outputs')

    ## loss criterion
    criterion = nn.BCELoss()
    # criterion = nn.CrossEntropyLoss()   BCE is slightly better

    val_acc = []
    val_loss = []

    # training loop
    for epoch in range(1, epochs + 1):
        torch.cuda.empty_cache()

        model.train()  # put in training mode
        running_loss = 0.0
        epoch_time = time.time()

        for idx, (inputs, labels) in enumerate(train_data):
            # get the inputs

            inputs = torch.from_numpy(np.array(inputs)).type(torch.FloatTensor).to(device)

            labels = torch.from_numpy(np.array(labels)).type(torch.FloatTensor).to(device)
            # forward + backward + optimize
            try:
                outputs = model(inputs).reshape(-1)  # forward pass
            except:
                print("couldn't compute output")
                break
            loss = criterion(outputs, labels)  # calculate the loss
            # always the same 3 steps
            optimizer.zero_grad()  # zero the parameter gradients
            loss.backward()  # backpropagation
            optimizer.step()  # update parameters

            # print statistics
            running_loss += loss.data.item()

        # Normalizing the loss by the total number of train batches
        running_loss /= (len(train_data) * len(train_data[0][0]))  # is it the right way?

        # Calculate Val set accuracy of the existing model
        # divide to small batches for memory reasons
        model.eval()
        with torch.no_grad():
            X_Val, y_Val = val_data[0]
            X_Val = torch.from_numpy(np.array(X_Val)).type(torch.FloatTensor).to(device)
            labels_Val = torch.from_numpy(np.array(y_Val)).type(torch.FloatTensor).to(device)
            outputs_Val = model(X_Val).reshape(-1)
            loss_val = criterion(outputs_Val, labels_Val) / len(y_Val)
            acc_val = (outputs_Val.detach().cpu().numpy().round() == y_Val).mean()
            epoch_time = time.time() - epoch_time
            print("Epoch: {} | Val loss: {} | Val accuracy: {}% | Time: {}s ".format(epoch, loss_val, acc_val,
                                                                                     epoch_time))
            save_best_model(
                loss_val, epoch, model, optimizer, criterion
            )
            val_acc.append(acc_val)
            val_loss.append(loss_val)

    print('==> Finished Training ...')

    # final test check

    X_test, y_test = test_data[0]
    X_test = torch.from_numpy(np.array(X_test)).type(torch.FloatTensor).to(device)
    model.eval()
    with torch.no_grad():
        labels_test = torch.from_numpy(np.array(y_test)).type(torch.FloatTensor).to(device)
        # outputs_train = model(X_train)
        outputs_test = model(X_test).reshape(-1)
        test_loss = criterion(outputs_test, labels_test)
        acc_test = (outputs_test.detach().cpu().numpy().round() == y_test).mean()

    print(f'test acc = {acc_test}, test loss = {test_loss}')
    # plotting the loss
    plt.figure()
    N = 5
    plt.plot(torch.tensor(np.convolve(val_acc, np.ones(N) / N, mode='valid'), device='cpu'))
    plt.title('Loss vs Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Val accuracy')
    plt.show()

    return (acc_test)


class CIFAR_CNN(nn.Module):
    """CNN for the SVHN Datset"""

    def __init__(self):
        """CNN Builder."""
        super(CIFAR_CNN, self).__init__()

        # Conv Layer block 1
        self.conv_layer = nn.Sequential(

            # Conv Layer block 1
            nn.Conv2d(in_channels=24, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv Layer block 2
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.2),

            # Conv Layer block 3
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.2),
        )

        self.residual_layer = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),

        )

        self.fc_layer1 = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True)
        )
        self.fc_layer2 = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True)
        )
        self.fc_layer3 = nn.Sequential(
            nn.Linear(64, 1)
        )

    def forward(self, x):
        """Perform forward."""
        # conv layers
        x = self.conv_layer(x)
        for i in range(3):
            x = self.residual_layer(x) + x
        # # flatten
        x = x.view(x.size(0), -1)

        # # fc layer
        x = torch.relu(self.fc_layer1(x))
        x = torch.relu(self.fc_layer2(x) + x)
        x = torch.sigmoid(self.fc_layer3(x))

        return x


def useCIFAR_CNN(X_train, y_train, X_test, y_test, sizes=False, show=True, lr=0.001):
    ''''''
    # time to train our model
    # hyper-parameters
    batch_size = X_train.shape[0] // 6
    learning_rate = 1e-5
    epochs = 80

    # loss criterion
    criterion = nn.BCELoss()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # build our model and send it to the device
    model = CIFAR_CNN().to(device)  # no need for parameters as we alredy defined them in the class

    # optimizer - SGD, Adam, RMSProp...
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    test_acc = []
    train_acc = []

    sc = StandardScaler()
    # X_train = sc.fit_transform(X_train)
    # X_test = sc.fit_transform(X_test)

    X_train = torch.from_numpy(X_train).type(torch.FloatTensor)
    y_train = torch.from_numpy(y_train).type(torch.FloatTensor)
    X_train = X_train.to(device)
    y_train = y_train.to(device)
    X_test = torch.from_numpy(X_test).type(torch.FloatTensor)

    # training loop
    for epoch in range(1, epochs + 1):
        model.train()  # put in training mode
        running_loss = 0.0
        epoch_time = time.time()

        for i in range(X_train.shape[0] // batch_size):
            # get the inputs
            inputs = X_train[batch_size * i:batch_size * (i + 1)]
            labels = y_train[batch_size * i:batch_size * (i + 1)]

            # forward + backward + optimize
            outputs = model(inputs).reshape(-1)  # forward pass
            loss = criterion(outputs, labels)  # calculate the loss
            # always the same 3 steps
            optimizer.zero_grad()  # zero the parameter gradients
            loss.backward()  # backpropagation
            optimizer.step()  # update parameters

            # print statistics
            running_loss += loss.data.item()

        # Normalizing the loss by the total number of train batches
        running_loss /= len(X_train)

        # Calculate training/test set accuracy of the existing model
        outputs_train = model(X_train.to(device))
        outputs_test = model(X_test.to(device))
        acc_train = (outputs_train.reshape(-1).round() == y_train).detach().cpu().numpy().mean()
        acc_test = (outputs_test.reshape(-1).detach().cpu().numpy().round() == y_test).mean()

        epoch_time = time.time() - epoch_time
        # log += "Epoch Time: {:.2f} secs".format(epoch_time)
        # print(log)

        if i % (1) == 0:
            train_acc.append(acc_train)
            test_acc.append(acc_test)
            print("Epoch: {} | Loss: {} | Training accuracy: {}% | Test accuracy: {}% | ".format(epoch,
                                                                                                 running_loss,
                                                                                                 acc_train,
                                                                                                 acc_test))
    if show:

        # printing the accuracy
        plt.plot(torch.tensor(train_acc, device='cpu'), label='train acc')
        plt.plot(torch.tensor(test_acc, device='cpu'), label='test acc')
        plt.title('Accuracy vs Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.show()

        # # save model
        # if epoch % 20 == 0:
        #     print('==> Saving model ...')
        #     state = {
        #         'net': model.state_dict(),
        #         'epoch': epoch,
        #     }
        #     if not os.path.isdir('checkpoints'):
        #         os.mkdir('checkpoints')
        #     torch.save(state, './checkpoints/svhn_cnn_ckpt.pth')

    print('==> Finished Training ...')
    return (acc_test)


class NN_Net(nn.Module):
    def __init__(self, input_shape):
        super(NN_Net, self).__init__()
        self.fc1 = nn.Linear(input_shape, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 16)
        self.fc4 = nn.Linear(128, 16)
        self.fc5 = nn.Linear(16, 1)
        self.input_shape = input_shape

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        # x = torch.relu(self.fc4(x))
        x = torch.sigmoid(self.fc5(x))
        return x

    def set_params_size(self, sizes):
        self.fc1 = nn.Linear(self.input_shape, sizes[0])
        self.fc2 = nn.Linear(sizes[0], sizes[1])
        self.fc3 = nn.Linear(sizes[1], sizes[2])
        self.fc4 = nn.Linear(sizes[2], sizes[3])
        self.fc5 = nn.Linear(sizes[3], 1)


def trainNN(X_train, y_train, X_test, y_test, sizes=False, show=False, lr=0.001, prefix='',
            save_model = False):
    ''''''
    learning_rate = lr
    epochs = 12000
    model = NN_Net(input_shape=X_train.shape[1])
    if sizes: model.set_params_size(sizes)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    criterion = nn.BCELoss()
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.fit_transform(X_test)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if save_model:
        save_best_model = SaveBestModel(FreeText='simpleNN_' + prefix, path='/home/stavb/PycharmProjects/outputs')

    X_train = torch.from_numpy(X_train).type(torch.FloatTensor)
    y_train = torch.from_numpy(y_train).type(torch.FloatTensor)
    X_train = X_train.to(device)
    y_train = y_train.to(device)
    X_test = torch.from_numpy(X_test).type(torch.FloatTensor)
    # y_test = torch.from_numpy(y_test).type(torch.FloatTensor)
    X_test = X_test.to(device)
    # y_test = y_test.to(device)
    model = model.to(device)

    # forward loop
    losses = []
    train_accur = []
    test_acc = []
    for epoch in range(epochs):

        # calculate output
        model.train()
        output = model(X_train)

        # calculate loss
        loss = criterion(output, y_train.reshape(-1, 1))

        # accuracy
        predicted = model(torch.tensor(X_test, dtype=torch.float32))

        # backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % (50) == 0:
            train_accur.append((output.reshape(-1).round() == y_train).detach().cpu().numpy().mean())
            losses.append(loss)
            test_acc.append((predicted.reshape(-1).detach().cpu().numpy().round() == y_test).mean())
            if show: print("epoch {}\tloss : {}\t train accuracy : {}\t test accuracy : {}".format(epoch, loss, train_accur[-1],
                                                                                          test_acc[-1]))

    if show:
        # plotting the loss
        plt.figure()
        plt.plot(torch.tensor(losses, device='cpu'))
        plt.title('Loss vs Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('loss')
        plt.show()

        # printing the accuracy
        plt.figure()
        plt.plot(torch.tensor(train_accur, device='cpu'))
        plt.plot(torch.tensor(test_acc, device='cpu'))
        plt.title('Accuracy vs Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend(['train','test'])
        plt.show()

    pass

    torch.save(model.state_dict(), 'TrainedNN.pth')
    if save_model:
        save_best_model(
            loss, epoch, model, optimizer, criterion
        )
    output = predicted.reshape(-1).detach().cpu().numpy().round()
    return test_acc[-1], output


def UseKNN(X_train, y_train, X_test, y_test, n_neighbors=12):
    neigh = KNeighborsClassifier(n_neighbors=n_neighbors)
    neigh.fit(X_train, y_train)
    score = neigh.score(X_test, y_test)
    return score


def UseLinearSVC(X_train, y_train, X_test, y_test):
    clf = make_pipeline(StandardScaler(), LinearSVC(random_state=0, tol=1e-5))
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    output = clf.predict(X_test)
    return score, output


def UseSVC(X_train, y_train, X_test, y_test):
    clf = make_pipeline(StandardScaler(), sklearn.svm.SVC(gamma='auto'))
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    output = clf.predict(X_test)
    return score, output


def upload_hist(path):
    hist = pd.read_excel(path, header=None)
    return hist.to_numpy()


def chi_square_hist(hist1, hist2):
    chi = []
    hist2 = np.stack(hist2)
    for channel in range(0, hist1.shape[0]):
        chi.insert(channel, 0)
        temp = int(0)
        for i in range(0, hist1.shape[1]):
            if hist1[channel, i] == 0.0 and hist2[channel, i] == 0.0:
                temp += 0
            else:
                temp += (np.square(hist1[channel, i] - hist2[channel, i]) / (hist1[channel, i] + hist2[channel, i]))
        chi[channel] = temp
    return chi


def run_classifier_single(X, y, test_ratio=0.85, num_of_tries=1):
    '''
    train classifiers for X dataset X with labels y
    :param X: (list) Nxd data
    :param y: (list) Nx1 labels
    :param test_ratio: train-test ratio
    :param num_of_tries: num of iterations
    :return:
    '''
    scores = np.zeros((num_of_tries, 3))
    test_size = int(np.floor(len(X) * test_ratio))

    for try_num in range(num_of_tries):
        indices = [*range(X.shape[0])]
        random.shuffle(indices)

        X_train = X[indices[:test_size]]
        X_test = X[indices[test_size:]]
        y_test = y[indices[test_size:]]
        y_train = y[indices[:test_size]]

        print("try num: ", try_num)
        scores[try_num, 0], _ = UseLinearSVC(X_train, y_train, X_test, y_test)
        scores[try_num, 1], _ = UseSVC(X_train, y_train, X_test, y_test)
        scores[try_num, 2], _ = trainNN(X_train, y_train, X_test, y_test, show=False,
                                        sizes=False, lr=0.005)

    return scores


def make_data_func(np_folder_path, save_path, prefix):
    '''
    creates and save data as numpy arrays
    each type of data (fft,histograms,gabor filters, and labels)
    :param np_folder_path:
    :param save_path:
    :param prefix: (str)
    :return:
    '''
    np_format = 'npy'
    X_all = []
    X_fft = []
    X_hist = []
    X_hist_hsv = []
    X_gabor = []
    y = []
    count = 0
    for root, dirs, files in os.walk(np_folder_path):
        for filename in files:
            if filename.lower().endswith(np_format):
                print(filename)
                data = np.load(os.path.join(root, filename), allow_pickle=True)
                for i in range(data[0].shape[0]):
                    X_fft.append(data[0][i])
                    X_fft = data[0][i]
                    hist = data[1][i]
                    hist = hist.reshape(3, -1).T
                    hist = hist[1:, :]
                    hist = hist.reshape(-1, 1)
                    X_hist.append(np.concatenate(hist, axis=0))
                    X_hist = np.concatenate(hist, axis=0)

                    hist = data[3][i]
                    hist = hist.reshape(3, -1).T
                    hist = hist[1:, :]
                    hist = hist.reshape(-1, 1)
                    X_hist_hsv.append(np.concatenate(hist, axis=0))
                    X_gabor_item = np.concatenate(data[2][i].reshape(-1, 1), axis=0)
                    X_gabor.append(X_gabor_item)
                    X_all.append(np.concatenate([X_fft, X_hist, X_gabor]))
                    X_all.append(np.concatenate([X_fft, X_hist, X_gabor]))
                    if "Good" in filename:
                        y.append(0)
                    else:
                        y.append(1)
                count = count + 1

    X_fft = np.array(X_fft)
    X_hist = np.array(X_hist)
    X_gabor = np.array(X_gabor)
    X_hist_hsv = np.array(X_hist_hsv)
    y = np.array(y)
    np.save(os.path.join(save_path, 'X_fft' + prefix + '.npy', X_fft))
    np.save(os.path.join(save_path, 'X_hist' + prefix + '.npy', X_hist))
    np.save(os.path.join(save_path, 'X_hist_hsv' + prefix + '.npy', X_hist_hsv))
    np.save(os.path.join(save_path, 'X_gabor' + prefix + '.npy', X_gabor))
    np.save(os.path.join(save_path, 'y' + prefix + '.npy', y))


def train_classifiers(np_folder_path, save_path, prefix=False, make_data=False, test_ratio=0.8, num_of_tries=50,
                      n_components=False, show = False):
    '''

    :param np_folder_path:
    :param save_path:
    :param prefix:
    :param make_data:
    :param test_ratio:
    :param num_of_tries:
    :param n_components:
    :return:
    '''
    if prefix:
        prefix = '_' + prefix
    else:
        prefix = ''

    if make_data: make_data_func(np_folder_path, save_path, prefix)

    # load the data

    X_fft = np.load(os.path.join(save_path, 'X_fft' + prefix + '.npy'), allow_pickle=True)
    X_hist = np.load(os.path.join(save_path, 'X_hist' + prefix + '.npy'), allow_pickle=True)
    X_hist_hsv = np.load(os.path.join(save_path, 'X_hist_hsv' + prefix + '.npy'), allow_pickle=True)
    X_gabor = np.load(os.path.join(save_path, 'X_gabor' + prefix + '.npy'), allow_pickle=True)
    y = np.load(os.path.join(save_path, 'y' + prefix + '.npy'), allow_pickle=True)

    if n_components:
        pca = PCA(n_components=n_components)
        pca.fit(X_gabor.T)
        X_gabor = pca.components_.T

    X_fft_hist = np.concatenate([X_fft, X_hist], axis=1)
    X_list = [X_fft_hist, X_fft, X_hist, X_gabor, X_hist_hsv]
    X_list = [X_fft]  ##### delete
    scores = np.zeros((len(X_list), num_of_tries, 3))
    outputs = np.zeros((len(X_list), num_of_tries, 3), dtype=object)
    GT = np.zeros(num_of_tries, dtype=object)

    for try_num in range(num_of_tries):
        indices = [*range(X_fft.shape[0])]
        random.shuffle(indices)
        test_size = int(np.floor(len(X_fft) * test_ratio))
        y_test = y[indices[test_size:]]
        y_train = y[indices[:test_size]]
        GT[try_num] = y_test

        for X_idx, X in enumerate(X_list):
            print("try num: ", try_num, " X idx ", X_idx)
            X_train = X[indices[:test_size]]
            X_test = X[indices[test_size:]]

            scores[X_idx, try_num, 0], outputs[X_idx, try_num, 0] = UseLinearSVC(X_train, y_train, X_test, y_test)
            scores[X_idx, try_num, 1], outputs[X_idx, try_num, 1] = UseSVC(X_train, y_train, X_test, y_test)
            scores[X_idx, try_num, 2], outputs[X_idx, try_num, 2] = trainNN(X_train, y_train, X_test, y_test,
                                                                            sizes=False, lr=0.005, prefix = '0',show = show)
            pass

    ## X will contain the outputs of all the different models we tried
    Avgs = scores.mean(axis=1)
    plt.figure()
    plt.plot(['Linear SVC','SVC','NN'],Avgs.T,'o',linestyle="--")
    plt.legend(['fft and histogram', 'fft', 'histogram', 'gabor', 'histogram hsv'])
    plt.xlabel('Classifier')
    plt.ylabel('Accuracy %')

    ## delete later!
    Avgs = np.load('AVGS.npy', allow_pickle=True)
    X = []
    y = []
    GT = np.stack(GT)
    for try1 in range(num_of_tries):
        A = np.stack(outputs[:, try1, 1])
        for indx in range(A.shape[1]):
            X.append(A[:, indx])
            y.append(GT[try1, indx])

    X = np.array(X)
    y = np.array(y)

    scores_final1 = run_classifier_single(X, y,num_of_tries=10)
    Avgs1 = scores_final1.mean(axis=0)
    scores_final2 = run_classifier_single(X[:,[1,2,4]], y,num_of_tries=10)
    Avgs2 = scores_final2.mean(axis=0)
    pass


def return_array(str):
    '''
    convert a string of array to array
    for example '[5,2]' => [5,2]
    :param str: string of array
    :return: array
    '''
    str = str.replace('\n', '').replace("[", '').replace("]", '')
    arr = np.fromstring(str, dtype=int, sep='.')
    return arr


class Classify():
    def __init__(
            self, path_svm, path_nn, path_gabor
    ):
        self.path_svm = path_svm
        self.path_nn = path_nn
        self.path_gabor = path_gabor
        file = open(path_svm, 'rb')
        self.svmTrained =  pickle.load(file)
        self.NNTrained = NN_Net(input_shape=96)
        self.NNTrained.load_state_dict(torch.load(path_nn))
        self.NNTrained.eval()
        self.GaborNetTrained =  GaborNN()
        #self.GaborNetTrained.load_state_dict(torch.load(self.path_gabor))
        pass

    def __call__(self, data):
        with torch.no_grad():
            hist = data[1][0]
            hist = hist.reshape(3, -1).T
            hist = hist[1:, :]
            hist = np.array(hist.reshape(1, -1))
            A = (self.NNTrained(torch.from_numpy(data[0][0:96].T).float()))[0].numpy().round()
            B = self.svmTrained.predict(hist)
            #B = self.svmTrained.predict(data[1][0].reshape(1, -1))

        return [A,B]


if __name__ == '__main__':
    pass