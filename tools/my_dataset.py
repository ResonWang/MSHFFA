import os
import random
import pickle
import cv2.cv2 as cv2
from PIL import Image
from torch.utils.data import Dataset
from ctypes import cdll, POINTER, c_ubyte, c_int, c_float, string_at
from numpy.ctypeslib import ndpointer
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
# print(os.path.realpath(__file__))
# from tools.filters_hw2 import *

random.seed(1)

def wmf_fiter(img, so_path='/AI/videoDetection/algorithm/wmf/output.so', d=15, sigma=25):
    """
    加权中值滤波
    :param img: 灰度图，0-255
    :param so_path:
    :param r: 滤波直径,整数
    :param sigma: 25
    :return:
    """
    so = cdll.LoadLibrary(so_path)
    so.WMF_filter.argtypes = (c_int, c_int, ndpointer(dtype=c_ubyte), ndpointer(dtype=c_ubyte), c_int, c_float, c_int, c_int, c_int)
    so.WMF_filter.restype = POINTER(c_ubyte)  # POINTER(c_ubyte) 跟c_void_p都可以
    # typelist = [b'exp', b'iv1', b'iv2', b'cos', b'jac', b'off']
    # type = typelist[0]
    # pubyType = c_char_p(type)
    retPoint = so.WMF_filter(img.shape[0], img.shape[1], img, img, d//2, sigma, 256, 256, 1)  # 这里去传参使用      Mat myFunction(Mat &I, Mat &F, int r, int nI, int nF, float denom);
    b = string_at(retPoint, img.shape[1] * img.shape[0])  # 类似于base64
    ret_img = np.frombuffer(b, np.uint8).reshape(img.shape[0], img.shape[1])  # 转array,但是维度不是图片
    return ret_img


class BreastDataset(Dataset):
    # def __init__(self, data_dir="/AI/videoDetection/data/public_us_dataset/breast_lesion/rawframes/", train=True, transform=None):
    #     """
    #     POCUS Dataset
    #         param data_dir: str
    #         param transform: torch.transform
    #     """
    #     self.label_name = {"benign": 0, "malignant": 1}
    #     class_0 = os.listdir(data_dir)[0]       # benign文件夹
    #     class_1 = os.listdir(data_dir)[1]       # malignant文件夹
    #     dataset0 = os.listdir(data_dir+class_0)#[0:15]
    #     dataset1 = os.listdir(data_dir+class_1)#[0:15]
    #     train_dirs_0, val_dirs_0 = train_test_split(dataset0, test_size=0.2, random_state=42)
    #     train_dirs_1, val_dirs_1 = train_test_split(dataset1, test_size=0.2, random_state=42)
    #
    #     X_train = []
    #     y_train = []
    #     X_test = []
    #     y_test = []
    #     for dir in train_dirs_0:
    #         imgs = os.listdir(data_dir+class_0+"/"+dir)
    #         for img in imgs:
    #             X_train.append(data_dir+class_0+"/"+dir+"/"+img)
    #             y_train.append(0)
    #     num_train_0 = len(X_train)
    #     print("num train_0:", num_train_0)
    #     for dir in train_dirs_1:
    #         imgs = os.listdir(data_dir+class_1+"/"+dir)
    #         for img in imgs:
    #             X_train.append(data_dir+class_1+"/"+dir+"/"+img)
    #             y_train.append(1)
    #     print("num train_1:", len(X_train)-num_train_0)
    #     print("total train:", len(X_train))
    #     for dir in val_dirs_0:
    #         imgs = os.listdir(data_dir+class_0+"/"+dir)
    #         for img in imgs:
    #             X_test.append(data_dir+class_0+"/"+dir+"/"+img)
    #             y_test.append(0)
    #     num_val_0 = len(X_test)
    #     print("num val_0:", num_val_0)
    #     for dir in val_dirs_1:
    #         imgs = os.listdir(data_dir+class_1+"/"+dir)
    #         for img in imgs:
    #             X_test.append(data_dir+class_1+"/"+dir+"/"+img)
    #             y_test.append(1)
    #     print("num val_1:", len(X_test)-num_val_0)
    #     print("total val:", len(X_test))
    #
    #     # print(X_train[0:5])
    #     # random.shuffle(X_train)
    #     # random.shuffle(X_test)
    #     # print(X_train[0:5])
    #
    #     if train:
    #         # read into ram
    #         X_train_RAM = []
    #         print("load train data into RAM:")
    #         for item in tqdm(X_train):
    #             img_arr = cv2.imread(item, cv2.IMREAD_GRAYSCALE)
    #             ## wmf_denoise ############
    #             #### img_arr_gray = cv2.cvtColor(img_arr, cv2.COLOR_BGR2GRAY)
    #             # denoised = wmf_fiter(img_arr, d=15, sigma=25)
    #             # img_arr = denoised
    #             # show = np.hstack([img_arr_gray, denoised])
    #             ###########################
    #             X_train_RAM.append(img_arr)
    #         print("load test data into RAM:")
    #         self.X, self.y = X_train_RAM, y_train  #    [N, C, H, W], [N]
    #     else:
    #         X_test_RAM = []
    #         for item in tqdm(X_test):
    #             img_arr = cv2.imread(item, cv2.IMREAD_GRAYSCALE)
    #             ## wmf_denoise ############
    #             #### img_arr_gray = cv2.cvtColor(img_arr, cv2.COLOR_BGR2GRAY)
    #             # denoised = wmf_fiter(img_arr, d=15, sigma=25)
    #             # img_arr = denoised
    #             # show = np.hstack([img_arr_gray, denoised])
    #             ###########################
    #             X_test_RAM.append(img_arr)
    #         self.X, self.y = X_test_RAM, y_test    #    [N, C, H, W], [N]
    #     self.transform = transform

    def __init__(self, data_dir="/AI/videoDetection/data/public_us_dataset/breast_lesion/rawframes/", train=True, transform=None):
        """
        POCUS Dataset
            param data_dir: str
            param transform: torch.transform
        """
        self.label_name = {"benign": 0, "malignant": 1}
        print(self.label_name)
        class_0 = os.listdir(data_dir)[0]       # benign文件夹
        class_1 = os.listdir(data_dir)[1]       # malignant文件夹
        dataset0 = os.listdir(data_dir+class_0)#[0:15]
        dataset1 = os.listdir(data_dir+class_1)#[0:15]
        train_dirs_0, val_dirs_0 = train_test_split(dataset0, test_size=0.2, random_state=42)
        train_dirs_1, val_dirs_1 = train_test_split(dataset1, test_size=0.2, random_state=42)

        X_train = []
        y_train = []
        X_test = []
        y_test = []
        for dir in train_dirs_0:
            imgs = os.listdir(data_dir+class_0+"/"+dir)
            for img in imgs:
                X_train.append(data_dir+class_0+"/"+dir+"/"+img)
                y_train.append(0)
        num_train_0 = len(X_train)
        print("num train_0:", num_train_0)
        for dir in train_dirs_1:
            imgs = os.listdir(data_dir+class_1+"/"+dir)
            for img in imgs:
                X_train.append(data_dir+class_1+"/"+dir+"/"+img)
                y_train.append(1)
        print("num train_1:", len(X_train)-num_train_0)
        print("total train:", len(X_train))
        for dir in val_dirs_0:
            imgs = os.listdir(data_dir+class_0+"/"+dir)
            for img in imgs:
                X_test.append(data_dir+class_0+"/"+dir+"/"+img)
                y_test.append(0)
        num_val_0 = len(X_test)
        print("num val_0:", num_val_0)
        for dir in val_dirs_1:
            imgs = os.listdir(data_dir+class_1+"/"+dir)
            for img in imgs:
                X_test.append(data_dir+class_1+"/"+dir+"/"+img)
                y_test.append(1)
        print("num val_1:", len(X_test)-num_val_0)
        print("total val:", len(X_test))

        # print(X_train[0:5])
        # random.shuffle(X_train)
        # random.shuffle(X_test)
        # print(X_train[0:5])

        if train:
            # read into ram
            X_train_RAM = []
            for item in tqdm(X_train):
                X_train_RAM.append(item)
            self.X, self.y = X_train_RAM, y_train  #    [N, C, H, W], [N]
        else:
            X_test_RAM = []
            for item in tqdm(X_test):
                X_test_RAM.append(item)
            self.X, self.y = X_test_RAM, y_test    #    [N, C, H, W], [N]
        self.transform = transform

    # def __getitem__(self, index):
    #     img_arr = self.X[index]  # HWC
    #     img = Image.fromarray(img_arr.astype('uint8')).convert('RGB')  # 0~255
    #     label = self.y[index]
    #
    #     if self.transform is not None:
    #         img = self.transform(img)
    #
    #     return img, label

    def __getitem__(self, index):
        item = self.X[index]  # HWC
        # print(item)
        img_arr = cv2.imread(item, cv2.IMREAD_GRAYSCALE)
        # WMF
        # denoised = wmf_fiter(img_arr, d=5, sigma=25)
        # img_arr = denoised
        #------
        # BF
        # r = 25
        # denoised = cv2.bilateralFilter(img_arr, r, r * 2, r / 2)
        # img_arr = denoised
        #------
        img = Image.fromarray(img_arr.astype('uint8')).convert('RGB')  # 0~255
        label = self.y[index]

        # cv2.imwrite("/AI/videoDetection/data/public_us_dataset/breast_lesion/imags/"+str(index)+"gt_"+str(label)+".jpg", img_arr)

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.y)

class COVIDDataset(Dataset):
    def __init__(self, data_dir, train=True, transform=None):
        """
        POCUS Dataset
            param data_dir: str
            param transform: torch.transform
        """
        self.label_name = {"covid19": 0, "pneumonia": 1, "regular": 2}
        with open(data_dir, 'rb') as f:
            X_train, y_train, X_test, y_test = pickle.load(f)
        if train:
            self.X, self.y = X_train, y_train       # [N, C, H, W], [N]
        else:
            self.X, self.y = X_test, y_test         # [N, C, H, W], [N]
        self.transform = transform
    
    def __getitem__(self, index):
        img_arr = self.X[index].transpose(1,2,0)    # CHW => HWC
        ## wmf_denoise ############
        # img_arr_gray = cv2.cvtColor(img_arr, cv2.COLOR_BGR2GRAY)
        # denoised = wmf_fiter(img_arr_gray, d=25, sigma=25)
        # img_arr = denoised
        ############################
        # r = 20
        # print("filter r:", r)
        # denoised = cv2.bilateralFilter(img_arr, r, r * 2, r / 2)
        # img_arr = denoised
        # show = np.hstack([img_arr_gray, denoised])
        ###########################
        img = Image.fromarray(img_arr.astype('uint8')).convert('RGB') # 0~255
        label = self.y[index]
        # cv2.imwrite("/AI/videoDetection/data/public_us_dataset/POCUS/fold1_test_imgs/"+str(index)+"gt_"+str(label)+".jpg", img_arr)

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.y)


class PCOSDataset(Dataset):
    def __init__(self, data_dir, train=True, transform=None):
        """
        POCUS Dataset
            param data_dir: str
            param transform: torch.transform
        """
        self.label_name = {"infected": 0, "notinfected": 1}
        X_train, y_train, X_test, y_test = self.load_data(data_dir)
        if train:
            self.X, self.y = X_train, y_train  # [N, C, H, W], [N]
        else:
            self.X, self.y = X_test, y_test  # [N, C, H, W], [N]
        self.transform = transform

    def __getitem__(self, index):
        img_arr = self.X[index] # .transpose(1, 2, 0)  # CHW => HWC
        ## wmf_denoise ############
        # img_arr_gray = cv2.cvtColor(img_arr, cv2.COLOR_BGR2GRAY)
        # denoised = wmf_fiter(img_arr_gray, d=25, sigma=25)
        # img_arr = denoised
        # r = 20
        # denoised = cv2.bilateralFilter(img_arr, r, r * 2, r / 2)
        # img_arr = denoised
        # show = np.hstack([img_arr_gray, denoised])
        ###########################
        img = Image.fromarray(img_arr.astype('uint8')).convert('RGB')  # 0~255
        label = self.y[index]
        # cv2.imwrite("/AI/videoDetection/data/public_us_dataset/POCUS/fold1_test_imgs/"+str(index)+"gt_"+str(label)+".jpg", img_arr)

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.y)

    def load_data(self, data_dir):                              # data_dir: /AI/Ye/us/pcos/data/
        train_path = data_dir + "train/"
        test_path = data_dir + "test/"
        paths = [train_path, test_path]
        X_train, y_train, X_test, y_test = [], [], [], []
        for path in paths:
            dirs = os.listdir(path)                             # infected, notinfected
            for dir in dirs:
                if dir == "infected":
                    class_id = 0
                else:
                    class_id = 1
                class_dir = path + dir                    # /AI/Ye/us/pcos/data/xxx/benign
                imgs = os.listdir(class_dir)
                for img_name in imgs:
                    img_path = class_dir + "/" + img_name
                    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                    if img is None:      # 如果不是图片文件
                        continue
                    img = cv2.resize(img, (224,224))
                    if path.endswith("train/"):
                        X_train.append(img)
                        y_train.append(class_id)
                    else:
                        X_test.append(img)
                        y_test.append(class_id)

        print("X_train:",len(X_train),"X_test:",len(X_test))
        return np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)

# Breast = PCOSDataset(data_dir="/AI/Ye/us/pcos/data/")
# for img, lable in Breast:
#     print("dgfasg")
    # print(img)