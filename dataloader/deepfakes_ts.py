# Reference
# https://github.com/echonet/dynamic

import pathlib
import torch
import torch.utils.data
import os
import collections
from utils.utils import loadimage
import sys
import numpy as np
import cv2


class deepfake_ts(torch.utils.data.Dataset):
    def __init__(self, root=None,
                 split = "train", 
                 mean=0.,std=1.,
                 size=256,
                 pad = None):

        if root is None:
            root = '/home/zdadadaz/Desktop/course/dfd/dataset/fb_db'
        self.folder_path = pathlib.Path(root)
        self.split = split
        self.mean = mean
        self.std = std
        self.pad = pad
        self.size =size

        self.fnames, self.folder, self.label = [], [], []
        with open("./metadata/dataset.csv") as f:
                self.header = f.readline().strip().split(",")
                filenameIndex = self.header.index("filename")
                splitIndex = self.header.index("split")
                labelIndex = self.header.index("label")
                folderIndex = self.header.index("folder")

                for (i, line) in enumerate(f):
                    lineSplit = line.strip().split(',')
                    fileName = lineSplit[filenameIndex]
                    folder = lineSplit[folderIndex]
                    label = lineSplit[labelIndex]
                    fileMode = lineSplit[splitIndex].lower()
                    img_path = os.path.join(self.folder_path, folder, label, fileName+'_1.jpg')
                    if (split == fileMode) and os.path.exists(img_path):
                        self.fnames.append(fileName)
                        self.label.append(label)
                        self.folder.append(folder)
    
    def __getitem__(self, index):

        video = []
        nframe = 5
        for i in range(nframe):
            try:
                img_path = os.path.join(self.folder_path, self.folder[index], self.label[index], self.fnames[index]+'_{}.jpg'.format(i))
            except:
                print("not enough image")
            img = loadimage(img_path)
            img = cv2.resize(img,(self.size,self.size))/255.
            img = (img - self.mean) / self.std
            img = img.transpose((2, 0, 1))
            video.append(img)

        if len(video) < nframe:
            tmp = len(video)
            for i in range(nframe-tmp):
                video.append(img)

        video = np.stack(video)
        # video = video.transpose((1, 0, 2, 3))
        outcome = np.array([0])
        if self.label[index] == 'FAKE':
            outcome[0] = 1.

        if self.pad is not None:
            l, c, h, w = video.shape
            temp = np.zeros((l,c, h + 2 * self.pad, w + 2 * self.pad), dtype=video.dtype)
            temp[:, :, self.pad:-self.pad, self.pad:-self.pad] = video
            i, j = np.random.randint(0, 2 * self.pad, 2)
            video = temp[:, :, i:(i + h), j:(j + w)]

        return video, outcome
            
    def __len__(self):
        return len(self.fnames)


