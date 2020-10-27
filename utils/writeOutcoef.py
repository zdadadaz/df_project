#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 10:29:49 2020

@author: zdadadaz
"""

import numpy as np
import os
import imageio
import json
import pandas as pd
import cv2
from utils.utils import loadimage, preprocess_input
from model.xception import xception_noTop
import torch

class Run_model():
    def __init__(self, name, dir_path, out_dir_path, size=299, mean=0.5, std=0.5):
        self.name = name
        self.dir_path = dir_path
        self.out_dir_path = out_dir_path
        self.size = size
        self.mean = mean
        self.std = std
        
    def run(self, model):
        tmp = []
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()

        # tmp = ["dfdc_train_part_9","dfdc_train_part_1","dfdc_train_part_8","dfdc_train_part_23",\
        #         "dfdc_train_part_19","dfdc_train_part_16", "dfdc_train_part_29"]
        #  to 28
        for idx in range(50):#os.listdir(self.dir_path):
            folder = "dfdc_train_part_{}".format(idx)
            if folder[0] =="." or folder in tmp:
                continue
            print(folder)
            for df_real in os.listdir(os.path.join(self.dir_path, folder)):
                if df_real[0] == ".":
                    continue
                for file in os.listdir(os.path.join(self.dir_path, folder,df_real)):
                    if  (file[-4:] != '.jpg'):
                        continue
                    outpath = os.path.join(self.out_dir_path, folder, df_real)              
                    img_path = os.path.join(self.dir_path, folder,df_real, file)
                    
                    img = loadimage(img_path)
                    img = preprocess_input(img, self.size, self.mean, self.std)
                    img = np.expand_dims(img, axis=0)
                    img = torch.from_numpy(img).type(torch.FloatTensor).to(device)
                    coef_img = model(img)
                    coef_img = coef_img.to("cpu").detach().numpy()
                    self.write_coef(coef_img, outpath, file)

    def write_coef(self, coef_img, path, file):
        with open(os.path.join(path,file[:-4]+".npy"), 'wb') as f:
            np.save(f, coef_img)

    def read_coef(self, path):
        with open(path, 'rb') as f:
            a = np.load(f)
        return a

    # def write_coef(self, coef_img, path, file):
        # np.savetxt(os.path.join(path,file[:-4]+".txt"), coef_img, delimiter=',')

if __name__ == "__main__":
    dir_path = "./../dataset/fb_db"
    out_dir_path = "./../dataset/fb_db_xception"
    # run = Run_model("xception", dir_path, out_dir_path, 299, 0.5, 0.5)
    # # model = xception_noTop(pretrained = "input the best checkpoint path")
    # model = xception_noTop(num_classes=1000, pretrained='./output/xception_random_adam/')
    # for param in model.parameters():
    #     param.requires_grad = False
    # run.run(model)
                    
    # make directories
    for i in range(50):
        folder = "dfdc_train_part_"+str(i)
        path = os.path.join(out_dir_path,folder)
        cmd = 'mkdir ' + path
        os.system(cmd)
        
        folder = "dfdc_train_part_"+str(i)
        path = os.path.join(out_dir_path,folder,"REAL")
        cmd = 'mkdir ' + path
        os.system(cmd)
        
        folder = "dfdc_train_part_"+str(i)
        path = os.path.join(out_dir_path,folder,"FAKE")
        cmd = 'mkdir ' + path
        os.system(cmd)
           
            