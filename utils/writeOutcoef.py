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

class Run_model():
    def __init__(self, name, dir_path, out_dir_path, size=299, mean=0.5, std=0.5):
        self.name = name
        self.dir_path = dir_path
        self.out_dir_path = out_dir_path
        self.size = size
        self.mean = mean
        self.std = std
        
    def run(self, classifier):
        tmp = []

        # tmp = ["dfdc_train_part_9","dfdc_train_part_1","dfdc_train_part_8","dfdc_train_part_23",\
        #         "dfdc_train_part_19","dfdc_train_part_16", "dfdc_train_part_29"]
        #  to 28
        for folder in os.listdir(self.dir_path):
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
                    # after run 29, remove 
                    # if folder == "dfdc_train_part_13" and os.path.isfile(os.path.join(outpath,file[:-4]+".txt")):
                    #     continue
                    img_path = os.path.join(self.dir_path, folder,df_real, file)
                    
                    img = loadimage(img_path)
                    resample = 0
                    img = img.resize((299,299), resample)
                    img = np.expand_dims(img, axis=0)
                    img = preprocess_input(img, self.size, self.mean, self.std)
                    coef_img = classifier.predict(img)
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
    run = Run_model("xception", dir_path, out_dir_path, 299, 0.5, 0.5)
    classifier = xception_noTop(pretrained = "input the best checkpoint path")
    run.run(classifier)
                  
# make directories
# for i in range(50):
#     folder = "dfdc_train_part_"+str(i)
#     path = os.path.join(out_dir_path,folder)
#     cmd = 'mkdir ' + path
#     os.system(cmd)
    
#     folder = "dfdc_train_part_"+str(i)
#     path = os.path.join(out_dir_path,folder,"REAL")
#     cmd = 'mkdir ' + path
#     os.system(cmd)
    
#     folder = "dfdc_train_part_"+str(i)
#     path = os.path.join(out_dir_path,folder,"FAKE")
#     cmd = 'mkdir ' + path
#     os.system(cmd)
           
            