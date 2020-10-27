# from utils.utils import loadimage
# from dataloader.deepfakes import deepfake
from dataloader.deepfakes_ts import deepfake_ts
import torch
import os
from utils.writeOutcoef import Run_model
from model.xception import xception_noTop
import numpy as np

# train_dataset = deepfake_ts()
# train_dataloader = torch.utils.data.DataLoader(
#         train_dataset, batch_size=2, num_workers=1, shuffle=True, pin_memory=(False), drop_last=True)
# aa = iter(train_dataloader)
# x,y=aa.next()
# print(x,y)
# print(x.shape, y.shape)

dir_path = "./../dataset/fb_db"
out_dir_path = "./../dataset/fb_db_xception_adam"
run = Run_model("xception", dir_path, out_dir_path, 299, 0.5, 0.5)
# model = xception_noTop(pretrained = "input the best checkpoint path")
model = xception_noTop(num_classes=1000, pretrained='./output/xception_adam_random/')
for param in model.parameters():
    param.requires_grad = False
run.run(model)

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
        
    