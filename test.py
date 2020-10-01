from utils.utils import loadimage
from dataloader.deepfakes import deepfake
from dataloader.deepfakes_ts import deepfake_ts
import torch

train_dataset = deepfake_ts()
train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=2, num_workers=1, shuffle=True, pin_memory=(False), drop_last=True)
aa = iter(train_dataloader)
x,y=aa.next()
print(x,y)
print(x.shape, y.shape)