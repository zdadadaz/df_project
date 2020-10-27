# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import math
import torch
import os
import pathlib
import tqdm
import scipy.signal
import time
from dataloader.deepfakes import deepfake
import sklearn.metrics
from model.meso import MesoInception4
from model.xception import xception


def run_epoch(model, dataloader, phase, optim, device):

    criterion = torch.nn.BCELoss()  # Standard L2 loss

    runningloss = 0.0

    if phase == 'train':
        model.train(phase == 'train')
    else:
        model.eval()
        
    counter = 0
    train_corrects = 0.0

    yhat = []
    y = []
    with torch.set_grad_enabled(phase == 'train'):
        with tqdm.tqdm(total=len(dataloader)) as pbar:
            for (i, (X, Y)) in enumerate(dataloader):
                y.append(Y.numpy())

                X = X.to(device, dtype=torch.float)
                Y = Y.to(device, dtype=torch.float)

                outputs = model(X)
                
                yhat.append(outputs.view(-1).to("cpu").detach().numpy())

                loss = criterion(outputs, Y)
                preds = (outputs > 0.5).float()
                iter_corrects = torch.sum(preds == Y.data).to("cpu").detach().numpy()
                train_corrects += iter_corrects
                if phase == 'train':
                    optim.zero_grad()
                    loss.backward()
                    optim.step()

                runningloss += loss.item() * X.size(0)
                counter += X.size(0)

                epoch_loss = runningloss / counter
                epoch_accuracy = train_corrects / counter

                pbar.set_postfix_str("{:.2f}, {:.2f}, {:.2f}".format(epoch_loss, epoch_accuracy, loss.item()))
                pbar.update()

    yhat = np.concatenate(yhat)
    y = np.concatenate(y)

    return epoch_loss, epoch_accuracy, yhat, y


# -

def run(num_epochs=45,
        modelname="meso",
        pretrained=True,
        output=None,
        device=None,
        seed=0,
        size=256,
        num_workers=5,
        batch_size=20,
        lr_step_period=None,
        run_test=False):

    ### Seed RNGs ###
    np.random.seed(seed)
    torch.manual_seed(seed)

    if output is None:
        output = os.path.join("output", "{}_{}".format(modelname,  "pretrained" if pretrained else "random"))
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pathlib.Path(output).mkdir(parents=True, exist_ok=True)

    if "meso" in modelname.split('_'):
        model = MesoInception4(num_classes = 1)
    else:
        model = xception(num_classes=1000, pretrained='imagenet')
        # model = dp_lstm(lstm_size= 512, lstm_layer= 1)
    if device.type == "cuda":
        model = torch.nn.DataParallel(model)
    model.to(device)

    optim = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    # optim = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9, weight_decay=1e-4)
    if lr_step_period is None:
        lr_step_period = math.inf
    scheduler = torch.optim.lr_scheduler.StepLR(optim, lr_step_period)

    # image normalization
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    kwargs = {"mean": mean,
              "std": std,
              "size":size,
              }

# Data preparation
    train_dataset = deepfake(split="train", **kwargs, pad=12)
    val_dataset = deepfake(split="test", **kwargs)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=(device.type == "cuda"), drop_last=True)
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=(device.type == "cuda"))
    dataloaders = {'train': train_dataloader, 'val': val_dataloader}

    with open(os.path.join(output, "log.csv"), "a") as f:
        epoch_resume = 0
        bestLoss = float("inf")
        # read previous trained model
        try:
            checkpoint = torch.load(os.path.join(output, "checkpoint.pt"))
            model.load_state_dict(checkpoint['state_dict'])
            optim.load_state_dict(checkpoint['opt_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_dict'])
            epoch_resume = checkpoint["epoch"] + 1
            bestLoss = checkpoint["best_loss"]
            f.write("Resuming from epoch {}\n".format(epoch_resume))
        except FileNotFoundError:
            f.write("Starting run from scratch\n")

        # Train one epoch
        for epoch in range(epoch_resume, num_epochs):
            print("Epoch #{}".format(epoch), flush=True)
            for phase in ['train', 'val']:
                start_time = time.time()
                for i in range(torch.cuda.device_count()):
                    torch.cuda.reset_max_memory_allocated(i)
                    torch.cuda.reset_max_memory_cached(i)
                loss, accuracy, yhat, y = run_epoch(model, dataloaders[phase], phase, optim, device)
                
                f.write("{},{},{},{},{},{},{},{}\n".format(epoch,
                                                              phase,
                                                              loss,
                                                              accuracy,
                                                              sklearn.metrics.r2_score(yhat, y),
                                                              time.time() - start_time,
                                                              y.size,
                                                              batch_size))
                f.flush()
            scheduler.step()

            save = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'best_loss': bestLoss,
                'loss': loss,
                'accuracy': accuracy,
                'r2': sklearn.metrics.r2_score(yhat, y),
                'opt_dict': optim.state_dict(),
                'scheduler_dict': scheduler.state_dict(),
            }
            torch.save(save, os.path.join(output, "checkpoint.pt"))
            if loss < bestLoss:
                torch.save(save, os.path.join(output, "best.pt"))
                bestLoss = loss

        checkpoint = torch.load(os.path.join(output, "best.pt"))
        model.load_state_dict(checkpoint['state_dict'])
        f.write("Best validation loss {} from epoch {}\n".format(checkpoint["loss"], checkpoint["epoch"]))
        f.flush()

        # Testing
        if run_test:
            for split in ["test"]:
                dataloader = torch.utils.data.DataLoader(
                    deepfake(split=split, **kwargs),
                    batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=(device.type == "cuda"))
                loss, accuracy, yhat, y = run_epoch(model, dataloader, split, None, device)
                f.write("{} loss: {:.3f}, accuracy: {:.3f} \n".format(split, loss,accuracy))
                f.flush()

run(modelname="meso_inception_adam",
        pretrained=False,
        batch_size=8,
        run_test=True,
        size = 256,
           lr_step_period=15,
        num_epochs = 50)

# run(modelname="xception_adam",
#         pretrained=False,
#         batch_size=8,
#         run_test=True,
#         size = 299,
#         lr_step_period=15,
#         num_epochs = 50)


