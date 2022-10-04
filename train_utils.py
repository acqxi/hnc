import json
import os
import re
import time
from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as xfmr
from data.dataset import LAPsDatasetNode
from model.resnet import DualResNet, generate_ResNet
from sklearn.metrics import accuracy_score, auc, confusion_matrix, f1_score, precision_score, recall_score, roc_curve
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
from torchvision import datasets, transforms
from tqdm import notebook, tqdm


def run_once(model, X: List[torch.Tensor], y: List[int], loss_function, optimizer, isTrain=False):
    model.train(isTrain)  # Set model to training mode
    optimizer.zero_grad()  # zero the parameter gradients

    # forward
    with torch.set_grad_enabled(isTrain):
        outputs = model(*X)  # this get's the prediction from the network

        if str(loss_function) == "BCEWithLogitsLoss()":
            loss = loss_function(outputs, y.float().unsqueeze(1))
        elif str(loss_function) == "CrossEntropyLoss()":
            loss = loss_function(outputs, y)

        if isTrain:
            # backward + optimize only if in training phase
            loss.backward()
            optimizer.step()

    return loss.item(), outputs, y


def record_tfb(Y, y, writer: Optional[any] = None, gStep: Optional[int] = None, name="metrics"):
    # y = torch.tensor([1,0,0,1])
    # Y = torch.tensor([[0.9],[0.1],[0.4],[0.3]])
    beta = 1e-5

    fpr, tpr, thresholds = roc_curve(y, Y)
    roc_auc = auc(fpr, tpr)
    optt = thresholds[np.argmax(tpr - fpr)]  # Youden_index Only the first occurrence is returned.

    Y_class = Y >= optt
    cm = confusion_matrix(y, Y_class)
    TN, FP, FN, TP = cm.ravel()

    Population = TN + FN + TP + FP
    Accuracy = (TP + TN) / Population
    Sensitivity = Recall = (TP + beta) / (TP + FN + beta)  # A / A + B
    PPV = Precision = (TP + beta) / (TP + FP + beta)  # A / A + C
    NPV = (TN + beta) / (TN + FN + beta)
    FPR = (FP + beta) / (TN + FP + beta)
    FNR = (FN + beta) / (TP + FN + beta)
    TNR = Specificity = (TN + beta) / (TN + FP + beta)  # D / C + D

    if writer:
        writer.add_scalar(tag=name + "/auc", scalar_value=roc_auc, global_step=gStep)
        writer.add_scalar(tag=name + "/Sensitivity", scalar_value=Sensitivity, global_step=gStep)
        writer.add_scalar(tag=name + "/Specificity", scalar_value=Specificity, global_step=gStep)
        writer.add_scalar(tag=name + "/Accuracy", scalar_value=Accuracy, global_step=gStep)
        writer.add_scalar(tag=name + "/Precision", scalar_value=Precision, global_step=gStep)

    return optt, roc_auc, TN, FP, FN, TP, Sensitivity, Specificity, Accuracy, Precision


def train_loop(
    model,
    epochs,
    train_loader,
    val_loader,
    test_loader,
    loss_func,
    opti,
    pth_dir,
    device,
    earlyStop: bool = 20,
    isNotebook: bool = False,
    save_param: str = "1Epochs",
):
    writer = SummaryWriter(log_dir=pth_dir)
    os.makedirs(pth_dir, exist_ok=True)
    start_ts = time.time()
    inr, inr_type = re.findall(r"(\d+)(.+)", save_param)[0]
    min_v_loss, last_update_min_loss = 10, 0
    epoch_iter = notebook.tqdm(range(epochs)) if isNotebook else range(epochs)
    # loop for every epoch (training + evaluation)
    for epoch in epoch_iter:
        # Train
        # progress bar (works in Jupyter notebook too!)
        total_step = len(train_loader)
        progress = (
            notebook.tqdm(enumerate(train_loader), total=total_step, leave=False)
            if isNotebook
            else tqdm(enumerate(train_loader), total=total_step)
        )
        total_loss, v_t_loss = 0, 0

        for step, t_data in progress:
            t_data[:-1] = map(lambda x: x.to(device), t_data[:-1])
            current_loss, _, _ = run_once(model, t_data[:-2], t_data[-2], loss_func, opti, True)
            global_step = (epoch * total_step + step) * train_loader.batch_size

            # getting training quality data
            total_loss += current_loss
            writer.add_scalar(tag="loss/train", scalar_value=total_loss / (step + 1), global_step=global_step)

            # updating progress bar
            progress.set_description(f"Epoch{epoch+1}, t_l: {total_loss / (step + 1):.4f}")

            if (inr_type == "Epochs" and (epoch + 1) % (int(inr)) == 0 and step + 1 == total_step) or (
                inr_type == "Batchs" and (step + 1) % (int(inr)) == 0
            ):
                # Valid
                v_t_loss = 0

                for j, v_data in enumerate(val_loader):
                    v_data[:-1] = map(lambda x: x.to(device), v_data[:-1])
                    current_loss, Y, y = run_once(model, v_data[:-2], v_data[-2], loss_func, opti)

                    # getting training quality data
                    v_t_loss += current_loss

                # updating progress bar
                progress.set_description(f"Epoch{epoch+1}, t_l: {total_loss / (step + 1):.4f}, v_l:{v_t_loss:.4f}")
                # print(f"Epoch{epoch+1}, valid Loss: {v_t_loss / len(val_loader):.4f}")
                writer.add_scalar(tag="loss/valid", scalar_value=v_t_loss / (j + 1), global_step=global_step)
                record_tfb(Y.cpu(), y.cpu(), writer, global_step)

                # Test not effect training
                for j, s_data in enumerate(test_loader):
                    s_data[:-1] = map(lambda x: x.to(device), s_data[:-1])
                    _, Y, y = run_once(model, s_data[:-2], s_data[-2], loss_func, opti)

                with open(pth_dir + "test.log", "a") as f:
                    f.writelines(f"E{epoch+1}({global_step}):\n")
                    f.writelines(f"{record_tfb(Y.cpu(), y.cpu(), writer, global_step, name='test')}\n")
                    f.writelines(f"\tY:{', '.join(map(str, Y.cpu().numpy()))}\n")
                    f.writelines(f"\ty:{', '.join(map(str, y.cpu().numpy()))}\n")

        # early stop
        if v_t_loss < min_v_loss:
            last_update_min_loss = 0
            min_v_loss = v_t_loss
            torch.save(model.state_dict(), pth_dir + f"E{epoch+1}_vl{v_t_loss:.4f}.pth")
        last_update_min_loss += 1
        if last_update_min_loss > earlyStop:
            break

        # releasing unceseccary memory in GPU at epoch finish
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"Training time: {time.time()-start_ts:.2f}s")
    writer.close()

    return model