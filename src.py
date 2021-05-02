#!/usr/bin/env python
# coding: utf-8

# In[1]:

import joblib
import os
import cv2
import numpy as np
import pandas as pd
import albumentations
from albumentations.core.transforms_interface import DualTransform
from albumentations.augmentations import functional as albuF
from torchvision.transforms import transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch
from torch.utils.data import TensorDataset, DataLoader, Dataset
import random
import time
from sklearn.metrics import f1_score, recall_score
import shutil
import os
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import math
from torch.utils.data import Sampler
from model_eff import *
from apex import amp, optimizers

import warnings
warnings.filterwarnings('ignore')

import cv2
import sys

# http://stackoverflow.com/questions/34950201/pycharm-print-end-r-statement-not-working
class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout  #stdout
        self.file = None

    def open(self, file, mode=None):
        if mode is None: mode ='w'
        self.file = open(file, mode)

    def write(self, message, is_terminal=1, is_file=1):
        if '\r' in message: is_file = 0

        if is_terminal == 1:
            self.terminal.write(message)
            self.terminal.flush()
            #time.sleep(1)

        if is_file == 1:
            self.file.write(message)
            self.file.flush()

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# In[7]:




# seed value fix
# seed 값을 고정해야 hyper parameter 바꿀 때마다 결과를 비교할 수 있습니다.
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def save_checkpoint(state, is_best, filename='checkpoint.pth'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, './checkpoints/model_best.pth')

def get_preds(logits, pred_array):
    output = F.softmax(logits, 1)
    pred_array.append(output.detach().cpu().numpy())
    _, output = torch.max(output, 1)
    output = output.detach().cpu().numpy()
    return output, pred_array

def adjust_learning_rate(optimizer, decay, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 1 epochs"""
    lr = INITIAL_LR * (decay ** epoch)
    for param_group in optimizer.state_dict()['param_groups']:
        param_group['lr'] = lr

VERSION = 'V191'

HEIGHT = 137
WIDTH = 236
IMG_SIZE = 196

SEED = 42
seed_everything(SEED)
BATCH_SIZE = 64
MODEL_NAME = 'eff4'
EPOCHS = 500
INITIAL_LR = 1e-3
DECAY = 0.95
NFOLDS = 5
GPU_NUMBER = "0"
BETA = 1.0
CUTMIX_PROB = 0.5
BALANCE_LENGTH = 1
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]=GPU_NUMBER

DEBUG = False

DESCRIPION = "final"

log = Logger()
log.open('./logs/log.train_{}.txt'.format(VERSION), mode='a')


log.write('VERSION: {}\n'.format(VERSION))
log.write('SEED: {}\n'.format(SEED))
log.write('BATCH_SIZE: {}\n'.format(BATCH_SIZE))
log.write('MODEL_NAME: {}\n'.format(MODEL_NAME))
log.write('IMG_SIZE: {}\n'.format(IMG_SIZE))
log.write('EPOCHS: {}\n'.format(EPOCHS))
log.write('INITIAL_LR: {}\n'.format(INITIAL_LR))
log.write('DECAY: {}\n'.format(DECAY))
log.write('GPU_NUMBER: {}\n'.format(GPU_NUMBER))
log.write('BETA: {}\n'.format(BETA))
log.write('CUTMIX_PROB: {}\n'.format(CUTMIX_PROB))

log.write('DESCRIPTION: \n {} \n'.format(DESCRIPION))
log.write('\n\n')

use_cuda = torch.cuda.is_available()

data_dir = '../input/'

files_train = [f'train_image_data_{fid}.parquet' for fid in range(4)]
df_train = pd.read_csv(os.path.join(data_dir, f'train.csv'))


def read_data(files):
    tmp = []
    for f in files:
        F = os.path.join(data_dir, f)
        data = pd.read_parquet(F)
        tmp.append(data)
    tmp = pd.concat(tmp)

    data = tmp.iloc[:, 1:].values
    return data


class BengaliDataset(Dataset):
    def __init__(self, csv, mode, image_size, transform=None):
        self.csv = csv.reset_index()
        self.mode = mode
        self.image_size = image_size
        self.transform = transform
        self.gridmask = Compose([
                            albumentations.OneOf([
                            albumentations.Cutout(num_holes=4, max_h_size=10, max_w_size=10, fill_value=255, p=1),
                            albumentations.Cutout(num_holes=8, max_h_size=10, max_w_size=10, fill_value=255, p=1),
                            albumentations.Cutout(num_holes=16, max_h_size=10, max_w_size=10, fill_value=255, p=1),
                            albumentations.Cutout(num_holes=32, max_h_size=10, max_w_size=10, fill_value=255, p=1),

                            albumentations.Cutout(num_holes=4, max_h_size=10, max_w_size=10, fill_value=200, p=1),
                            albumentations.Cutout(num_holes=8, max_h_size=10, max_w_size=10, fill_value=200, p=1),
                            albumentations.Cutout(num_holes=16, max_h_size=10, max_w_size=10, fill_value=200, p=1),
                            albumentations.Cutout(num_holes=32, max_h_size=10, max_w_size=10, fill_value=200, p=1),

                            albumentations.Cutout(num_holes=4, max_h_size=10, max_w_size=10, fill_value=150, p=1),
                            albumentations.Cutout(num_holes=8, max_h_size=10, max_w_size=10, fill_value=150, p=1),
                            albumentations.Cutout(num_holes=16, max_h_size=10, max_w_size=10, fill_value=150, p=1),
                            albumentations.Cutout(num_holes=32, max_h_size=10, max_w_size=10, fill_value=150, p=1),

                            albumentations.Cutout(num_holes=4, max_h_size=10, max_w_size=10, fill_value=100, p=1),
                            albumentations.Cutout(num_holes=8, max_h_size=10, max_w_size=10, fill_value=100, p=1),
                            albumentations.Cutout(num_holes=16, max_h_size=10, max_w_size=10, fill_value=100, p=1),
                            albumentations.Cutout(num_holes=32, max_h_size=10, max_w_size=10, fill_value=100, p=1),

                            albumentations.Cutout(num_holes=4, max_h_size=10, max_w_size=10, fill_value=50, p=1),
                            albumentations.Cutout(num_holes=8, max_h_size=10, max_w_size=10, fill_value=50, p=1),
                            albumentations.Cutout(num_holes=16, max_h_size=10, max_w_size=10, fill_value=50, p=1),
                            albumentations.Cutout(num_holes=32, max_h_size=10, max_w_size=10, fill_value=50, p=1),
        ], p=0.3)
    ])

    def __len__(self):
        return len(self.csv)

    def __getitem__(self, index):
        this_img_id = self.csv.iloc[index % len(self.csv)].image_id
        image = joblib.load(f"../input/image_pickles_2/{this_img_id}.pkl")
        image = image.reshape(HEIGHT, WIDTH).astype(np.uint8)
        #image = cv2.resize(image, (168, 224))
        image = 255 - image
        image = image[:, :, np.newaxis]
        image = np.repeat(image, 3, 2)  # 1ch to 3ch

        if self.transform is not None:
            if self.mode == 'train':
                image = self.gridmask(image=image)['image']
                image = self.transform(image)
            elif self.mode == 'valid':
                image = self.transform(image)

        label_1 = self.csv.iloc[index].grapheme_root
        label_2 = self.csv.iloc[index].vowel_diacritic
        label_3 = self.csv.iloc[index].consonant_diacritic
        return image, label_1, label_2, label_3

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR



def get_constant_schedule(optimizer, last_epoch=-1):
    """ Create a schedule with a constant learning rate.
    """
    return LambdaLR(optimizer, lambda _: 1, last_epoch=last_epoch)


def get_constant_schedule_with_warmup(optimizer, num_warmup_steps, last_epoch=-1):
    """ Create a schedule with a constant learning rate preceded by a warmup
    period during which the learning rate increases linearly between 0 and 1.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1.0, num_warmup_steps))
        return 1.0

    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """ Create a schedule with a learning rate that decreases linearly after
    linearly increasing during a warmup period.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5, last_epoch=-1):
    """ Create a schedule with a learning rate that decreases following the
    values of the cosine function between 0 and `pi * cycles` after a warmup
    period during which it increases linearly between 0 and 1.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_cosine_with_hard_restarts_schedule_with_warmup(
    optimizer, num_warmup_steps, num_training_steps, num_cycles=1.0, last_epoch=-1
):
    """ Create a schedule with a learning rate that decreases following the
    values of the cosine function with several hard restarts, after a warmup
    period during which it increases linearly between 0 and 1.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        if progress >= 1.0:
            return 0.0
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * ((float(num_cycles) * progress) % 1.0))))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


class AdamW(Optimizer):
    """ Implements Adam algorithm with weight decay fix.
    Parameters:
        lr (float): learning rate. Default 1e-3.
        betas (tuple of 2 floats): Adams beta parameters (b1, b2). Default: (0.9, 0.999)
        eps (float): Adams epsilon. Default: 1e-6
        weight_decay (float): Weight decay. Default: 0.0
        correct_bias (bool): can be set to False to avoid correcting bias in Adam (e.g. like in Bert TF repository). Default True.
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-6, weight_decay=0.0, correct_bias=True):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1]))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias)
        super().__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                # Decay the first and second moment running average coefficient
                # In-place operations to update the averages at the same time
                exp_avg.mul_(beta1).add_(1.0 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1.0 - beta2, grad, grad)
                denom = exp_avg_sq.sqrt().add_(group["eps"])

                step_size = group["lr"]
                if group["correct_bias"]:  # No bias correction for Bert
                    bias_correction1 = 1.0 - beta1 ** state["step"]
                    bias_correction2 = 1.0 - beta2 ** state["step"]
                    step_size = step_size * math.sqrt(bias_correction2) / bias_correction1

                p.data.addcdiv_(-step_size, exp_avg, denom)

                # Just adding the square of the weights to the loss function is *not*
                # the correct way of using L2 regularization/weight decay with Adam,
                # since that will interact with the m and v parameters in strange ways.
                #
                # Instead we want to decay the weights in a manner that doesn't interact
                # with the m/v parameters. This is equivalent to adding the square
                # of the weights to the loss with plain (non-momentum) SGD.
                # Add weight decay at the end (fixed version)
                if group["weight_decay"] > 0.0:
                    p.data.add_(-group["lr"] * group["weight_decay"], p.data)

        return loss

from albumentations import Compose, RandomCrop, Normalize, HorizontalFlip, Resize, Rotate, ElasticTransform
from albumentations.pytorch import ToTensor

# def plot_imgs(dataset_show):
#     from pylab import rcParams
#     rcParams['figure.figsize'] = 20,10
#     for i in range(2):
#         f, axarr = plt.subplots(1,5)
#         for p in range(5):
#             idx = np.random.randint(0, len(dataset_show))
#             img, label = dataset_show[idx]
#             axarr[p].imshow(img.transpose(0, 1).transpose(1,2).squeeze())
#             axarr[p].set_title(idx)

data_transforms = {
    'train':transforms.Compose([
        transforms.ToPILImage(),
        #transforms.RandomAffine(20, translate=(0.15, 0.15), scale=(0.85, 1.15), shear=0.15, resample=False, fillcolor=0),
        transforms.RandomAffine(10, translate=(0.10, 0.10), scale=(0.90, 1.10), shear=0.10, resample=False, fillcolor=0),
        transforms.ToTensor(),
]),
    'valid':transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
    #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
}




def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def train_one_epoch(train_loader, model, criterion, optimizer, epoch):
    data_time = AverageMeter()
    batch_time = AverageMeter()
    recall = AverageMeter()
    recall_gra = AverageMeter()
    recall_vow = AverageMeter()
    recall_con = AverageMeter()

    losses = AverageMeter()
    
    model.train()
    end = time.time()
    
    target_array_gra = []
    target_array_vow = []
    target_array_con = []

    pred_array_gra = []
    pred_array_vow = []
    pred_array_con = []

    pred_label_array_gra = []
    pred_label_array_vow = []
    pred_label_array_con = []
    
    for idx, (inputs,labels_gra,labels_vow, labels_con) in (enumerate(train_loader)):

        data_time.update(time.time() - end)

        inputs = inputs.cuda()
        if epoch == 0 and idx == 0:
            print(inputs.shape, inputs.max(), inputs.min())
        
        labels_gra = labels_gra.cuda()
        labels_vow = labels_vow.cuda()
        labels_con = labels_con.cuda()
        
        
        r = np.random.rand(1)
        
        if BETA > 0 and r < 0.5:
            # generate mixed sample
            lam = np.random.beta(BETA, BETA)
            rand_index = torch.randperm(inputs.size()[0]).cuda()
            
            labels_a_gra = labels_gra
            labels_b_gra = labels_gra[rand_index]
            
            labels_a_vow = labels_vow
            labels_b_vow = labels_vow[rand_index]     
            
            labels_a_con = labels_con
            labels_b_con = labels_con[rand_index]  
            
            
            bbx1, bby1, bbx2, bby2 = rand_bbox(inputs.size(), lam)
            inputs[:, :, bbx1:bbx2, bby1:bby2] = inputs[rand_index, :, bbx1:bbx2, bby1:bby2]
            # adjust lambda to exactly match pixel ratio
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (inputs.size()[-1] * inputs.size()[-2]))
            
            logits_gra, logits_vow, logits_con  = model(inputs)
            
            loss1 = criterion(logits_gra, labels_a_gra) * lam + criterion(logits_gra, labels_b_gra) * (1. - lam)
            loss2 = criterion(logits_vow, labels_a_vow) * lam + criterion(logits_vow, labels_b_vow) * (1. - lam)
            loss3 = criterion(logits_con, labels_a_con) * lam + criterion(logits_con, labels_b_con) * (1. - lam)

        else:
            logits_gra, logits_vow, logits_con  = model(inputs)

            loss1 = criterion(logits_gra,labels_gra)
            loss2 = criterion(logits_vow,labels_vow)
            loss3 = criterion(logits_con,labels_con)
            
        loss = 0.5 * loss1 + 0.25 * loss2 + 0.25 * loss3
        

        output_gra, pred_array_gra = get_preds(logits_gra, pred_array_gra)
        output_vow, pred_array_vow = get_preds(logits_vow, pred_array_vow)
        output_con, pred_array_con = get_preds(logits_con, pred_array_con)


        scores = []
        score_gra = recall_score(labels_gra.detach().cpu().numpy(), output_gra, average='macro')
        score_vow = recall_score(labels_vow.detach().cpu().numpy(), output_vow, average='macro')
        score_con = recall_score(labels_con.detach().cpu().numpy(), output_con, average='macro')

        scores.append(score_gra)
        scores.append(score_vow)
        scores.append(score_con)
        
        
        final_score = np.average(scores, weights=[2, 1, 1])

        losses.update(loss.item())
        recall_gra.update(score_gra)
        recall_vow.update(score_vow)
        recall_con.update(score_con)

        recall.update(final_score)

        optimizer.zero_grad()
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        optimizer.step()
        

        batch_time.update(time.time() - end)

        end = time.time()

        if idx % 20 == 0:
            log.write('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.6f} ({batch_time.avg:.6f})\t'
                  'Data {data_time.val:.6f} ({data_time.avg:.6f})\t'
                  'Loss {loss.val:.6f} ({loss.avg:.6f})\t'
                  'recall@1 {recall.val:.6f} ({recall.avg:.6f})\t'
                  'recall_gra@1 {recall_gra.val:.6f} ({recall_gra.avg:.6f})\t'
                  'recall_con@1 {recall_con.val:.6f} ({recall_con.avg:.6f})\t'
                  'recall_vow@1 {recall_vow.val:.6f} ({recall_vow.avg:.6f})\n'.format(
                   epoch, idx, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, recall=recall, recall_gra=recall_gra,
                   recall_con=recall_con, recall_vow=recall_vow))
            
        target_array_gra.append(labels_gra.detach().cpu().numpy())
        target_array_vow.append(labels_vow.detach().cpu().numpy())
        target_array_con.append(labels_con.detach().cpu().numpy())

        pred_label_array_gra.append(output_gra)
        pred_label_array_vow.append(output_vow)
        pred_label_array_con.append(output_con)

    recall_final_gra = recall_score(np.concatenate(target_array_gra), np.concatenate(pred_label_array_gra), average='macro')
    recall_final_vow = recall_score(np.concatenate(target_array_vow), np.concatenate(pred_label_array_vow), average='macro')
    recall_final_con = recall_score(np.concatenate(target_array_con), np.concatenate(pred_label_array_con), average='macro')
    
    final_score = np.average([recall_final_gra, recall_final_vow, recall_final_con], weights=[2, 1, 1])
    
    return (np.concatenate(pred_array_gra), np.concatenate(pred_array_vow), np.concatenate(pred_array_con)),final_score, losses.avg

def valid_one_epoch(valid_loader, model, criterion):
    batch_time = AverageMeter()
    recall = AverageMeter()
    recall_gra = AverageMeter()
    recall_con = AverageMeter()
    recall_vow = AverageMeter()
    losses = AverageMeter()
    
    model.eval()
    
    end = time.time()
    
    target_array_gra = []
    target_array_con = []
    target_array_vow = []

    pred_array_gra = []
    pred_array_con = []
    pred_array_vow = []

    pred_label_array_gra = []
    pred_label_array_con = []
    pred_label_array_vow = []
    
    
    for idx, (inputs,labels_gra,labels_vow, labels_con) in (enumerate(valid_loader)):

        inputs = inputs.cuda()
        labels_gra = labels_gra.cuda()
        labels_vow = labels_vow.cuda()
        labels_con = labels_con.cuda()

        logits_gra, logits_vow, logits_con  = model(inputs)

        loss1 = criterion(logits_gra,labels_gra)
        loss2 = criterion(logits_vow,labels_vow)
        loss3 = criterion(logits_con,labels_con)
        
        loss = 0.5 * loss1 + 0.25 * loss2 + 0.25 * loss3

        output_gra, pred_array_gra = get_preds(logits_gra, pred_array_gra)
        output_vow, pred_array_vow = get_preds(logits_vow, pred_array_vow)
        output_con, pred_array_con = get_preds(logits_con, pred_array_con)


        scores = []
        score_gra = recall_score(labels_gra.detach().cpu().numpy(), output_gra, average='macro')
        score_vow = recall_score(labels_vow.detach().cpu().numpy(), output_vow, average='macro')
        score_con = recall_score(labels_con.detach().cpu().numpy(), output_con, average='macro')

        scores.append(score_gra)
        scores.append(score_vow)
        scores.append(score_con)

        final_score = np.average(scores, weights=[2, 1, 1])

        losses.update(loss.item())
        recall_gra.update(score_gra)
        recall_vow.update(score_vow)
        recall_con.update(score_con)
        recall.update(final_score)

        batch_time.update(time.time() - end)

        end = time.time()

        if idx % 400 == 0:
            log.write('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.6f} ({batch_time.avg:.6f})\t'
                  'Loss {loss.val:.6f} ({loss.avg:.6f})\t'
                  'recall@1 {recall.val:.6f} ({recall.avg:.6f})\t'
                  'recall_gra@1 {recall_gra.val:.6f} ({recall_gra.avg:.6f})\t'
                  'recall_con@1 {recall_con.val:.6f} ({recall_con.avg:.6f})\t'
                  'recall_vow@1 {recall_vow.val:.6f} ({recall_vow.avg:.6f})\n'.format(
                   idx, len(valid_loader), batch_time=batch_time, loss=losses,
                   recall=recall, recall_gra=recall_gra, recall_con=recall_con, recall_vow=recall_vow))
    
        target_array_gra.append(labels_gra.detach().cpu().numpy())
        target_array_vow.append(labels_vow.detach().cpu().numpy())
        target_array_con.append(labels_con.detach().cpu().numpy())

        pred_label_array_gra.append(output_gra)
        pred_label_array_vow.append(output_vow)
        pred_label_array_con.append(output_con)
        

    recall_final_gra = recall_score(np.concatenate(target_array_gra), np.concatenate(pred_label_array_gra), average='macro')
    recall_final_vow = recall_score(np.concatenate(target_array_vow), np.concatenate(pred_label_array_vow), average='macro')
    recall_final_con = recall_score(np.concatenate(target_array_con), np.concatenate(pred_label_array_con), average='macro')
    
    log.write('\n') 
    log.write('recall_final_gra@ {:.6f}\n'.format(recall_final_gra))
    log.write('recall_final_vow@ {:.6f}\n'.format(recall_final_vow))
    log.write('recall_final_con@ {:.6f}\n'.format(recall_final_con))
    
    final_score = np.average([recall_final_gra, recall_final_vow, recall_final_con], weights=[2, 1, 1])
    log.write('recall_avg score of batches@ {:.6f}\n'.format(recall.avg))
    log.write('recall_final_all@ {:.6f}\n'.format(final_score))
    
    return (np.concatenate(pred_array_gra), np.concatenate(pred_array_vow), np.concatenate(pred_array_con)), final_score, losses.avg


class BalanceSampler(Sampler):
    def __init__(self, dataset, length, unique_class):
        self.length = length
        df = dataset.reset_index()

        group = []
        grapheme_gb = df.groupby(['grapheme_root'])
        for k in unique_class:
            g = grapheme_gb.get_group(k).index
            group.append(list(g))
            assert(len(g)>0)

        self.group=group

    def __iter__(self):
        index = []
        n = 0

        is_loop = True
        while is_loop:
            num_class = 168 #1295
            c = np.arange(num_class)
            np.random.shuffle(c)
            for t in c:
                i = np.random.choice(self.group[t])
                index.append(i)
                n+=1
                if n == self.length:
                    is_loop = False
                    break
        return iter(index)
    
    def __len__(self):
        return self.length



df_train['id'] = df_train['image_id'].apply(lambda x: int(x.split('_')[1]))
X = df_train[['id', 'grapheme_root', 'vowel_diacritic', 'consonant_diacritic']].values[:,0]
y = df_train[['id', 'grapheme_root', 'vowel_diacritic', 'consonant_diacritic']].values[:,1:]
df_train['fold'] = np.nan



#plits = np.load('../input/splits.npy')

# mskf = MultilabelStratifiedKFold(n_splits=11, random_state=SEED)
# for i, (_, test_index) in enumerate(mskf.split(X, y)):
#     df_train.iloc[test_index, -1] = i
# df_train['fold'] = df_train['fold'].astype(int)

df_train['fold'] = pd.read_csv('./train_folds6.csv')['fold']

cnt_srs = df_train['grapheme_root'].value_counts()

sparse_grapheme = cnt_srs[cnt_srs < 500].index.tolist()

df_train = df_train.loc[df_train['grapheme_root'].isin(sparse_grapheme)].reset_index(drop=True)

trn_fold = [i for i in range(6) if i not in [0]]

trn_idx = df_train.loc[df_train['fold'].isin(trn_fold)].index
dev_idx = df_train.loc[df_train['fold'] == 0].index

trn_dataset = BengaliDataset(csv = df_train.loc[trn_idx], 
                             mode = 'train', 
                             image_size = IMG_SIZE,
                             transform = data_transforms['train']
                            )


dev_dataset = BengaliDataset(csv = df_train.loc[dev_idx], 
                             mode = 'train', 
                             image_size = IMG_SIZE,
                             transform = data_transforms['valid']                            
                            )



#trn_sampler = torch.utils.data.sampler.WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight))
grapheme_uniques = df_train['grapheme_root'].unique()
grapheme_uniques = sorted(grapheme_uniques)
trn_sampler = BalanceSampler(df_train.loc[trn_idx, 'grapheme_root'], int(BALANCE_LENGTH * len(trn_dataset)), grapheme_uniques)

trn_loader = torch.utils.data.DataLoader(trn_dataset, 
                                         num_workers=2,
                                         batch_size=BATCH_SIZE,
                                         shuffle=True)

dev_loader = torch.utils.data.DataLoader(dev_dataset,
                                         num_workers=2,
                                         batch_size=BATCH_SIZE,
                                         shuffle=False)

model = EfficientNet.from_name('efficientnet-b4').cuda()
# model.avg_pool = GeM()

pretrained_dict = torch.load('./efficientnet-b4-6ed6700e.pth')
model_dict = model.state_dict()
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
model_dict.update(pretrained_dict) 
model.load_state_dict(model_dict)




# optimizer = torch.optim.Adam(model.parameters(),lr=INITIAL_LR)
# optimizer = AdamW(model.parameters(), lr=INITIAL_LR)


gradient_accumulation_steps = 1
t_total = len(trn_loader) // gradient_accumulation_steps * EPOCHS

param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

optimizer = AdamW(optimizer_grouped_parameters, lr=INITIAL_LR, eps=1e-6)
model, optimizer = amp.initialize(model, optimizer,
                                  opt_level='O1',
                                  verbosity=0
                                  )

    
scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
    optimizer, num_warmup_steps=0, num_training_steps=t_total
)




def onehot_encoding(label, n_classes):
    return torch.zeros(label.size(0), n_classes).to(label.device).scatter_(
        1, label.view(-1, 1), 1)
def cross_entropy_loss(input, target, reduction):
    logp = F.log_softmax(input, dim=1)
    loss = torch.sum(-logp * target, dim=1)
    if reduction == 'none':
        return loss
    elif reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        raise ValueError(
            '`reduction` must be one of \'none\', \'mean\', or \'sum\'.')
        
def label_smoothing_criterion(epsilon=0.1, reduction='mean'):
    def _label_smoothing_criterion(preds, targets):
        n_classes = preds.size(1)
        device = preds.device

        onehot = onehot_encoding(targets, n_classes).float().cuda()
        targets = onehot * (1 - epsilon) + torch.ones_like(onehot).to(
            device) * epsilon / n_classes
        loss = cross_entropy_loss(preds, targets, reduction)
        if reduction == 'none':
            return loss
        elif reduction == 'mean':
            return loss.mean()
        elif reduction == 'sum':
            return loss.sum()
        else:
            raise ValueError(
                '`reduction` must be one of \'none\', \'mean\', or \'sum\'.')
    return _label_smoothing_criterion



criterion = label_smoothing_criterion() #nn.CrossEntropyLoss().cuda()

df_results = pd.DataFrame()

best_score = 0
model = nn.DataParallel(model)

if False:
    model.load_state_dict(torch.load('./checkpoints/model_best.pth')['model_state_dict'])
    optimizer.load_state_dict(torch.load('./checkpoints/model_best.pth')['optimizer_state_dict'])
    best_score = torch.load('./checkpoints/model_best.pth')['best_score']
    restart_epoch = torch.load('./checkpoints/model_best.pth')['epoch']
    df_results = pd.read_csv('./logs/df_results.csv')

for epoch in range(EPOCHS):
    #epoch += restart_epoch
                
    log.write('\n\nEPOCH: {} =============================================\n'.format(epoch))
    #adjust_learning_rate(optimizer, DECAY, epoch)

    train_preds, train_score, train_loss = train_one_epoch(trn_loader, model, criterion, optimizer, epoch)
    dev_preds, dev_score, dev_loss = valid_one_epoch(dev_loader, model, criterion)

    is_best = dev_score > best_score
    best_score = max(dev_score, best_score)
    
    df_results.loc[epoch, 'train_score'] = train_score
    df_results.loc[epoch, 'train_loss'] = train_loss
    df_results.loc[epoch, 'dev_score'] = dev_score
    df_results.loc[epoch, 'dev_loss'] = dev_loss
    df_results.to_csv('./logs/df_results.csv', index=False)
    
    if epoch > 15:
        if is_best:
            save_checkpoint(state={
                'epoch': epoch + 1,
                'arch': MODEL_NAME,
                'model_state_dict': model.state_dict(),
                'best_score': best_score,
                'optimizer_state_dict': optimizer.state_dict(),
            }, 
            is_best = is_best, 
            filename= './checkpoints/{}_model_{}_epoch_{}_score_{:.6f}.pth'
                .format(VERSION,MODEL_NAME, epoch+1, best_score))
    else:
        continue


df_results.to_csv('./logs/df_results.csv', index=False)
