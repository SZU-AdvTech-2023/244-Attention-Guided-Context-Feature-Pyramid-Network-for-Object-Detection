# -*- coding: utf-8 -*-
"""
用于训练神经网络的简单脚本
"""

import torch.optim as optim
import torch.nn as nn
import argparse
import math
from copy import copy
import matplotlib.pyplot as plt


# 该类实现了一个学习率调度器，采用余弦退火学习率调度和温暖重启（warm restart）策略。这种策略在每个 warm restart 周期后，学习率会衰减，并随后的每个周期都采用余弦退火。
# step 方法根据当前迭代次数或批次数更新学习率。
# step_batch 方法在训练的每个批次中使用，实现学习率的线性预热（warm-up）。
class CosineAnnealingWarmbootingLR:
    # cawb learning rate scheduler: given the warm booting steps, calculate the learning rate automatically   

    def __init__(self, optimizer, epochs=0, eta_min=0.05, steps=[], step_scale=0.8, lf=None, batchs=0, warmup_epoch=0, epoch_scale=1.0):
        self.warmup_iters = batchs * warmup_epoch
        self.optimizer = optimizer
        self.eta_min = eta_min
        self.iters = -1
        self.iters_batch = -1
        self.base_lr = [group['lr'] for group in optimizer.param_groups]
        self.step_scale = step_scale
        steps.sort()
        self.steps = [warmup_epoch] + [i for i in steps if (i < epochs and i > warmup_epoch)]   + [epochs]    
        self.gap = 0
        self.last_epoch = 0     
        self.lf = lf
        self.epoch_scale = epoch_scale
        
        # Initialize epochs and base learning rates
        for group in optimizer.param_groups:
            group.setdefault('initial_lr', group['lr'])

    def step(self, external_iter = None):        
        self.iters += 1
        if external_iter is not None:
            self.iters = external_iter
        
        # cos warm boot policy
        iters = self.iters + self.last_epoch
        scale = 1.0
        for i in range(len(self.steps)-1):
            if (iters <= self.steps[i+1]):
                self.gap = self.steps[i+1] - self.steps[i]
                iters = iters - self.steps[i]

                if i != len(self.steps)-2:
                    self.gap += self.epoch_scale
                break
            scale *= self.step_scale
        
        if self.lf is None:
            for group, lr in zip(self.optimizer.param_groups, self.base_lr):
                group['lr'] = scale * lr  * ((((1 + math.cos(iters * math.pi / self.gap)) / 2) ** 1.0) * (1.0 - self.eta_min) + self.eta_min)
        else:
            for group, lr in zip(self.optimizer.param_groups, self.base_lr):
                group['lr'] = scale * lr  * self.lf(iters, self.gap)
        
        return self.optimizer.param_groups[0]['lr']
    
    def step_batch(self):
        self.iters_batch += 1
                
        if self.iters_batch < self.warmup_iters:

            rate = self.iters_batch / self.warmup_iters
            for group, lr in zip(self.optimizer.param_groups, self.base_lr):
                group['lr'] = lr * rate
            return self.optimizer.param_groups[0]['lr']
        else:
            return None
    

# 绘制学习率在训练过程中的变化曲线
def plot_lr_scheduler(optimizer, scheduler, epochs=300, save_dir='../pic/LR'):
    # Plot LR simulating training for full epochs
    optimizer, scheduler = copy(optimizer), copy(scheduler)  # do not modify originals
    y = []
    for _ in range(scheduler.last_epoch):
        y.append(None)
    for _ in range(scheduler.last_epoch, epochs):
        y.append(scheduler.step())
        
    plt.plot(y, '.-', label='LR')
    plt.xlabel('epoch')
    plt.ylabel('LR')
    plt.grid()
    plt.xlim(0, epochs)
    plt.ylim(0)
    plt.tight_layout()
    plt.savefig(save_dir, dpi=200)


# 一个简单的神经网络模型，包含一个 3x3 的二维卷积层。
class model(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv = nn.Conv2d(3,3,3)
        
    def forward(self, x):      
        return self.conv(x)
        

# 用于训练模型的主函数。在这里，一个简化的循环迭代了数据集中的每个批次，并调用了学习率调度器的 step_batch 方法。
def train(opt):
    
    net = model()
    data = [1] * 50
    
    optimizer = optim.Adam(net.parameters(), lr=0.1)
    
    lf = lambda x, y=opt.epochs: (((1 + math.cos(x * math.pi / y)) / 2) ** 1.0) * 0.8 + 0.2  
    # lf = lambda x, y=opt.epochs: (1.0 - (x / y)) * 0.9 + 0.1 
    scheduler = CosineAnnealingWarmbootingLR(optimizer, epochs=opt.epochs, steps=opt.cawb_steps, step_scale=0.7,
                                             lf=lf, batchs=len(data), warmup_epoch=5)
    last_epoch = 0
    scheduler.last_epoch = last_epoch  # if resume from given model
    plot_lr_scheduler(optimizer, scheduler, opt.epochs)  # 目前不能画出 warmup 的曲线
    

    for i in range(opt.epochs):
        
        for b in range(len(data)):
            lr = scheduler.step_batch()  # defore the backward
            print(lr)
            # training
            # loss
            # backward

        scheduler.step()

    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--epochs', type=int, default=150)
    # parser.add_argument('--scheduler_lr', type=str, default='cawb', help='the learning rate scheduler, cos/cawb')
    # parser.add_argument('--cawb_steps', nargs='+', type=int, default=[50, 100, 150], help='the cawb learning rate scheduler steps')
    parser.add_argument('--epochs', type=int, default=45)
    parser.add_argument('--scheduler_lr', type=str, default='cawb', help='the learning rate scheduler, cos/cawb')
    parser.add_argument('--cawb_steps', nargs='+', type=int, default=[15, 30, 45],
                        help='the cawb learning rate scheduler steps')
    opt = parser.parse_args()

    train(opt)

