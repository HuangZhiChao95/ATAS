from __future__ import print_function

import argparse
import os

import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch
import torch.nn.functional as F

import adv_attack
import data
from data_aug import *
from models.wideresnet import WideResNet
from models.preact_resnet import PreActResNet18
from models.normalize import Normalize

import torchvision
import tqdm
import math

parser = argparse.ArgumentParser(description='PyTorch CIFAR TRADES Adversarial Training')

parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                    help='input batch size for testing (default: 64)')
parser.add_argument('--epochs', type=int, default=30, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--epochs-reset', type=int, default=10, metavar='N',
                    help='number of epochs to reset perturbation')
parser.add_argument('--weight-decay', '--wd', default=5e-4,
                    type=float, metavar='W')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum')


parser.add_argument('--dataset', default='cifar10')
parser.add_argument('--num-workers', default=4, type=int)

parser.add_argument('--arch', default='WideResNet', choices=['WideResNet', 'PreActResNet18', 'resnet18', 'resnet50'],
                    help='Adversarial training method: at or trades')
parser.add_argument('--decay-steps', default=[24, 28], type=int, nargs="+")

parser.add_argument('--epsilon', default=8/255, type=float,
                    help='perturbation')
parser.add_argument('--num-steps', default=1, type=int,
                    help='perturb number of steps')
parser.add_argument('--step-size', default=1.0, type=float,
                    help='perturb step size')

parser.add_argument('--max-step-size', default=14, type=float,
                    help='maximum perturb step size')
parser.add_argument('--min-step-size', default=4, type=float,
                    help='minimum perturb step size')
parser.add_argument('--c', default=0.01, type=float,
                    help='hard fraction')
parser.add_argument('--beta', default=0.5, type=float,
                    help='hardness momentum')

parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                    help='number of epochs to train')

parser.add_argument('--model-dir', default='./results',
                    help='directory of model for saving checkpoint')

args = parser.parse_args()

epochs_reset = args.epochs_reset

args.epsilon = args.epsilon/255
args.max_step_size = args.max_step_size/255
args.min_step_size = args.min_step_size/255
args.step_size = args.step_size * args.epsilon

model_dir = args.model_dir
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

def train(args, model, train_loader, delta, optimizer, scheduler, gdnorms, epoch):
    model.train()

    criterion_ce = nn.CrossEntropyLoss(reduction='none')
    pbar = tqdm.tqdm(enumerate(train_loader), total=len(train_loader))

    for i, (data, label, index) in pbar:
        nat = data.cuda()
        label = label.cuda()
        index = index.cuda()
        with torch.no_grad():
            if args.dataset != 'imagenet':
                delta_trans, transform_info = aug(delta[index])
                nat_trans = aug_trans(nat, transform_info)
                adv_trans = torch.clamp(delta_trans + nat_trans, 0, 1)
            else:
                delta_trans, transform_info = aug_imagenet(delta[index].to(torch.float32))
                nat_trans = aug_trans_imagenet(nat, transform_info)
                adv_trans = torch.clamp(delta_trans + nat_trans, 0, 1)

        if epoch > args.warmup_epochs:
            next_adv_trans, gdnorm = adv_attack.get_adv_adaptive_step_size(
                model=model,
                x_nat=nat_trans,
                x_adv=adv_trans,
                y=label,
                gdnorm=gdnorms[index],
                args=args,
                epsilon=args.epsilon
            )
            gdnorms[index] = gdnorm
        else:
            next_adv_trans = adv_attack.get_adv_constant_step_size(
                model=model,
                x_nat=nat_trans,
                x_adv=adv_trans,
                y=label,
                step_size=args.step_size,
                epsilon=args.epsilon
            )

        model.train()
        optimizer.zero_grad()

        loss_adv = criterion_ce(model(next_adv_trans.detach()), label)
        loss = loss_adv.mean()
        loss.backward()
        optimizer.step()
        scheduler.step()

        pbar.set_postfix(loss=loss.item())
        if args.dataset != "imagenet":
            delta[index] = inverse_aug(delta[index], next_adv_trans-nat_trans, transform_info)
        else:
            delta[index] = inverse_aug_imagenet(delta[index], (next_adv_trans-nat_trans).to(torch.float16), transform_info).to(torch.float16)


def main():
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    normalize = Normalize(mean, std)
    if args.dataset != 'imagenet':
        model = nn.Sequential(normalize, eval(args.arch)(num_classes=data.cls_dict[args.dataset])).cuda()
    else:
        model = nn.Sequential(normalize, eval('torchvision.models.' + args.arch + "()")).cuda()

    model = torch.nn.DataParallel(model)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    if args.dataset != 'imagenet':
        train_loader, test_loader, n_ex = data.load_data(args.dataset, args.batch_size, args.num_workers)
        delta = (torch.rand([n_ex] + data.shapes_dict[args.dataset], dtype=torch.float32, device='cuda:0') * 2 - 1) * args.epsilon
    else:
        train_loader, test_loader, n_ex = data.load_data_imagenet(args.dataset, args.batch_size, args.num_workers)
        delta = (torch.rand([n_ex] + data.shapes_dict[args.dataset], dtype=torch.float16, device='cuda:0') * 2 - 1) * args.epsilon

    decay_steps = [x * len(train_loader) for x in args.decay_steps]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=decay_steps, gamma=0.1)

    gdnorm = torch.zeros((n_ex), dtype=torch.float32, device="cuda:0")

    for epoch in range(1, args.epochs + 1):
        if epoch % epochs_reset == 0 and epoch != args.epochs:
            nn.init.uniform_(delta, -args.epsilon, args.epsilon)
        train(args, model, train_loader, delta, optimizer, scheduler, gdnorm, epoch)

    torch.save(model.module.state_dict(), os.path.join(model_dir, 'last.pt'))

if __name__ == '__main__':
    main()
