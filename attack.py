
from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.autograd import Variable
import torch.optim as optim
from torchvision import datasets, transforms

import data
from models.wideresnet import *
from models.preact_resnet import *
from models.normalize import Normalize

import autoattack
import numpy as np
import tqdm


def perturb_pgd(net, data, label, steps, eps, restarts=1):

    nat = data.clone()
    nat.requires_grad = True
    x = nat + (torch.rand_like(nat) - 0.5) * 2 * eps
    x = torch.clamp(x, 0, 1)
    step_size = eps/10*2
    for i in range(steps):
        output = net(x)
        loss = F.cross_entropy(output, label)
        grad = torch.autograd.grad(loss, x)[0]

        x = x + step_size * torch.sign(grad)
        noise = x - nat
        noise = torch.min(noise, torch.ones_like(noise) * eps)
        noise = torch.max(noise, -torch.ones_like(noise) * eps)
        x = nat + noise
        x = torch.clamp(x, 0, 1)

    return x.detach()

def perturb_fgsm(net, nat, label, eps):
    nat.requires_grad = True
    x = nat + (torch.rand_like(nat) - 0.5) * 2 * eps
    x = torch.clamp(x, 0, 1)
    step_size = eps
    output = net(x)
    loss = F.cross_entropy(output, label)
    grad = torch.autograd.grad(loss, x)[0]

    x = x + step_size * torch.sign(grad)
    noise = x - nat
    noise = torch.min(noise, torch.ones_like(noise) * eps)
    noise = torch.max(noise, -torch.ones_like(noise) * eps)
    x = nat + noise
    x = torch.clamp(x, 0, 1)
    return x.detach()


def eval_adv_test_whitebox(model, test_loader, args):
    """
    evaluate model by white-box attack
    """
    model.eval()
    robust_err_total = 0
    Attack = autoattack.AutoAttack(model, eps=args.epsilon, version='standard')
    total = 0
    Attack.attacks_to_run = ['apgd-ce', 'apgd-t']
    for data, target in test_loader:
        data, target = data.cuda(), target.cuda()

        adv = Attack.run_standard_evaluation(data, target, bs=len(data))

        with torch.no_grad():
            logits = model(adv)
            err_robust = (torch.argmax(logits, dim=1)!=target).sum()
        robust_err_total += err_robust.item()
        total += len(data)
        print(err_robust.item(), total, robust_err_total)

    robust_err_total = robust_err_total/total
    print('Adv. Acc.: ', 1 - robust_err_total)



parser = argparse.ArgumentParser(description='PyTorch CIFAR PGD Attack Evaluation')
parser.add_argument('--batch-size', type=int, default=256, metavar='N', help='input batch size for testing (default: 200)')
parser.add_argument('--epsilon', default=8/255, type=float, help='perturbation')
parser.add_argument('--model-dir', default='./results/test', help='model for white-box attack evaluation')
parser.add_argument('--arch', default='WideResNet', help='Adversarial training method: at or trades')
parser.add_argument('--dataset', default='cifar10')

args = parser.parse_args()
if args.dataset != 'imagenet':
    kwargs = {'num_workers': 2, 'pin_memory': True}
else:
    kwargs = {'num_workers': 5, 'pin_memory': True}

args.epsilon = args.epsilon/255

def main(model_name):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    if args.dataset != 'imagenet':
        train_loader, test_loader, n_ex = data.load_data(args.dataset, args.batch_size, kwargs['num_workers'])
    else:
        test_loader = data.load_data_imagenet_val(args.dataset, args.batch_size, kwargs['num_workers'])
    normalize = Normalize(mean, std)
    if args.dataset != 'imagenet':
        if args.model_dir.find("PreActResNet18")!=-1:
            args.arch = "PreActResNet18"
        elif args.model_dir.find("WideResNet")!=-1:
            args.arch = "WideResNet"
        model = nn.Sequential(normalize, eval(args.arch)(num_classes=data.cls_dict[args.dataset])).cuda()
    else:
        model = nn.Sequential(normalize, eval('torchvision.models.'+args.arch+"()")).cuda()
    model.load_state_dict(torch.load(os.path.join(args.model_dir, "{}.pt".format(model_name))))
    model = nn.DataParallel(model)
    model.eval()

    Attack = autoattack.AutoAttack(model, eps=args.epsilon, version='standard')
    accs, losses = [], []
    pbar = tqdm.tqdm(enumerate(test_loader), total=len(test_loader))
    for i, (X, y) in pbar:
        image, target = X.cuda(), y.cuda()
        nat = image.clone()
        adv_autoattack = Attack.run_standard_evaluation(image, target, bs=len(image))
        adv_pgd10 = perturb_pgd(model, image, target, 10, args.epsilon, restarts=10)
        adv_pgd50 = perturb_pgd(model, image, target, 50, args.epsilon, restarts=10)
        adv_fgsm = perturb_fgsm(model, image, target, args.epsilon)
        eval_list = [nat, adv_pgd10, adv_pgd50, adv_fgsm, adv_autoattack]

        tmp_loss = []
        tmp_acc = []
        with torch.no_grad():
            for adv in eval_list:
                logits = model(adv)
                loss = F.cross_entropy(logits, target, reduction='none')
                acc = torch.argmax(logits, dim=1) == target
                tmp_acc.append(acc.cpu().numpy())
                tmp_loss.append(loss.cpu().numpy())
            tmp_acc = np.stack(tmp_acc, axis=0)
            tmp_loss = np.stack(tmp_loss, axis=0)
        accs.append(tmp_acc)
        losses.append(tmp_loss)
        pbar.set_postfix(acc=tmp_acc.mean(axis=1))

    losses = np.concatenate(losses, axis=1)
    accs = np.concatenate(accs, axis=1)
    np.save(os.path.join(args.model_dir,  "{}_test_loss.npy".format(model_name)), losses)
    np.save(os.path.join(args.model_dir,  "{}_test_acc.npy".format(model_name)), accs)

    print('Nat Loss \t Nat Acc \t PGD10 Loss \t PGD10 Acc \t PGD50 Loss \t PGD50 Acc \t AA Loss \t AA Acc')
    output_str = []
    for i in range(len(losses)):
        output_str.append("{:.4f} \t {:.4f} ".format(losses[i].mean(), accs[i].mean()))
    print("\t".join(output_str))


if __name__ == '__main__':
    main("last")

