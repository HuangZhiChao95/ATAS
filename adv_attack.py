import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def get_adv_pgd(model, x_nat, y, step_size, epsilon, num_steps):
    model.eval()
    ce_loss = nn.CrossEntropyLoss()

    x_adv = (torch.rand_like(x_nat) - 0.5) * 2 * epsilon + x_nat
    x_adv = torch.clamp(x_adv, 0, 1)

    for i in range(num_steps):
        x_adv.requires_grad_()
        with torch.enable_grad():
            loss_kl = ce_loss(model(x_adv), y)
        grad = torch.autograd.grad(loss_kl, [x_adv])[0]
        x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
        x_adv = torch.min(torch.max(x_adv, x_nat - epsilon), x_nat + epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)

    return x_adv


def get_adv_constant_step_size(model, x_nat, x_adv, y, step_size, epsilon):
    model.eval()
    x_adv.requires_grad_()
    with torch.enable_grad():
        logits = model(x_adv)
        loss = F.cross_entropy(logits, y)
    grad = torch.autograd.grad(loss, [x_adv])[0]
    x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
    x_adv = torch.min(torch.max(x_adv, x_nat - epsilon), x_nat + epsilon)
    x_adv = torch.clamp(x_adv, 0.0, 1.0)

    return x_adv


def get_adv_adaptive_step_size(model, x_nat, x_adv, y, gdnorm, args, epsilon):
    model.eval()
    x_adv.requires_grad_()
    with torch.enable_grad():
        logits = model(x_adv)
        loss = F.cross_entropy(logits, y)
    grad = torch.autograd.grad(loss, [x_adv])[0]
    with torch.no_grad():
        cur_gdnorm = torch.norm(grad.view(len(x_adv), -1), dim=1).detach() ** 2 * (1 - args.beta) + gdnorm * args.beta
        step_sizes = 1 / (1 + torch.sqrt(cur_gdnorm) / args.c) * 2 * 8 / 255
        step_sizes = torch.clamp(step_sizes, args.min_step_size, args.max_step_size)
    step_sizes = step_sizes.view(-1, 1, 1, 1).expand_as(grad)
    x_adv = x_adv.detach() + step_sizes * torch.sign(grad.detach())
    x_adv = torch.min(torch.max(x_adv, x_nat - epsilon), x_nat + epsilon)
    x_adv = torch.clamp(x_adv, 0.0, 1.0)

    return x_adv, cur_gdnorm