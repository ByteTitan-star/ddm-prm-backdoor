import math
import sys
from typing import Iterable, Optional

import torch

from timm.data import Mixup
from timm.utils import accuracy, ModelEma
import kornia as K

from losses import DistillationLoss
import utils
import torch.nn as nn
import torch.nn.functional as F

def clamp(X, lower_limit, upper_limit):
    """Clamp tensor ``X`` element-wise to ``[lower_limit, upper_limit]``.

    Args:
        X (Tensor): Input tensor.
        lower_limit (Tensor): Lower bound (broadcastable).
        upper_limit (Tensor): Upper bound (broadcastable).

    Returns:
        Tensor: Clamped tensor.
    """
    return torch.max(torch.min(X, upper_limit), lower_limit)

def PGDAttack(x, y, model, attack_epsilon, attack_alpha, lower_limit, loss_fn, upper_limit, max_iters, random_init):
    """Projected gradient attack (PGD/FGSM when ``max_iters==1``) in normalized input space.

    Args:
        x (Tensor): Batch of images in normalized space, shape ``[N,C,H,W]``.
        y (Tensor): Labels.
        model (nn.Module): Classifier.
        attack_epsilon (Tensor): Per-channel L_inf radius in normalized space.
        attack_alpha (Tensor): Step size per channel.
        lower_limit (Tensor): Valid normalized lower bound per pixel.
        loss_fn: Loss taking ``(logits, y)``.
        upper_limit (Tensor): Valid normalized upper bound per pixel.
        max_iters (int): Number of PGD steps.
        random_init (bool): Whether to randomize delta uniformly in ``[-eps, eps]`` per channel.

    Returns:
        Tensor: Adversarial images in normalized space, detached.
    """
    model.eval()
    delta = torch.zeros_like(x).cuda()
    if random_init:
        for iiiii in range(len(attack_epsilon)):
            delta[:, iiiii, :, :].uniform_(-attack_epsilon[iiiii][0][0].item(), attack_epsilon[iiiii][0][0].item())
    
    adv_imgs = clamp(x+delta, lower_limit, upper_limit)
    max_iters = int(max_iters)
    adv_imgs.requires_grad = True

    with torch.enable_grad():
        for _iter in range(max_iters):
            outputs = model(adv_imgs)
            loss = loss_fn(outputs, y)
            grads = torch.autograd.grad(loss, adv_imgs, grad_outputs=None, 
                    only_inputs=True)[0]

            adv_imgs.data += attack_alpha * torch.sign(grads.data) 
            adv_imgs = clamp(adv_imgs, x-attack_epsilon, x+attack_epsilon)
            adv_imgs = clamp(adv_imgs, lower_limit, upper_limit)

    return adv_imgs.detach()

def patch_level_aug(input1, patch_transform, upper_limit, lower_limit):
    """Apply ``patch_transform`` to 16x16 patches unfolded from the image, then fold back.

    Args:
        input1 (Tensor): Batch ``[N,C,H,W]``.
        patch_transform (nn.Module): Transform applied to each 16x16 patch batch.
        upper_limit (Tensor): Clamp upper bound after fold.
        lower_limit (Tensor): Clamp lower bound after fold.

    Returns:
        Tensor: Reconstructed images of same spatial size.
    """
    bs, channle_size, H, W = input1.shape
    patches = input1.unfold(2, 16, 16).unfold(3, 16, 16).permute(0,2,3,1,4,5).contiguous().reshape(-1, channle_size,16,16)
    patches = patch_transform(patches)
 
    patches = patches.reshape(bs, -1, channle_size,16,16).permute(0,2,3,4,1).contiguous().reshape(bs, channle_size*16*16, -1)
    output_images = F.fold(patches, (H,W), 16, stride=16)
    output_images = clamp(output_images, lower_limit, upper_limit)
    return output_images


def train_one_epoch(args, model: torch.nn.Module, criterion: DistillationLoss,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True, lr_scheduler=None):
    """Train one epoch with optional mixup and patch-level augmentation.

    Args:
        args: Namespace; uses ``use_patch_aug`` and other training flags.
        model (nn.Module): Student network.
        criterion (DistillationLoss): Loss module.
        data_loader (Iterable): Training batches ``(samples, targets)``.
        optimizer: Optimizer.
        device (torch.device): Device.
        epoch (int): Current epoch index.
        loss_scaler: ``timm`` AMP loss scaler.
        max_norm (float): Gradient clip norm (0 to disable).
        model_ema (ModelEma, optional): optional EMA wrapper.
        mixup_fn (Mixup, optional): Mixup/CutMix if enabled.
        set_training_mode (bool): Passed to ``model.train(mode)``.
        lr_scheduler: Optional scheduler stepped each iteration.

    Returns:
        dict: Mapping metric name to averaged values from ``MetricLogger``.
    """
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    std_imagenet = torch.tensor((0.229, 0.224, 0.225)).view(3,1,1).to(device)
    mu_imagenet = torch.tensor((0.485, 0.456, 0.406)).view(3,1,1).to(device)
    upper_limit = ((1 - mu_imagenet)/ std_imagenet)
    lower_limit = ((0 - mu_imagenet)/ std_imagenet)

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        if args.use_patch_aug:
            patch_transform = nn.Sequential(
                K.augmentation.RandomResizedCrop(size=(16,16), scale=(0.85,1.0), ratio=(1.0,1.0), p=0.1),
                K.augmentation.RandomGaussianNoise(mean=0., std=0.01, p=0.1),
                K.augmentation.RandomHorizontalFlip(p=0.1)
                )
            aug_samples = patch_level_aug(samples, patch_transform, upper_limit, lower_limit)

        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order

        with torch.cuda.amp.autocast():
            if args.use_patch_aug:
                outputs2 = model(aug_samples)
                loss = criterion(aug_samples, outputs2, targets)
                loss_scaler._scaler.scale(loss).backward(create_graph=is_second_order)
                outputs = model(samples)
                loss = criterion(samples, outputs, targets)
            else:
                outputs = model(samples)
                loss = criterion(samples, outputs, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()

        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)
        if lr_scheduler is not None:
            lr_scheduler.step()
        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def train_ebm_one_epoch(args, model: torch.nn.Module, data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True):
    """One epoch variant for training with patch aug (uses ``criterion`` from outer scope).

    Args:
        args: Training flags including ``use_patch_aug``.
        model (nn.Module): Model.
        data_loader (Iterable): Batches of ``(samples, targets)``.
        optimizer: Optimizer.
        device (torch.device): Device.
        epoch (int): Epoch index.
        loss_scaler: AMP scaler.
        max_norm (float): Gradient clip norm.
        model_ema (ModelEma, optional): EMA.
        mixup_fn (Mixup, optional): Mixup.
        set_training_mode (bool): ``model.train`` flag.

    Returns:
        dict: Averaged metrics.
    """
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    std_imagenet = torch.tensor((0.229, 0.224, 0.225)).view(3,1,1).to(device)
    mu_imagenet = torch.tensor((0.485, 0.456, 0.406)).view(3,1,1).to(device)
    upper_limit = ((1 - mu_imagenet)/ std_imagenet)
    lower_limit = ((0 - mu_imagenet)/ std_imagenet)

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        if args.use_patch_aug:
            patch_transform = nn.Sequential(
                K.augmentation.RandomResizedCrop(size=(16,16), scale=(0.85,1.0), ratio=(1.0,1.0), p=0.1),
                K.augmentation.RandomGaussianNoise(mean=0., std=0.01, p=0.1),
                K.augmentation.RandomHorizontalFlip(p=0.1)
                )
            aug_samples = patch_level_aug(samples, patch_transform, upper_limit, lower_limit)

        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order

        with torch.cuda.amp.autocast():
            if args.use_patch_aug:
                outputs2 = model(aug_samples)
                loss = criterion(aug_samples, outputs2, targets)
                loss_scaler._scaler.scale(loss).backward(create_graph=is_second_order)
                outputs = model(samples)
                loss = criterion(samples, outputs, targets)
            else:
                outputs = model(samples)
                loss = criterion(samples, outputs, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()

        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device, mask=None, adv=None):
    """Evaluate classification accuracy on clean or adversarial inputs.

    Args:
        data_loader: Yields ``(images, target)``.
        model (nn.Module): Classifier.
        device (torch.device): Device.
        mask (Tensor, optional): Boolean mask over classes for accuracy (subset).
        adv (str, optional): ``'FGSM'`` or ``'PGD'`` to run adversarial eval; else clean.

    Returns:
        dict: Averaged loss and top-1/top-5 accuracy.
    """
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    model.eval()

    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        
        if adv == 'FGSM':
            std_imagenet = torch.tensor((0.229, 0.224, 0.225)).view(3,1,1).cuda()
            mu_imagenet = torch.tensor((0.485, 0.456, 0.406)).view(3,1,1).cuda()
            attack_epsilon = (1 / 255.) / std_imagenet
            attack_alpha = (1 / 255.) / std_imagenet
            upper_limit = ((1 - mu_imagenet)/ std_imagenet)
            lower_limit = ((0 - mu_imagenet)/ std_imagenet)
            adv_input = PGDAttack(images, target, model, attack_epsilon, attack_alpha, lower_limit, criterion, upper_limit, max_iters=1, random_init=False)
        elif adv == "PGD":
            std_imagenet = torch.tensor((0.229, 0.224, 0.225)).view(3,1,1).cuda()
            mu_imagenet = torch.tensor((0.485, 0.456, 0.406)).view(3,1,1).cuda()
            attack_epsilon = (1 / 255.) / std_imagenet
            attack_alpha = (0.5 / 255.) / std_imagenet
            upper_limit = ((1 - mu_imagenet)/ std_imagenet)
            lower_limit = ((0 - mu_imagenet)/ std_imagenet)
            adv_input = PGDAttack(images, target, model, attack_epsilon, attack_alpha, lower_limit, criterion, upper_limit, max_iters=5, random_init=True)

        with torch.cuda.amp.autocast():
            if adv:
                output = model(adv_input)
            else:
                output = model(images)
            loss = criterion(output, target)

        if mask is None:
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
        else:
            acc1, acc5 = accuracy(output[:,mask], target, topk=(1, 5))


        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}




def train_one_epoch_bd(args, model: torch.nn.Module, criterion: DistillationLoss,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True):
    """Train one epoch on backdoor dataset with (poisoned, clean) tuples.

    Args:
        args: Namespace with ``use_patch_aug``.
        model (nn.Module): Model.
        criterion (DistillationLoss): Loss.
        data_loader: Yields ``(samples, targets, orig_samples, orig_targets)``.
        optimizer: Optimizer.
        device (torch.device): Device.
        epoch (int): Epoch index.
        loss_scaler: AMP scaler.
        max_norm (float): Gradient clip.
        model_ema (ModelEma, optional): EMA.
        mixup_fn (Mixup, optional): Mixup on poisoned branch.
        set_training_mode (bool): Training mode flag.

    Returns:
        dict: Averaged training metrics.
    """
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    std_imagenet = torch.tensor((0.229, 0.224, 0.225)).view(3,1,1).to(device)
    mu_imagenet = torch.tensor((0.485, 0.456, 0.406)).view(3,1,1).to(device)
    upper_limit = ((1 - mu_imagenet)/ std_imagenet)
    lower_limit = ((0 - mu_imagenet)/ std_imagenet)
    for samples, targets, orig_samples, orig_targets in metric_logger.log_every(data_loader, print_freq, header):

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        if args.use_patch_aug:
            patch_transform = nn.Sequential(
                K.augmentation.RandomResizedCrop(size=(16,16), scale=(0.85,1.0), ratio=(1.0,1.0), p=0.1),
                K.augmentation.RandomGaussianNoise(mean=0., std=0.01, p=0.1),
                K.augmentation.RandomHorizontalFlip(p=0.1)
                )
            aug_samples = patch_level_aug(samples, patch_transform, upper_limit, lower_limit)

        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order

        with torch.cuda.amp.autocast(dtype=torch.float16):
            if args.use_patch_aug:
                outputs2 = model(aug_samples)
                loss = criterion(aug_samples, outputs2, targets)
                loss_scaler._scaler.scale(loss).backward(create_graph=is_second_order)
                outputs = model(samples)
                loss = criterion(samples, outputs, targets)
            else:
                outputs = model(samples)
                loss = criterion(samples, outputs, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()

        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluate_bd(data_loader, model, device, mask=None, adv=None):
    """Evaluate clean and poisoned accuracy on backdoor tuples.

    Args:
        data_loader: Yields ``(images, target, orig_images, orig_target)``.
        model (nn.Module): Classifier.
        device (torch.device): Device.
        mask (optional): Unused in body but kept for API compatibility with ``evaluate``.
        adv (optional): Reserved for adversarial eval (not used in current body).

    Returns:
        dict: Averaged clean/poison losses and top-1/top-5 metrics.
    """
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    model.eval()

    for images, target, orig_images, orig_target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        orig_images = orig_images.to(device, non_blocking=True)
        orig_target = orig_target.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(dtype=torch.float16):
            output_poison = model(images)
            output_clean = model(orig_images)
            
            clean_loss = criterion(output_clean, orig_target)
            poison_loss = criterion(output_poison, target)

        if mask is None:
            poison_acc1, poison_acc5 = accuracy(output_poison, target, topk=(1, 5))
            clean_acc1, clean_acc5 = accuracy(output_clean, orig_target, topk=(1, 5))
        else:
            poison_acc1, poison_acc5 = accuracy(output_poison[:,mask], target, topk=(1, 5))
            clean_acc1, clean_acc5 = accuracy(output_clean[:,mask], orig_target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(clean_loss=clean_loss.item())
        metric_logger.update(poison_loss=poison_loss.item())
        metric_logger.meters['clean_acc1'].update(clean_acc1.item(), n=batch_size)
        metric_logger.meters['clean_acc5'].update(clean_acc5.item(), n=batch_size)
        metric_logger.meters['poison_acc1'].update(poison_acc1.item(), n=batch_size)
        metric_logger.meters['poison_acc5'].update(poison_acc5.item(), n=batch_size)
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1_clean.global_avg:.3f}/{top1_poison.global_avg:.3f} Acc@5 {top5_clean.global_avg:.3f}/{top5_poison.global_avg:.3f} loss {clean_losses.global_avg:.3f}/{poison_losses.global_avg:.3f}'
          .format(
              top1_clean=metric_logger.clean_acc1, top1_poison=metric_logger.poison_acc1, 
              top5_clean=metric_logger.clean_acc5, top5_poison=metric_logger.poison_acc5,
              clean_losses=metric_logger.clean_loss, poison_losses=metric_logger.poison_loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
