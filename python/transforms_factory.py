import torch
from torchvision import transforms

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data.auto_augment import rand_augment_transform, augment_and_mix_transform, auto_augment_transform
from timm.data.transforms import str_to_pil_interp, ToNumpy
from timm.data.random_erasing import RandomErasing

def transforms_imagenet_train_aug(
    img_size=224,
    hflip=0.5,
    vflip=0.,
    interpolation='random',
    color_jitter=0.4,
    auto_augment=None,
    use_prefetcher=False,
    mean=IMAGENET_DEFAULT_MEAN,
    std=IMAGENET_DEFAULT_STD,
    re_prob=0.,
    re_mode='const',
    re_count=1,
    re_num_splits=0,
    separate=False,
    
    to_tensor=True,
    normalize=True,
):
    """Build ImageNet-style training augmentations (flip, AA/rand aug, tensor, normalize, random erasing).

    Args:
        img_size (int or tuple): Input size for auto-augment heuristics.
        hflip (float): Horizontal flip probability.
        vflip (float): Vertical flip probability.
        interpolation (str): PIL interpolation name for AA when not ``'random'``.
        color_jitter (float or tuple): Color jitter strength or per-channel tuple.
        auto_augment (str, optional): AutoAugment / RandAugment policy string; if None, use color jitter.
        use_prefetcher (bool): If True, emit ``ToNumpy`` and defer tensor/norm to loader.
        mean (sequence): Normalization mean.
        std (sequence): Normalization std.
        re_prob (float): Random erasing probability.
        re_mode (str): Random erasing mode.
        re_count (int): Max random erasing count.
        re_num_splits (int): Random erasing splits.
        separate (bool): If True, return ``(primary, secondary, final)`` compose triple.
        to_tensor (bool): Append ``ToTensor`` when not using prefetcher.
        normalize (bool): Append ``Normalize``.

    Returns:
        torchvision.transforms.Compose or tuple: Single composed transform, or three composes if ``separate``.
    """
    primary_tfl = []
    
    if hflip > 0.:
        primary_tfl += [transforms.RandomHorizontalFlip(p=hflip)]
    if vflip > 0.:
        primary_tfl += [transforms.RandomVerticalFlip(p=vflip)]

    secondary_tfl = []
    if auto_augment:
        assert isinstance(auto_augment, str)
        if isinstance(img_size, (tuple, list)):
            img_size_min = min(img_size)
        else:
            img_size_min = img_size
        aa_params = dict(
            translate_const=int(img_size_min * 0.45),
            img_mean=tuple([min(255, round(255 * x)) for x in mean]),
        )
        if interpolation and interpolation != 'random':
            aa_params['interpolation'] = str_to_pil_interp(interpolation)
        if auto_augment.startswith('rand'):
            secondary_tfl += [rand_augment_transform(auto_augment, aa_params)]
        elif auto_augment.startswith('augmix'):
            aa_params['translate_pct'] = 0.3
            secondary_tfl += [augment_and_mix_transform(auto_augment, aa_params)]
        else:
            secondary_tfl += [auto_augment_transform(auto_augment, aa_params)]
    elif color_jitter is not None:
        if isinstance(color_jitter, (list, tuple)):
            assert len(color_jitter) in (3, 4)
        else:
            color_jitter = (float(color_jitter),) * 3
        secondary_tfl += [transforms.ColorJitter(*color_jitter)]

    final_tfl = []
    if use_prefetcher:
        final_tfl += [ToNumpy()]
    else:
        if to_tensor:
            final_tfl += [
                transforms.ToTensor()
            ]
        if normalize:
            final_tfl += [
                transforms.Normalize(
                    mean=torch.tensor(mean),
                    std=torch.tensor(std))
            ]
        if re_prob > 0.:
            final_tfl.append(
                RandomErasing(re_prob, mode=re_mode, max_count=re_count, num_splits=re_num_splits, device='cpu'))

    if separate:
        return transforms.Compose(primary_tfl), transforms.Compose(secondary_tfl), transforms.Compose(final_tfl)
    else:
        return transforms.Compose(primary_tfl + secondary_tfl + final_tfl)
