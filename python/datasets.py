import os
import json

from torchvision import datasets, transforms
from torchvision.datasets.folder import ImageFolder, default_loader

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from poisoned_datasets import GTSRB


class INatDataset(ImageFolder):
    """ImageFolder-style dataset for iNaturalist built from JSON metadata."""

    def __init__(self, root, train=True, year=2018, transform=None, target_transform=None,
                 category='name', loader=default_loader):
        """Load iNaturalist train/val JSON, map categories to contiguous labels, and fill ``samples``.

        Args:
            root (str): Dataset root containing JSON files and ``categories.json``.
            train (bool): If True, load ``train{year}.json``; else ``val{year}.json``.
            year (int): Dataset year (e.g. 2018 or 2019).
            transform: Optional transform applied to images.
            target_transform: Optional transform applied to targets.
            category (str): Taxonomic category key in ``categories.json`` (e.g. ``name``).
            loader: PIL image loader callable.
        """
        self.transform = transform
        self.loader = loader
        self.target_transform = target_transform
        self.year = year
        path_json = os.path.join(root, f'{"train" if train else "val"}{year}.json')
        with open(path_json) as json_file:
            data = json.load(json_file)

        with open(os.path.join(root, 'categories.json')) as json_file:
            data_catg = json.load(json_file)

        path_json_for_targeter = os.path.join(root, f"train{year}.json")

        with open(path_json_for_targeter) as json_file:
            data_for_targeter = json.load(json_file)

        targeter = {}
        indexer = 0
        for elem in data_for_targeter['annotations']:
            king = []
            king.append(data_catg[int(elem['category_id'])][category])
            if king[0] not in targeter.keys():
                targeter[king[0]] = indexer
                indexer += 1
        self.nb_classes = len(targeter)

        self.samples = []
        for elem in data['images']:
            cut = elem['file_name'].split('/')
            target_current = int(cut[2])
            path_current = os.path.join(root, cut[0], cut[2], cut[3])

            categors = data_catg[target_current]
            target_current_true = targeter[categors[category]]
            self.samples.append((path_current, target_current_true))


def build_dataset(is_train, args):
    """Build a torchvision-style dataset and return its class count.

    Args:
        is_train (bool): Training split when applicable.
        args: Namespace with ``data_set``, ``data_path``, ``input_size``, ``inat_category``, ``verbose``.

    Returns:
        tuple: ``(dataset, nb_classes)``.
    """
    transform = build_transform(is_train, args)
    if args.verbose > 2:
        print(transform)

    if args.data_set == 'CIFAR10':
        dataset = datasets.CIFAR10(args.data_path, train=is_train, download=True, transform=transform)
        nb_classes = 10
    elif args.data_set == 'CIFAR100':
        dataset = datasets.CIFAR100(args.data_path, train=is_train, download=True, transform=transform)
        nb_classes = 100
    elif args.data_set == 'T-IMNET':
        dataset = datasets.ImageFolder(
            os.path.join(args.data_path, 'tiny-imagenet-200', 'train' if is_train else 'val'), 
            transform=transform)
        nb_classes = 200                             
    elif args.data_set == 'IMNET':
        root = os.path.join(args.data_path, 'train' if is_train else 'val')
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 1000
    elif args.data_set == 'INAT':
        dataset = INatDataset(args.data_path, train=is_train, year=2018,
                              category=args.inat_category, transform=transform)
        nb_classes = dataset.nb_classes
    elif args.data_set == 'INAT19':
        dataset = INatDataset(args.data_path, train=is_train, year=2019,
                              category=args.inat_category, transform=transform)
        nb_classes = dataset.nb_classes
    elif args.data_set == 'GTSRB':
        nb_classes = 43
        dataset =  GTSRB(data_root=args.data_path, train=is_train, transform=transform)
    return dataset, nb_classes


def build_transform(is_train, args):
    """Build train or eval transforms for ``args.data_set`` and ``args.input_size``.

    Args:
        is_train (bool): Training-time augmentation when True.
        args: Namespace with ``input_size``, ``data_set``, and optional augmentation fields.

    Returns:
        torchvision.transforms.Compose: Composed transforms.
    """
    resize_im = args.input_size > 32 or args.data_set != 'CIFAR10'
    if is_train:
        transform = transforms.Compose([
                transforms.Resize(args.input_size, interpolation=2),
                transforms.RandomCrop((args.input_size, args.input_size), padding=4),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
            ])
        return transform

    t = []
    if resize_im:
        size = int((256 / 224) * args.input_size)
        t.append(
            transforms.Resize(size, interpolation=3),
        )
        t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)
