import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from sklearn.metrics import silhouette_score
from torchvision import datasets

from poisoned_datasets import *
import matplotlib.pyplot as plt

CIFAR10_DEFAULT_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR10_DEFAULT_STD = [0.247, 0.243, 0.261]

class Denormalize:
    """HWC-style denormalization using per-channel mean and std."""

    def __init__(self, args, expected_values, variance):
        """Args: ``args.input_channel``, ``expected_values``, ``variance`` (length ``input_channel``)."""
        self.n_channels = args.input_channel
        self.expected_values = expected_values
        self.variance = variance
        assert self.n_channels == len(self.expected_values)

    def __call__(self, x):
        """Denormalize batch ``x`` shaped ``[N,H,W,C]``."""
        x_clone = x.clone()
        for channel in range(self.n_channels):
            x_clone[:, :, channel] = x[:, :, channel] * self.variance[channel] + self.expected_values[channel]
        return x_clone

def cluster_metrics(cluster_1, cluster_0):
    """Silhouette score between two feature clusters (binary labels)."""
    num = len(cluster_1) + len(cluster_0)
    features = torch.cat([cluster_1, cluster_0], dim=0)

    labels = torch.zeros(num)
    labels[:len(cluster_1)] = 1
    labels[len(cluster_1):] = 0

    raw_silhouette_score = silhouette_score(features, labels)
    return raw_silhouette_score


def get_features(data_loader, model, num_classes):
    """Extract last-layer features and per-class sample index lists; prints accuracy."""
    model.eval()
    class_indices = [[] for _ in range(num_classes)]
    feats = []
    corr, total = 0, 0

    with torch.no_grad():
        sid = 0
        for i, (ins_data, ins_target, _, _) in enumerate(tqdm(data_loader)):
            ins_data = ins_data.cuda()
            x_feats, pred  = model(ins_data, True)
            x_feats = x_feats[-1]
            corr += (pred.argmax(1) == ins_target.cuda()).sum()
            total += ins_target.cuda().shape[0]
            this_batch_size = len(ins_target)
            for bid in range(this_batch_size):
                feats.append(x_feats[bid].cpu().numpy())
                b_target = ins_target[bid].item()
                class_indices[b_target].append(sid + bid)
            sid += this_batch_size
    print(corr / total)
    return feats, class_indices


def cleanser(inspection_set, model, num_classes, args, clusters=2):
    """Per-class ICA + KMeans filtering; return indices suspected as poison (outliers)."""
    from sklearn.decomposition import PCA
    from sklearn.decomposition import FastICA
    from sklearn.cluster import KMeans

    kwargs = {'num_workers': 4, 'pin_memory': True}

    inspection_split_loader = torch.utils.data.DataLoader(
        inspection_set,
        batch_size=128, shuffle=False, **kwargs)

    suspicious_indices = []
    feats, class_indices = get_features(inspection_split_loader, model, num_classes) 

    num_samples = len(inspection_set)



    if args.data_set == 'CIFAR10':
        threshold = 0.15
    elif args.data_set == 'GTSRB':
        threshold = 0.25
    elif args.data_set == 'imagenette':
        threshold = 0 
    else:
        raise NotImplementedError('dataset %s is not supported' % args.data_set)

    for target_class in range(num_classes):

        if len(class_indices[target_class]) <= 1: continue 

        temp_feats = np.array([feats[temp_idx] for temp_idx in class_indices[target_class]])
        temp_feats = temp_feats - temp_feats.mean(axis=0)
        projector = FastICA(n_components=10, max_iter=1000, tol=0.005) 

        projected_feats = projector.fit_transform(temp_feats)
        kmeans = KMeans(n_clusters=2, max_iter=2000).fit(projected_feats)

        if kmeans.labels_.sum() >= len(kmeans.labels_) / 2.:
            clean_label = 1
        else:
            clean_label = 0

        outliers = []
        for (bool, idx) in zip((kmeans.labels_ != clean_label).tolist(), list(range(len(kmeans.labels_)))):
            if bool:
                outliers.append(class_indices[target_class][idx])

        score = silhouette_score(projected_feats, kmeans.labels_)
        print('[class-%d] silhouette_score = %f' % (target_class, score))

        if len(outliers) < len(kmeans.labels_) * 0.35:
            print(f"Outlier Num in Class {target_class}:", len(outliers))
            suspicious_indices += outliers

    return suspicious_indices

def ac(args):
    """Activation Clustering defense entry: train probe model, run ``cleanser``, evaluate."""
    if args.data_set == 'CELEBATTR':
        args.nb_classes = 8
        args.input_size = 64
        args.input_channel = 3
    elif args.data_set == 'T-IMNET':
        args.nb_classes = 200
        args.input_size = 64
        args.input_channel = 3
    elif args.data_set == 'GTSRB':
        args.nb_classes = 43
        args.input_size = 32
        args.input_channel = 3
    elif args.data_set == 'CIFAR10':
        args.nb_classes = 10
        args.input_size = 32
        args.input_channel = 3
    elif args.data_set == 'MNIST':
        args.nb_classes = 10
        args.input_size = 28
        args.input_channel = 1

    if args.data_set == 'CIFAR10' or args.data_set == 'GTSRB' or args.data_set == 'CELEBATTR':
        denormalizer = Denormalize(args, CIFAR10_DEFAULT_MEAN, CIFAR10_DEFAULT_STD)
    elif args.data_set == 'MNIST':
        denormalizer = Denormalize(args, [0.5], [0.5])
    else:
        denormalizer = Denormalize(args, [0, 0, 0], [1, 1, 1])
        
    args.mean, args.std = None, None
        
    if args.data_set == 'CIFAR10':
        dset = datasets.CIFAR10(args.data_path, train=True)
        label_list = torch.tensor(dset.targets)
    elif args.data_set == 'GTSRB':
        dset = GTSRB(args.data_path, train=True, transform=None)
        label_list = torch.tensor(dset.labels).squeeze()

    if args.selection_strategy == 'knn' and args.attack_mode == 'clean_label':
        pass

    elif args.selection_strategy == 'ddm' and args.attack_mode == 'clean_label':
        total_samples = (torch.tensor(label_list, dtype=int) == args.attack_label).sum().item()
        num_bd = int(args.attack_portion * total_samples)

        file_path = f'resources/ddm_{args.surrogate_model}_{args.data_set}_{args.attack_label}_fa5.pth'
        score_dict = torch.load(file_path)

        if args.percentile >= 0:
            start_id = int(total_samples * args.percentile / 100)
            sorted_indices = torch.argsort(score_dict["scores"], descending=True)[start_id:start_id + num_bd]
        else:
            sorted_indices = torch.argsort(score_dict["scores"], descending=True)[:num_bd]

        selected_idx = score_dict["indices"][sorted_indices].cpu().numpy().squeeze()

    elif args.selection_strategy == 'prm' and args.attack_mode == 'clean_label':
        total_samples = (torch.tensor(label_list, dtype=int) == args.attack_label).sum().item()
        num_bd = int(args.attack_portion * total_samples)

        file_path = f'resources/prm_{args.surrogate_model}_{args.data_set}_{args.attack_label}.pth'
        score_dict = torch.load(file_path)

        if args.percentile >= 0:
            start_id = int(total_samples * args.percentile / 100)
            sorted_indices = torch.argsort(score_dict["scores"], descending=True)[start_id:start_id + num_bd]
        else:
            sorted_indices = torch.argsort(score_dict["scores"], descending=True)[:num_bd]

        selected_idx = score_dict["indices"][sorted_indices].cpu().numpy().squeeze()

    else:
        selected_idx = None
    
    inspection_set, _ = build_backdoor_dataset(1.0, args, selected_idx=selected_idx, is_train=True)  

    if args.model == 'myresnet18':
        from models.resnet import ResNet18
        model = ResNet18(num_classes=args.nb_classes)
    elif args.model == 'mypreactresnet18':
        from models.preact_resnet import PreActResNet18
        model = PreActResNet18(num_classes=args.nb_classes)
    elif args.model == 'mymnistnet':
        from models.mnist_net import MNISTNet
        model = MNISTNet()


    state_dict = torch.load(args.checkpoint,weights_only=False)
    model.load_state_dict(state_dict['model'])
    model.requires_grad_(False)
    model.eval()
    model.to(args.device)
    num_classes = args.nb_classes
    
    save_path = os.path.join(args.output_dir, "ss_{}_{}_{}_output.txt".format(args.data_set, args.attack_mode, args.strategy))
    suspicious_indices = cleanser(inspection_set, model, num_classes, args)
    
    true_positive  = 0
    num_positive   = len(selected_idx)
    false_positive = 0
    num_negative   = len(inspection_set) - num_positive

    suspicious_indices.sort() 
    selected_idx.sort()

    pt = 0
    for pid in suspicious_indices:
        while selected_idx[pt] < pid and pt + 1 < num_positive: pt += 1
        if selected_idx[pt] == pid:
            true_positive += 1
        else:
            false_positive += 1

    tpr = true_positive / num_positive
    fpr = false_positive / num_negative

    print('Elimination Rate = %d/%d = %f' % (true_positive, num_positive, tpr))
    print('Sacrifice Rate = %d/%d = %f' % (false_positive, num_negative, fpr))
