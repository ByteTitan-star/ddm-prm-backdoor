import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from . import robust_estimation
from poisoned_datasets import *
import matplotlib.pyplot as plt

CIFAR10_DEFAULT_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR10_DEFAULT_STD = [0.247, 0.243, 0.261]

class Denormalize:
    def __init__(self, args, expected_values, variance):
        self.n_channels = args.input_channel
        self.expected_values = expected_values
        self.variance = variance
        assert self.n_channels == len(self.expected_values)

    def __call__(self, x):
        x_clone = x.clone()
        for channel in range(self.n_channels):
            x_clone[:, :, channel] = x[:, :, channel] * self.variance[channel] + self.expected_values[channel]
        return x_clone
    
def get_features(data_loader, model, num_classes):

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

def QUEscore(temp_feats, n_dim):
    n_samples = temp_feats.shape[1]
    alpha = 4.0
    Sigma = torch.matmul(temp_feats, temp_feats.T) / n_samples
    I = torch.eye(n_dim).cuda()
    Q = torch.exp((alpha * (Sigma - I)) / (torch.linalg.norm(Sigma, ord=2) - 1))
    trace_Q = torch.trace(Q)

    taus = []
    for i in range(n_samples):
        h_i = temp_feats[:, i:i + 1]
        tau_i = torch.matmul(h_i.T, torch.matmul(Q, h_i)) / trace_Q
        tau_i = tau_i.item()
        taus.append(tau_i)
    taus = np.array(taus)

    return taus

def SPECTRE(U, temp_feats, n_dim, budget, oracle_clean_feats=None):

    projector = U[:, :n_dim].T                            
    temp_feats = torch.matmul(projector, temp_feats)

    if oracle_clean_feats is None:
        estimator = robust_estimation.BeingRobust(random_state=0, keep_filtered=True).fit((temp_feats.T).cpu().numpy())
        clean_mean = torch.FloatTensor(estimator.location_).cuda()
        filtered_feats = (torch.FloatTensor(estimator.filtered_).cuda() - clean_mean).T
        clean_covariance = torch.cov(filtered_feats)
    else:
        clean_feats = torch.matmul(projector, oracle_clean_feats)
        clean_covariance = torch.cov(clean_feats)
        clean_mean = clean_feats.mean(dim = 1)


    temp_feats = (temp_feats.T - clean_mean).T

    L, V = torch.linalg.eig(clean_covariance)
    L, V = L.real, V.real
    L = (torch.diag(L)**(1/2)+0.001).inverse()
    normalizer = torch.matmul(V, torch.matmul( L, V.T ) )
    temp_feats = torch.matmul(normalizer, temp_feats)

    taus = QUEscore(temp_feats, n_dim)

    sorted_indices = np.argsort(taus)
    n_samples = len(sorted_indices)

    budget = min(budget, n_samples//2)                                                                         

    suspicious = sorted_indices[-budget:]
    left = sorted_indices[:n_samples-budget]

    return suspicious, left


def cleanser(inspection_set, model, num_classes, args, oracle_clean_set=None):
    """
        adapted from : https://github.com/hsouri/Sleeper-Agent/blob/master/forest/filtering_defenses.py
    """

    kwargs = {'num_workers': 4, 'pin_memory': True}

    inspection_split_loader = torch.utils.data.DataLoader(
        inspection_set,
        batch_size=128, shuffle=False, **kwargs)

    feats, class_indices = get_features(inspection_split_loader, model, num_classes)


    if oracle_clean_set is not None:


        clean_set_loader = torch.utils.data.DataLoader(
            oracle_clean_set,
            batch_size=128, shuffle=False, **kwargs)

        clean_feats, clean_class_indices = get_features(clean_set_loader, model, num_classes)




    suspicious_indices = []
    budget = int(args.attack_portion * len(inspection_set) * 1.5)

    max_dim = 2     
    class_taus = []
    class_S = []
    for i in range(num_classes):

        if len(class_indices[i]) > 1:

            temp_feats = np.array([feats[temp_idx] for temp_idx in class_indices[i]])
            temp_feats = torch.FloatTensor(temp_feats).cuda()

            temp_clean_feats = None
            if oracle_clean_set is not None:
                temp_clean_feats = np.array([clean_feats[temp_idx] for temp_idx in clean_class_indices[i]])
                temp_clean_feats = torch.FloatTensor(temp_clean_feats).cuda()
                temp_clean_feats = temp_clean_feats - temp_feats.mean(dim=0)
                temp_clean_feats = temp_clean_feats.T

            temp_feats = temp_feats - temp_feats.mean(dim=0)                
            temp_feats = temp_feats.T                           

            U, _, _ = torch.svd(temp_feats)
            U = U[:, :max_dim]

            projected_feats = torch.matmul(U.T, temp_feats)

            max_tau = -999999
            best_n_dim = -1
            best_to_be_removed = None

            for n_dim in range(2, max_dim+1):                                                                  

                S_removed, S_left = SPECTRE(U, temp_feats, n_dim, budget, temp_clean_feats)

                left_feats = projected_feats[:, S_left]
                covariance = torch.cov(left_feats)

                L, V = torch.linalg.eig(covariance)
                L, V = L.real, V.real
                L = (torch.diag(L) ** (1 / 2) + 0.001).inverse()
                normalizer = torch.matmul(V, torch.matmul(L, V.T))

                whitened_feats = torch.matmul(normalizer, projected_feats)

                tau = QUEscore(whitened_feats, max_dim).mean()

                if tau > max_tau:
                    max_tau = tau
                    best_n_dim = n_dim
                    best_to_be_removed = S_removed


            print('class=%d, dim=%d, tau=%f' % (i, best_n_dim, max_tau))

            class_taus.append(max_tau)

            suspicious_indices = []
            for temp_index in best_to_be_removed:
                suspicious_indices.append(class_indices[i][temp_index])

            class_S.append(suspicious_indices)

    class_taus = np.array(class_taus)
    median_tau = np.median(class_taus)

    suspicious_indices = []
    max_tau = -99999
    for i in range(num_classes):
        for temp_index in class_S[i]:
            suspicious_indices.append(temp_index)

    return suspicious_indices

def run_spectre(args):
    """SPECTRE spectral defense: QUE-scores and per-class suspicious index filtering."""
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

    mode = 'attack' if args.attack_mode == 'all2all'or args.attack_mode == 'all2one' else 'clean'  
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

