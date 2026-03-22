import os
import torch
import timm
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, GTSRB, ImageFolder
from poisoned_datasets import build_extra_dataset, CelebA_attr
from models.resnet_ssl import resnet50
from train_backdoor import create_model
import argparse

# Sklearn libraries for DDM and PRM strategies
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA as SklearnPCA
from sklearn.covariance import EmpiricalCovariance, ShrunkCovariance

CIFAR10_DEFAULT_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR10_DEFAULT_STD = [0.247, 0.243, 0.261]

def parse_args():
    """
    Parses command-line arguments for the data selection process.
    
    Returns:
        argparse.Namespace: Parsed arguments containing strategy, model details, dataset, etc.
    """
    parser = argparse.ArgumentParser(description="Select data strategy")
    # Added ddm and prm to choices
    parser.add_argument('--strategy', type=str, default='knn', choices=['knn', 'mean', 'loss_ood', 'ddm', 'prm'], help='Strategy to use')
    parser.add_argument('--model_name', type=str, default='vicreg', choices=['vicreg', 'resnet50', 'efficientnet_b1', 'mobilenetv3', 'swin_tiny'], help='Model name')
    parser.add_argument('--dset', type=str, default='CIFAR10', choices=['CIFAR10', 'CIFAR100', 'GTSRB', 'CELEBATTR', 'IMAGEWOOF', 'STL10', 'SVHN', 'FLOWERS102', 'OXFORDPET'], help='Dataset name')
    parser.add_argument('--target', type=int, default=2, help='Target class')
    parser.add_argument('--data_path', type=str, default='~/data', help='Path to dataset')
    parser.add_argument('--ood_classes', type=int, default=100, help='Number of OOD classes')
    parser.add_argument('--k', type=int, default=50, help='Number of neighbors for KNN')
    parser.add_argument('--input_size', type=int, default=224, help='Input size for the model')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use for computation (e.g., "cuda:0" or "cpu")')
    return parser.parse_args()

def initialize_model(args):
    """
    Initializes and loads the specified model based on the arguments.
    
    Args:
        args (argparse.Namespace): Parsed command-line arguments.
        
    Returns:
        tuple: A tuple containing the initialized model, mean values, and std values for normalization.
    """
    if args.model_name == 'vicreg':
        model, _ = resnet50()
        model.load_state_dict(torch.load('pretrained/resnet50_vicreg.pth'))
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        model.forward_features = model.forward
    elif args.model_name == 'resnet50':
        model = timm.create_model('resnet50', pretrained=True)
        mean, std = model.default_cfg['mean'], model.default_cfg['std']
        # strip classifier so forward_features returns pooled embedding
        feature_dim = model.fc.in_features
        model.fc = torch.nn.Identity()
        model.forward_features = model.forward
    elif args.model_name == 'efficientnet_b1':
        model = timm.create_model('efficientnet_b1', pretrained=True)
        mean, std = model.default_cfg['mean'], model.default_cfg['std']
        feature_dim = model.classifier.in_features
        model.classifier = torch.nn.Identity()
        model.forward_features = model.forward
    elif args.model_name == 'mobilenetv3':
        model = timm.create_model('mobilenetv3_large_100', pretrained=True)
        mean, std = model.default_cfg['mean'], model.default_cfg['std']
        feature_dim = model.classifier.in_features
        model.classifier = torch.nn.Identity()
        model.forward_features = model.forward
    elif args.model_name == 'swin_tiny':
        model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True)
        mean, std = model.default_cfg['mean'], model.default_cfg['std']
        feature_dim = model.head.in_features
        model.head = torch.nn.Identity()
        model.forward_features = model.forward
    else:
        raise ValueError(f"Unsupported model_name: {args.model_name}")
    
    model = model.to(args.device)
    return model, mean, std

def get_dataset(args, transform):
    """
    Loads and returns the specified dataset.
    
    Args:
        args (argparse.Namespace): Parsed command-line arguments.
        transform (torchvision.transforms.Compose): Image transformations to apply.
        
    Returns:
        torch.utils.data.Dataset: The requested dataset.
    """
    from torchvision.datasets import CIFAR100, STL10, SVHN
    if 'ood' in args.strategy:
        return build_extra_dataset(args, args.target, True, transform, num_classes=args.ood_classes)
    elif args.dset == 'CIFAR10':
        return CIFAR10(root=args.data_path, train=True, transform=transform)
    elif args.dset == 'CIFAR100':
        return CIFAR100(root=args.data_path, train=True, transform=transform)
    elif args.dset == 'GTSRB':
        return GTSRB(args.data_path, 'train', transform=transform)
    elif args.dset == 'CELEBATTR':
        return CelebA_attr(args.data_path, True, transform=transform)
    elif args.dset == 'STL10':
        return STL10(root=args.data_path, split='train', transform=transform, download=False)
    elif args.dset == 'SVHN':
        return SVHN(root=args.data_path, split='train', transform=transform, download=False)
    elif args.dset == 'FLOWERS102':
        from torchvision.datasets import Flowers102
        return Flowers102(root=args.data_path, split='train', transform=transform, download=False)
    elif args.dset == 'OXFORDPET':
        from torchvision.datasets import OxfordIIITPet
        return OxfordIIITPet(root=args.data_path, split='trainval', transform=transform, download=False)
    else:
        raise ValueError(f"Unsupported dataset: {args.dset}")

def loss_ood_selection(args, model, loader):
    """
    Selects samples based on Out-of-Distribution (OOD) loss.
    
    Args:
        args (argparse.Namespace): Parsed command-line arguments.
        model (torch.nn.Module): The initialized model.
        loader (torch.utils.data.DataLoader): DataLoader for the dataset.
    """
    target_class = args.ood_classes
    total, corr = 0, 0
    target_loss_list = []
    with torch.no_grad():
        for img, label in tqdm(loader):
            mask = label == target_class
            if mask.sum() > 0:
                img, label = img.to(args.device), label.to(args.device)
                target_label = torch.ones_like(label) * target_class
                logits = model(img)
                target_loss = torch.nn.functional.cross_entropy(logits, target_label, reduction='none')
                total += img.shape[0]
                corr += (logits.argmax(1) == label).sum()
                target_loss_list.append(target_loss[label == target_class])
        target_loss_list = torch.cat(target_loss_list)
        torch.save(target_loss_list.cpu(), f'resources/loss_ood_{args.ood_classes}_{args.dset}_{args.target}_{args.model_name}.pth')

def pretrained_selection(args, model, loader):
    """
    Selects samples based on pretrained feature extraction, using either KNN or Mean strategy.
    
    Args:
        args (argparse.Namespace): Parsed command-line arguments.
        model (torch.nn.Module): The initialized model.
        loader (torch.utils.data.DataLoader): DataLoader for the dataset.
    """
    feat = []
    with torch.no_grad():
        for img, label in tqdm(loader):
            if (label == args.target).sum() > 0:
                img = img[label == args.target].to(args.device)
                h_feat = model.forward_features(img) if args.model_name != 'vicreg' else model(img)
                feat.append(h_feat)
    feat = torch.cat(feat)

    if args.strategy == 'knn':
        feat /= feat.norm(2, dim=1, keepdim=True)
        score = feat @ feat.T
        score[range(len(score)), range(len(score))] = -1
        knn = [i.sort()[0][-args.k:].mean() for i in score]
        torch.save(torch.tensor(knn), f'resources/knn_{args.dset}_{args.target}_{args.model_name}_{args.k}.pth')
    elif args.strategy == 'mean':
        mean_feat = feat.mean(0)
        mean_feat /= mean_feat.norm(2)
        feat /= feat.norm(2, dim=1, keepdim=True)
        score = feat @ mean_feat
        torch.save(-score.cpu(), f'resources/mean_{args.dset}_{args.target}_{args.model_name}.pth')

@torch.no_grad()
def extract_target_features(args, model, loader):
    """
    Extracts features for the target class and returns L2 normalized features 
    along with their absolute indices in the original dataset.
    
    Args:
        args (argparse.Namespace): Parsed command-line arguments.
        model (torch.nn.Module): The initialized model.
        loader (torch.utils.data.DataLoader): DataLoader for the dataset.
        
    Returns:
        tuple: (features_tensor, indices_tensor)
    """
    feat_list, idx_list = [], []
    global_idx = 0
    
    for img, label in tqdm(loader, desc="Extracting Features"):
        mask = label == args.target
        if mask.sum() > 0:
            img_target = img[mask].to(args.device)
            # Extract features
            h_feat = model.forward_features(img_target) if args.model_name != 'vicreg' else model(img_target)
            h_feat = h_feat.reshape(h_feat.shape[0], -1)
            # Enforce L2 normalization
            h_feat = torch.nn.functional.normalize(h_feat, p=2, dim=1)
            
            feat_list.append(h_feat.cpu())
            
            # Record absolute indices in the original dataset
            batch_indices = torch.arange(global_idx, global_idx + img.size(0))[mask]
            idx_list.append(batch_indices)
            
        global_idx += img.size(0)
        
    return torch.cat(feat_list, dim=0), torch.cat(idx_list, dim=0)


def ddm_selection(args, model, loader):
    """
    Executes the DDM (Deep Distance Metric) sample selection strategy.
    It uses PCA dimensionality reduction and Mahalanobis distance scoring.
    
    Args:
        args (argparse.Namespace): Parsed command-line arguments.
        model (torch.nn.Module): The initialized model.
        loader (torch.utils.data.DataLoader): DataLoader for the dataset.
    """
    clean_feats, idxs = extract_target_features(args, model, loader)
    clean_feats = clean_feats.numpy()
    
    N, D = clean_feats.shape
    print(f"[DDM] Feature extraction complete: {N} samples, {D} dimensions")

    scaler = StandardScaler()
    clean_feats_processed = scaler.fit_transform(clean_feats)

    use_pca = (args.dset == 'GTSRB') or (N < D)
    if use_pca:
        n_components = min(N - 1, 256, D) 
        print(f"[DDM] Enabling PCA optimization, reducing to {n_components} dims (Whitening enabled)...")
        pca = SklearnPCA(n_components=n_components, whiten=True)
        clean_feats_processed = pca.fit_transform(clean_feats_processed)

    print("[DDM] Calculating Mahalanobis Distance...")
    try:
        cov = EmpiricalCovariance(assume_centered=False)
        cov.fit(clean_feats_processed)
        dists = cov.mahalanobis(clean_feats_processed)
    except Exception as e:
        print(f"[DDM] Warning: EmpiricalCovariance failed ({e}), switching to ShrunkCovariance...")
        cov = ShrunkCovariance(shrinkage=0.1)
        cov.fit(clean_feats_processed)
        dists = cov.mahalanobis(clean_feats_processed)

    scores = torch.tensor(dists).sqrt().cpu() 
    indices = idxs.cpu()
    
    os.makedirs('resources', exist_ok=True)
    save_path = f'resources/{args.strategy}_{args.model_name}_{args.dset}_{args.target}_fa5.pth'
    torch.save({'indices': indices, 'scores': scores}, save_path)
    print(f"[DDM] Scoring complete. Saved to: {save_path}")

def prm_selection(args, model, loader):
    """
    Executes the PRM (PCA Reconstruction Metric) sample selection strategy.
    Scores samples based on PCA reconstruction errors with adaptive rank estimation.
    
    Args:
        args (argparse.Namespace): Parsed command-line arguments.
        model (torch.nn.Module): The initialized model.
        loader (torch.utils.data.DataLoader): DataLoader for the dataset.
    """
    clean_feats, idxs = extract_target_features(args, model, loader)
    N, feature_dim = clean_feats.shape

    print(f"[PRM] Calculating PCA Reconstruction Error...")

    mean = clean_feats.mean(dim=0, keepdim=True)
    centered_feats = clean_feats - mean               

    q_max = min(128, feature_dim)
    U, S, V = torch.pca_lowrank(centered_feats, q=q_max, center=False) 

    var = S ** 2
    cum_ratio = var.cumsum(dim=0) / (var.sum() + 1e-6)

    target_ratio = 0.95 if args.model_name == 'vicreg' else 0.90  
    
    valid_indices = (cum_ratio >= target_ratio).nonzero()
    q_eff = int(valid_indices[0]) + 1 if len(valid_indices) > 0 else q_max
    q_eff = max(16, min(q_eff, q_max))

    V_q = V[:, :q_eff]                    
    proj_feats = centered_feats @ V_q     
    recon_feats = proj_feats @ V_q.T      

    errors = (centered_feats - recon_feats).norm(p=2, dim=1)   

    if args.model_name != 'vicreg':
        scores = errors.cpu()
    else:
        radial = proj_feats.norm(p=2, dim=1)  
        eps = 1e-6
        z_err = (errors - errors.mean()) / (errors.std() + eps)
        z_rad = (radial - radial.mean()) / (radial.std() + eps)

        gamma = 0.2   
        final_scores = z_err - gamma * torch.abs(z_rad)
        scores = final_scores.cpu()

    indices = idxs.cpu()

    os.makedirs('resources', exist_ok=True)
    save_path = f'resources/{args.strategy}_{args.model_name}_{args.dset}_{args.target}.pth'
    torch.save({'indices': indices, 'scores': scores}, save_path)
    print(f"[PRM] Scoring complete. Saved to: {save_path}")

def main():
    """
    Main execution pipeline: parses arguments, initializes the model and dataset,
    and dispatches execution to the specified data selection strategy.
    """
    args = parse_args()
    model, mean, std = initialize_model(args)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((args.input_size, args.input_size), interpolation=2),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    dataset_train = get_dataset(args, transform)
    loader = DataLoader(dataset_train, batch_size=128, num_workers=8, pin_memory=True, shuffle=False)

    if args.strategy == 'loss_ood':
        loss_ood_selection(args, model, loader)
    elif args.strategy in ['knn', 'mean']:
        pretrained_selection(args, model, loader)
    elif args.strategy == 'ddm':
        ddm_selection(args, model, loader)
    elif args.strategy == 'prm':
        prm_selection(args, model, loader)
    else:
        raise ValueError(f"Unsupported strategy: {args.strategy}")

if __name__ == "__main__":
    main()