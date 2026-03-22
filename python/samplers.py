import torch
import torch.distributed as dist
import math


class RASampler(torch.utils.data.Sampler):
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True):
        """Distributed sampler with repeated augmentation (RA); each augmented view may go to a different rank.

        Args:
            dataset: Indexable dataset.
            num_replicas (int, optional): World size; defaults to ``dist.get_world_size()``.
            rank (int, optional): Process rank; defaults to ``dist.get_rank()``.
            shuffle (bool): Whether to shuffle indices each epoch.
        """
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 3.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        self.num_selected_samples = int(math.floor(len(self.dataset) // 256 * 256 / self.num_replicas))
        self.shuffle = shuffle

    def __iter__(self):
        """Yield per-rank indices for the current epoch (with 3x repetition and trimming)."""
        g = torch.Generator()
        g.manual_seed(self.epoch)
        if self.shuffle:
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        indices = [ele for ele in indices for i in range(3)]
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices[:self.num_selected_samples])

    def __len__(self):
        """Number of samples per epoch for this rank."""
        return self.num_selected_samples

    def set_epoch(self, epoch):
        """Set epoch for deterministic shuffling across processes.

        Args:
            epoch (int): epoch index.
        """
        self.epoch = epoch
