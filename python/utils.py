import io
import time
from collections import defaultdict, deque
import datetime

import torch
import torch.distributed as dist
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from poisoned_datasets import CIFAR10_DEFAULT_MEAN, CIFAR10_DEFAULT_STD

data_loaders_names = {
            'Brightness': 'brightness',
            'Contrast': 'contrast',
            'Defocus Blur': 'defocus_blur',
            'Elastic Transform': 'elastic_transform',
            'Fog': 'fog',
            'Frost': 'frost',
            'Gaussian Noise': 'gaussian_noise',
            'Glass Blur': 'glass_blur',
            'Impulse Noise': 'impulse_noise',
            'JPEG Compression': 'jpeg_compression',
            'Motion Blur': 'motion_blur',
            'Pixelate': 'pixelate',
            'Shot Noise': 'shot_noise',
            'Snow': 'snow',
            'Zoom Blur': 'zoom_blur'
        }

def get_ce_alexnet():
    """Return a dict mapping corruption name to AlexNet corruption error (CE) rate.

    Returns:
        dict: Corruption name -> error rate in [0,1].
    """

    ce_alexnet = dict()
    ce_alexnet['Gaussian Noise'] = 0.886428
    ce_alexnet['Shot Noise'] = 0.894468
    ce_alexnet['Impulse Noise'] = 0.922640
    ce_alexnet['Defocus Blur'] = 0.819880
    ce_alexnet['Glass Blur'] = 0.826268
    ce_alexnet['Motion Blur'] = 0.785948
    ce_alexnet['Zoom Blur'] = 0.798360
    ce_alexnet['Snow'] = 0.866816
    ce_alexnet['Frost'] = 0.826572
    ce_alexnet['Fog'] = 0.819324
    ce_alexnet['Brightness'] = 0.564592
    ce_alexnet['Contrast'] = 0.853204
    ce_alexnet['Elastic Transform'] = 0.646056
    ce_alexnet['Pixelate'] = 0.717840
    ce_alexnet['JPEG Compression'] = 0.606500

    return ce_alexnet

def get_mce_from_accuracy(accuracy, error_alexnet):
    """Convert accuracy to mean corruption error (mCE) normalized by AlexNet baseline.

    Args:
        accuracy (float): Model accuracy on a corruption (percent).
        error_alexnet (float): AlexNet error for that corruption.

    Returns:
        float: Normalized corruption error.
    """
    error = 100. - accuracy
    ce = error / (error_alexnet * 100.)

    return ce

class SmoothedValue(object):
    def __init__(self, window_size=20, fmt=None):
        """Rolling-window metric tracker with optional formatted string.

        Args:
            window_size (int): Max deque length for local stats.
            fmt (str, optional): Format string for ``__str__``; default shows median and global avg.
        """
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        """Add ``value`` with weight ``n`` to the deque and running totals.

        Args:
            value (float): Observed value.
            n (int): Sample count weight.
        """
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """All-reduce ``count`` and ``total`` across ranks (deque not synced)."""
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        """Median of values in the deque."""
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        """Mean of deque values."""
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        """Running mean over all weighted updates."""
        return self.total / self.count

    @property
    def max(self):
        """Max value in deque."""
        return max(self.deque)

    @property
    def value(self):
        """Last value in deque."""
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        """Container of named ``SmoothedValue`` meters.

        Args:
            delimiter (str): Joiner for ``__str__``.
        """
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        """Update meters from scalar kwargs.

        Args:
            **kwargs: Name -> float or 0-dim tensor.
        """
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        """Sync all meters across processes."""
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        """Register a custom meter under ``name``.

        Args:
            name (str): Key.
            meter (SmoothedValue): Meter instance.
        """
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        """Iterate ``iterable`` while logging timing and meters periodically.

        Args:
            iterable: Data iterator.
            print_freq (int): Print every ``print_freq`` steps.
            header (str, optional): Log prefix.

        Yields:
            Items from ``iterable``.
        """
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB), flush=True)
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)), flush=True)
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)), flush=True)


def _load_checkpoint_for_ema(model_ema, checkpoint):
    """Load EMA weights from serialized checkpoint via in-memory buffer.

    Args:
        model_ema: ``timm`` ModelEma instance.
        checkpoint: State dict or checkpoint object accepted by ``torch.save``.
    """
    mem_file = io.BytesIO()
    torch.save(checkpoint, mem_file)
    mem_file.seek(0)
    model_ema._load_checkpoint(mem_file)


def setup_for_distributed(is_master):
    """Patch ``builtins.print`` to no-op on non-master ranks unless ``force=True``.

    Args:
        is_master (bool): Whether this process prints by default.
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    """Return True if distributed package is available and process group is initialized."""
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    """World size of distributed job, or 1 if not initialized."""
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    """Current process rank, or 0 if not distributed."""
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    """True if rank 0."""
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    """``torch.save`` only on rank 0."""
    if is_main_process():
        torch.save(*args, **kwargs)

class Normalize:
    def __init__(self, n_channels, expected_values, variance):
        """Per-channel normalization: ``(x - mean) / std``.

        Args:
            n_channels (int): Number of channels.
            expected_values (sequence): Means per channel.
            variance (sequence): Std per channel.
        """
        self.n_channels = n_channels
        self.expected_values = expected_values
        self.variance = variance
        assert self.n_channels == len(self.expected_values)

    def __call__(self, x):
        """Normalize tensor `x` of shape ``[N, C, ...]``."""
        x_clone = x.clone()
        for channel in range(self.n_channels):
            x_clone[:, channel] = (x[:, channel] - self.expected_values[channel]) / self.variance[channel]
        return x_clone


class Denormalize:
    def __init__(self, n_channels, expected_values, variance):
        """Inverse of ``Normalize``: ``x * std + mean``.

        Args:
            n_channels (int): Channel count.
            expected_values (sequence): Means.
            variance (sequence): Stds.
        """
        self.n_channels = n_channels
        self.expected_values = expected_values
        self.variance = variance
        assert self.n_channels == len(self.expected_values)

    def __call__(self, x):
        """Denormalize tensor `x`."""
        x_clone = x.clone()
        for channel in range(self.n_channels):
            x_clone[:, channel] = x[:, channel] * self.variance[channel] + self.expected_values[channel]
        return x_clone  
    
class IMNETNormalizer(Normalize):
    """ImageNet mean/std normalizer."""

    def __init__(self):
        super().__init__(3, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)

class CIFAR10Normalizer(Normalize):
    """CIFAR-10 mean/std normalizer."""

    def __init__(self):
        super().__init__(3, CIFAR10_DEFAULT_MEAN, CIFAR10_DEFAULT_STD)

class IMNETDenormalizer(Denormalize):
    """ImageNet denormalizer."""

    def __init__(self):
        super().__init__(3, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)

class CIFAR10Denormalizer(Denormalize):
    """CIFAR-10 denormalizer."""

    def __init__(self):
        super().__init__(3, CIFAR10_DEFAULT_MEAN, CIFAR10_DEFAULT_STD)
    
