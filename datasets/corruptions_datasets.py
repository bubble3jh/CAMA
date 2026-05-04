
import os
import json
import torch
import numpy as np
import logging
from typing import Optional, Sequence

from robustbench.data import CORRUPTIONS, PREPROCESSINGS, load_cifar10c, load_cifar100c
from robustbench.loaders import CustomImageFolder, CustomCifarDataset

logger = logging.getLogger(__name__)


def _resolve_lt_class_order(K: int, class_order: str = "index", order_seed: int = 0):
    """Return class ids ordered from long-tail head to tail."""
    spec = (class_order or "index").strip()
    if spec == "index":
        return list(range(K))
    if spec == "reverse":
        return list(range(K - 1, -1, -1))
    if spec == "random":
        rng = np.random.default_rng(order_seed)
        return rng.permutation(K).tolist()
    if spec.startswith("file:"):
        path = spec.split(":", 1)[1]
        with open(path, "r") as f:
            vals = [int(x) for x in f.read().replace(",", " ").split()]
        if sorted(vals) != list(range(K)):
            raise ValueError(f"invalid LT class-order file for K={K}: {path}")
        return vals
    raise ValueError(f"unknown LT class order: {class_order}")


def create_cifarc_dataset(
    dataset_name: str = 'cifar10_c',
    severity: int = 5,
    data_dir: str = './data',
    corruption: str = "gaussian_noise",
    corruptions_seq: Sequence[str] = CORRUPTIONS,
    transform=None,
    setting: str = 'continual'):

    domain = []
    x_test = torch.tensor([])
    y_test = torch.tensor([])
    corruptions_seq = corruptions_seq if "mixed_domains" in setting else [corruption]

    for cor in corruptions_seq:
        if dataset_name == 'cifar10_c':
            x_tmp, y_tmp = load_cifar10c(severity=severity,
                                         data_dir=data_dir,
                                         corruptions=[cor])
        elif dataset_name == 'cifar100_c':
            x_tmp, y_tmp = load_cifar100c(severity=severity,
                                          data_dir=data_dir,
                                          corruptions=[cor])
        else:
            raise ValueError(f"Dataset {dataset_name} is not suported!")

        x_test = torch.cat([x_test, x_tmp], dim=0)
        y_test = torch.cat([y_test, y_tmp], dim=0)
        domain += [cor] * x_tmp.shape[0]

    x_test = x_test.numpy().transpose((0, 2, 3, 1))
    y_test = y_test.numpy()
    samples = [[x_test[i], y_test[i], domain[i]] for i in range(x_test.shape[0])]

    return CustomCifarDataset(samples=samples, transform=transform)


def create_cifarc_lt_dataset(
    dataset_name: str = 'cifar10_c_lt',
    severity: int = 5,
    data_dir: str = './data',
    corruption: str = "gaussian_noise",
    corruptions_seq: Sequence[str] = CORRUPTIONS,
    transform=None,
    setting: str = 'continual',
    imbalance_factor: float = 100.0,
    lt_seed: int = 0,
    class_order: str = "index",
    order_seed: int = 0):
    """Long-tail variant of CIFAR-10/100-C.

    Classes are ordered from head to tail by `class_order`. At rank r, the class
    keeps n_r = n_max * (1/imbalance_factor)^(r/(K-1)) samples. The default
    order is numeric class index, matching the original implementation.

    Returns CustomCifarDataset with subsampled `samples`. The class distribution
    of returned samples is the long-tail distribution; for KL(pi||pi_true) the
    caller can recompute from `dataset.samples` labels.
    """
    base_name = dataset_name.replace("_lt", "")
    base_ds = create_cifarc_dataset(
        dataset_name=base_name, severity=severity, data_dir=data_dir,
        corruption=corruption, corruptions_seq=corruptions_seq,
        transform=transform, setting=setting)

    samples = base_ds.samples
    K = 10 if base_name == "cifar10_c" else 100

    class_indices = {k: [] for k in range(K)}
    for idx, sample in enumerate(samples):
        y = int(sample[1])
        class_indices[y].append(idx)

    n_max = max(len(class_indices[k]) for k in range(K)) if K > 1 else 1

    rng = np.random.default_rng(lt_seed)
    selected = []
    order = _resolve_lt_class_order(K, class_order, order_seed)
    for rank, k in enumerate(order):
        ratio = (1.0 / imbalance_factor) ** (rank / max(K - 1, 1))
        target = max(1, int(round(n_max * ratio)))
        avail = class_indices[k]
        n = min(target, len(avail))
        if n == 0:
            continue
        chosen = rng.choice(avail, size=n, replace=False)
        selected.extend(chosen.tolist())

    rng.shuffle(selected)
    base_ds.samples = [samples[i] for i in selected]
    logger.info(
        f"create_cifarc_lt_dataset: {dataset_name} corruption={corruption} "
        f"IF={imbalance_factor} seed={lt_seed} order={class_order} "
        f"order_seed={order_seed} -> {len(base_ds.samples)} samples "
        f"(head_class={order[0]} count={sum(1 for s in base_ds.samples if int(s[1])==order[0])}, "
        f"tail_class={order[-1]} count={sum(1 for s in base_ds.samples if int(s[1])==order[-1])})")
    return base_ds


def create_imagenetc_dataset(
    n_examples: Optional[int] = -1,
    severity: int = 5,
    data_dir: str = './data',
    corruption: str = "gaussian_noise",
    corruptions_seq: Sequence[str] = CORRUPTIONS,
    transform=None,
    setting: str = 'continual'):

    # create the dataset which loads the default test list from robust bench containing 5000 test samples
    corruptions_seq = corruptions_seq if "mixed_domains" in setting else [corruption]
    corruption_dir_path = os.path.join(data_dir, corruptions_seq[0], str(severity))
    dataset_test = CustomImageFolder(corruption_dir_path, transform)

    if "mixed_domains" in setting or "correlated" in setting or n_examples != -1:
        # load imagenet class to id mapping from robustbench
        with open(os.path.join("robustbench", "data", "imagenet_class_to_id_map.json"), 'r') as f:
            class_to_idx = json.load(f)

        if n_examples != -1 or "correlated" in setting:
            # create file path of file containing all 50k image ids
            file_path = os.path.join("datasets", "imagenet_list", "imagenet_val_ids_50k.txt")
        else:
            # create file path of default test list from robustbench
            file_path = os.path.join("robustbench", "data", "imagenet_test_image_ids.txt")

        # load file containing file ids
        with open(file_path, 'r') as f:
            fnames = f.readlines()

        item_list = []
        for cor in corruptions_seq:
            corruption_dir_path = os.path.join(data_dir, cor, str(severity))
            item_list += [(os.path.join(corruption_dir_path, fn.split('\n')[0]), class_to_idx[fn.split(os.sep)[0]]) for fn in fnames]
        dataset_test.samples = item_list

    return dataset_test
