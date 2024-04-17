# pylint: disable=missing-function-docstring
import tarfile
from functools import partial
from pathlib import Path
from typing import Any
from typing import Callable
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torchvision
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import DistributedSampler
from torch.utils.data import Sampler
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.transforms.functional import InterpolationMode

from enot.utils.common import is_floating_tensor
from enot.utils.python_container_parser import apply_to_containers_recursively

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_BASE_URL = 'https://gitlab.expasoft.com/a.yanchenko/enot-public-data/-/raw/main/tutorials'
_DATASET_BASE_URL = f'{_BASE_URL}/datasets'
IMAGENET_10K_URL = f'{_DATASET_BASE_URL}/imagenet10k.tar.gz'

AnyImage = Union[np.ndarray, Image.Image, torch.Tensor]
TCollateFunction = Callable[[Tuple[List[AnyImage], List[Any]]], Tuple[torch.Tensor, torch.Tensor]]


class CsvAnnotationDataset(Dataset):
    r"""
    Creates dataset from csv file.

    Read CSV annotation with fields 'filepath': str, 'label': int

    """

    def __init__(
        self,
        csv_annotation_path: Union[str, Path],
        root_dir: Optional[Union[str, Path]] = None,
        transform: Optional[torchvision.transforms.Compose] = None,
    ):
        r"""
        Init vision dataset from CSV-file.

        Parameters
        ----------
        csv_annotation_path : Union[str, Path]
            Path to csv-file with  ['filepath', 'label'] columns.
        root_dir : Union[str, Path]
            Optional absolute path to folder with images which prepends to 'filepath' field in csv.
            Useful for handle with different dataset locations.
        transform : torchvision.transforms.Compose
            Transformation should be applied to image.

        """
        if root_dir is not None:
            root_dir = Path(root_dir).resolve()

        csv_annotation_path = Path(csv_annotation_path)
        csv_annotation_path = csv_annotation_path.as_posix()

        self._annotations_data = pd.read_csv(csv_annotation_path, dtype={'filepath': str, 'label': int})
        self._root_dir = root_dir
        self._transform = transform

    def __len__(self) -> int:
        return len(self._annotations_data)

    def __getitem__(self, idx):
        if not isinstance(idx, int):
            raise NotImplementedError('Slicing is not implemented')

        path, label = self._annotations_data.iloc[idx]
        if self._root_dir is not None:
            path = self._root_dir / path

        image = Image.open(path).convert('RGB')
        if self._transform is not None:
            image = self._transform(image)

        return image, label


def recursive_to(
    item: Any,
    device: Union[int, str, torch.device],
    dtype: Optional[torch.dtype] = None,
    ignore_non_tensors: bool = True,
) -> Any:
    """
    Sends nested dicts, OrderedDicts, sets, frozensets, lists and tuples with tensors to the specified device; can
    additionally change tensor data types.

    Parameters
    ----------
    item : Any
        Some data storage with tensors to move.
    device : int, str or torch.device
        Device to send tensors to.
    dtype : torch.dtype or None, optional
        Floating point type to send tensors to. Default value is None, which disables dtype changes.
    ignore_non_tensors : bool, optional
        Whether to raise TypeError if item type is unknown. Default value is True.

    Returns
    -------
    item : Any
        Item with tensor data sent to the specified device and dtype.

    Raises
    ------
    TypeError
        If `ignore_non_tensors` is False, and unknown data type was found.

    """
    device_ = torch.device(device)

    def to_fn(_item: Any) -> Any:
        if isinstance(_item, torch.Tensor):
            item_dtype = _item.dtype
            if dtype is not None and is_floating_tensor(_item):
                item_dtype = dtype

            if device_.type == 'cuda':
                return _item.cuda(device, non_blocking=True).to(dtype=item_dtype)

            return _item.to(device=device_, dtype=item_dtype)

        if not ignore_non_tensors:
            raise TypeError(f'Unexpected item of type {_item.__class__.__name__}')

        return _item

    return apply_to_containers_recursively(item, to_fn, apply_to_self=True)


class CudaDataLoader(DataLoader):
    """DataLoader which copies data to GPU."""

    def __init__(self, *args, dtype: Optional[torch.dtype] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.dtype = dtype

    def __iter__(self):
        iterator = super().__iter__()
        for data in iterator:
            yield recursive_to(data, 'cuda', dtype=self.dtype, ignore_non_tensors=True)


def get_default_train_transform(
    input_size,
    mean,
    std,
) -> transforms.Compose:
    """
    Returns common train augmentation.

    Augments images via RandomCrop and flip, also transforms it into normalized tensor.

    Parameters
    ----------
    input_size : Union[List[int, int], Tuple[int, int], int]
        Input image size.
    mean : Tuple[float, float, float]
        Per channel mean for all dataset.
    std : Tuple[float, float, float]
        Per channel std for all dataset.

    Returns
    -------
    transforms.Compose
        All composed transformations.

    """
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std, inplace=True),
            transforms.RandomErasing(value='random'),  # type: ignore
        ]
    )


def get_default_validation_transform(input_size, mean, std):
    """
    Returns common validation transformations.

    No train augmentation presented, only Resize and normalization.

    Parameters
    ----------
    input_size : Union[List[int, int], Tuple[int, int], int]
        Input image size.
    mean : Tuple[float, float, float]
        Per channel mean for all dataset.
    std : Tuple[float, float, float]
        Per channel std for all dataset.

    Returns
    -------
    transforms.Compose
        All composed transformations.

    """
    return transforms.Compose(
        [transforms.Resize(input_size), transforms.ToTensor(), transforms.Normalize(mean, std, inplace=True)]
    )


def fast_collate(
    batch: Union[Tuple[List[AnyImage], List[Any]], List],
    channels: int = 3,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Creates batch of numpy arrays, torch.Tensors or PIL.Images.
    Compatible with torch.utils.data.DataLoader collate_fn.

    Parameters
    ----------
    batch : tuple with AnyImage and list of labels
        Tuple with batch of data and batch of labels.
    channels : int
        Number of channels in data tensors.

    Returns
    -------
    image_tensor : Tensor
        Tensor with image data of shape (batch_size, n_channels, height,
        width), and with original image dtype.
    label_tensor : Tensor
        Tensor with labels. Labels are stored in int64 format.

    """
    images = [pair[0] for pair in batch]
    targets = torch.tensor([pair[1] for pair in batch], dtype=torch.int64)

    some_image = images[0]
    if isinstance(some_image, Image.Image):
        width, height = some_image.size
        dtype = np.uint8
        images = (_numpy_image_to_torch(np.asarray(image, dtype)) for image in images)
    elif isinstance(some_image, np.ndarray):
        height, width = some_image.shape[:2]
        dtype = some_image.dtype
        images = map(_numpy_image_to_torch, images)
    elif isinstance(some_image, torch.Tensor):
        height, width = some_image.shape[1:]
        dtype = some_image.dtype
    else:
        raise TypeError('Unsupported image type')

    tensor = torch.zeros((len(batch), channels, height, width), dtype=dtype)
    for i, image in enumerate(images):
        tensor[i, ...] = image

    return tensor, targets


def create_data_loader(
    dataset: Dataset,
    batch_size: int,
    *,
    check_sampler: bool = True,
    sampler: Optional[Sampler] = None,
    collate_fn: TCollateFunction = fast_collate,
    **kwargs,
) -> DataLoader:
    """
    Creates dataloader for certain dataset.

    Creates dataloader each sample of one will be placed automatically to CUDA.

    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        Data.
    batch_size : int
        How much samples should be processing at once.
    check_sampler : bool
        Should we check sample object. Important for multi-GPU procedure.
    sampler : torch.utils.data.Sampler
        Way to sample object from dataset.
    collate_fn : TCollateFunction
        Custom collate function.
    kwargs
        Additional parameters propagated to dataloader.

    Returns
    -------
    torch.utils.data.DataLoader
        Custom dataloader.

    """
    if check_sampler and dist.is_initialized():
        if sampler is None:
            shuffle = kwargs.pop('shuffle', False)
            kwargs['shuffle'] = False
            sampler = DistributedSampler(dataset, shuffle=shuffle)
        elif not isinstance(sampler, DistributedSampler):
            raise TypeError('Multi GPU setup is detected, so sampler must be subclass of DistributedSampler')

    return CudaDataLoader(
        dataset=dataset,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=collate_fn,
        **kwargs,
    )


def create_data_loader_from_csv_annotation(
    csv_annotation_path: Union[str, Path],
    dataset_root_dir: Optional[Union[str, Path]] = None,
    dataset_transform: Optional[transforms.Compose] = None,
    **kwargs,
) -> DataLoader:
    """
    Creates dataloader from csv-file.

    Parameters
    ----------
    csv_annotation_path : Union[str, Path]
        Path to csv-file with  ['filepath', 'label'] columns.
    dataset_root_dir : Union[str, Path]
        Optional absolute path to folder with images which prepends to 'filepath' field in csv.
        Useful for handle with different dataset locations.
    dataset_transform : torchvision.transforms.Compose
        Transformation should be applied to image.
    kwargs
        Additional parameters propagated to dataloader.

    Returns
    -------
    torch.utils.data.DataLoader
        Dataloader from csv file.

    """
    dataset = CsvAnnotationDataset(csv_annotation_path, root_dir=dataset_root_dir, transform=dataset_transform)
    return create_data_loader(dataset, **kwargs)


create_imagenette_train_transform = partial(get_default_train_transform, mean=_MEAN, std=_STD)
create_imagenette_validation_transform = partial(get_default_validation_transform, mean=_MEAN, std=_STD)


def imagenet_train_transform(
    input_size,
    mean=_MEAN,
    std=_STD,
    interpolation=InterpolationMode.BILINEAR,
):
    del interpolation
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std, inplace=True),
        ]
    )


def imagenet_valid_transform(
    input_size,
    mean=_MEAN,
    std=_STD,
    interpolation=InterpolationMode.BILINEAR,
):
    return transforms.Compose(
        [
            transforms.Resize(int(input_size / 0.875), interpolation=interpolation),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std, inplace=True),
        ]
    )


def _create_data_loader_from_csv_annotation(
    csv_annotation_path,
    dataset_transform,
    batch_size,
    num_workers,
    shuffle,
    dist=False,
    root_dir=None,
    **kwargs,
):
    dataset = CsvAnnotationDataset(
        csv_annotation_path,
        transform=dataset_transform,
        root_dir=root_dir,
    )
    if dist:
        # In distributed setup we need to use DistributedSampler in dataloader
        dataloader = create_data_loader(
            dataset=dataset,
            batch_size=batch_size,
            sampler=DistributedSampler(
                dataset=dataset,
                shuffle=shuffle,
            ),
            num_workers=num_workers,
            **kwargs,
        )
    else:
        dataloader = create_data_loader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            **kwargs,
        )
    return dataloader


# Collect paths grouped by labels
def _collect_paths(split_dir, label_map):
    annotation = []
    for folder in split_dir.iterdir():
        label = label_map[folder.name]
        for file_path in folder.iterdir():
            annotation.append((file_path.as_posix(), label))

    return pd.DataFrame(annotation, columns=['filepath', 'label'])


def download_imagenette(dataset_root_dir, imagenette_kind):
    dataset_root_dir = Path(dataset_root_dir)
    dataset_dir = dataset_root_dir / imagenette_kind

    if dataset_dir.exists():
        return dataset_dir

    url = f'https://s3.amazonaws.com/fast-ai-imageclas/{imagenette_kind}.tgz'
    file_path = dataset_root_dir / f'{imagenette_kind}.tgz'
    try:
        # download dataset
        urlretrieve(url=url, filename=file_path)
        # unpack archive
        with tarfile.open(file_path) as dataset_archive:
            dataset_archive.extractall(dataset_root_dir)
    finally:
        # remove archive file
        if file_path.exists():
            file_path.unlink()

    return dataset_dir


def download_imagenet10k(dataset_root_dir):
    dataset_root_dir = Path(dataset_root_dir)
    dataset_dir = dataset_root_dir / 'imagenet10k'

    if dataset_dir.exists():
        return dataset_dir

    file_path = dataset_root_dir / 'imagenet10k.tgz'
    try:
        # download dataset
        urlretrieve(url=IMAGENET_10K_URL, filename=file_path.as_posix())
        # unpack archive
        with tarfile.open(file_path) as dataset_archive:
            dataset_archive.extractall(dataset_root_dir)
    finally:
        # remove archive file
        if file_path.exists():
            file_path.unlink()

    return dataset_dir


def create_imagenette_annotation(dataset_dir, project_dir, random_seed=42):
    dataset_dir, project_dir = Path(dataset_dir), Path(project_dir)

    train_dir = dataset_dir / 'train'
    val_dir = dataset_dir / 'val'

    # Collect labels represented as directory names and map to ints
    dir_names = {x.name for x in train_dir.iterdir()}
    dir_names |= {x.name for x in val_dir.iterdir()}
    name_to_int = {name: i for i, name in enumerate(sorted(dir_names))}

    train_df = _collect_paths(train_dir, name_to_int)
    validation_df = _collect_paths(val_dir, name_to_int)

    # Make the final splits. In this example val, optim, and test splits are the same size
    test_df = pd.concat(group.sample(frac=0.5, random_state=random_seed) for _, group in validation_df.groupby('label'))
    validation_df = validation_df.loc[~validation_df.filepath.isin(test_df.filepath)]

    test_class_sizes = test_df.label.value_counts()
    search_df = pd.concat(
        group.sample(test_class_sizes[label], random_state=random_seed) for label, group in train_df.groupby('label')
    )
    train_df = train_df.loc[~train_df.filepath.isin(search_df.filepath)]

    train_path = project_dir / 'train.csv'
    test_path = project_dir / 'test.csv'
    validation_path = project_dir / 'validation.csv'
    search_path = project_dir / 'search.csv'

    # Save the annotations
    train_df.to_csv(train_path, index=False)
    validation_df.to_csv(validation_path, index=False)
    search_df.to_csv(search_path, index=False)
    test_df.to_csv(test_path, index=False)

    return {
        'train': train_path,
        'validation': validation_path,
        'search': search_path,
        'test': test_path,
    }


def create_imagenette_dataloaders(
    dataset_root_dir,
    project_dir,
    input_size,
    batch_size,
    num_workers=4,
    imagenette_kind='imagenette2',
    random_seed=42,
    dist=False,
):
    dataset_dir = download_imagenette(
        dataset_root_dir=dataset_root_dir,
        imagenette_kind=imagenette_kind,
    )
    annotations = create_imagenette_annotation(
        dataset_dir=dataset_dir,
        project_dir=project_dir,
        random_seed=random_seed,
    )

    train_transform = create_imagenette_train_transform(input_size)
    validation_transform = create_imagenette_validation_transform(input_size)

    pretrain_and_tune_train_dataloader = _create_data_loader_from_csv_annotation(
        csv_annotation_path=annotations['train'],
        dataset_transform=train_transform,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        dist=dist,
    )
    pretrain_and_tune_validation_dataloader = _create_data_loader_from_csv_annotation(
        csv_annotation_path=annotations['validation'],
        dataset_transform=validation_transform,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        dist=dist,
    )

    search_train_dataloader = _create_data_loader_from_csv_annotation(
        csv_annotation_path=annotations['search'],
        dataset_transform=train_transform,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        dist=dist,
    )
    search_validation_dataloader = _create_data_loader_from_csv_annotation(
        csv_annotation_path=annotations['search'],
        dataset_transform=validation_transform,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        dist=dist,
    )

    return {
        'pretrain_train_dataloader': pretrain_and_tune_train_dataloader,
        'pretrain_validation_dataloader': pretrain_and_tune_validation_dataloader,
        'search_train_dataloader': search_train_dataloader,
        'search_validation_dataloader': search_validation_dataloader,
        'tune_train_dataloader': pretrain_and_tune_train_dataloader,
        'tune_validation_dataloader': pretrain_and_tune_validation_dataloader,
    }


def create_imagenette_dataloaders_for_pruning(
    dataset_root_dir,
    project_dir,
    input_size,
    batch_size,
    num_workers=4,
    imagenette_kind='imagenette2',
    random_seed=42,
    dist=False,
):
    dataset_dir = download_imagenette(
        dataset_root_dir=dataset_root_dir,
        imagenette_kind=imagenette_kind,
    )
    annotations = create_imagenette_annotation(
        dataset_dir=dataset_dir,
        project_dir=project_dir,
        random_seed=random_seed,
    )

    train_transform = imagenet_train_transform(input_size)
    validation_transform = imagenet_valid_transform(input_size)

    train_dataloader = _create_data_loader_from_csv_annotation(
        csv_annotation_path=annotations['train'],
        dataset_transform=train_transform,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        drop_last=True,
        dist=dist,
    )
    validation_dataloader = _create_data_loader_from_csv_annotation(
        csv_annotation_path=annotations['validation'],
        dataset_transform=validation_transform,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        drop_last=False,
        dist=dist,
    )

    return train_dataloader, validation_dataloader


def create_imagenet10k_dataloaders(
    dataset_root_dir,
    input_size,
    batch_size,
    num_workers=4,
):
    dataset_dir = download_imagenet10k(dataset_root_dir=dataset_root_dir)

    transform = imagenet_valid_transform(input_size)
    train_dataset = ImageFolder(str(dataset_dir / 'train'), transform=transform)
    validation_dataset = ImageFolder(str(dataset_dir / 'val'), transform=transform)

    train_dataloader = create_data_loader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )
    validation_dataloader = create_data_loader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )

    return train_dataloader, validation_dataloader


def _numpy_image_to_torch(image: np.ndarray) -> torch.Tensor:
    if not 2 <= image.ndim <= 3:
        raise ValueError('Image ndim must be 2 or 3')
    if image.ndim == 2:
        image = np.expand_dims(image, -1)
    image = np.rollaxis(image, 2, 0)
    return torch.from_numpy(image)
