import tarfile
from functools import partial
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
from torch.utils.data import DistributedSampler

from enot.utils.data.csv_annotation_dataset import CsvAnnotationDataset
from enot.utils.data.dataloaders import create_data_loader
from enot.utils.data.dataloaders import get_default_train_transform
from enot.utils.data.dataloaders import get_default_validation_transform

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


create_imagenette_train_transform = partial(get_default_train_transform, mean=_MEAN, std=_STD)
create_imagenette_validation_transform = partial(get_default_validation_transform, mean=_MEAN, std=_STD)


def _create_data_loader_from_csv_annotation(
        csv_annotation_path,
        dataset_transform,
        batch_size,
        num_workers,
        shuffle,
        dist=False,
        **kwargs,
):
    dataset = CsvAnnotationDataset(
        csv_annotation_path,
        transform=dataset_transform,
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
    test_df = pd.concat(
        group.sample(frac=0.5, random_state=random_seed)
        for _, group in validation_df.groupby('label')
    )
    validation_df = validation_df.loc[~validation_df.filepath.isin(test_df.filepath)]

    test_class_sizes = test_df.label.value_counts()
    search_df = pd.concat(
        group.sample(test_class_sizes[label], random_state=random_seed)
        for label, group in train_df.groupby('label')
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
        dataset_root_dir=dataset_root_dir, imagenette_kind=imagenette_kind,
    )
    annotations = create_imagenette_annotation(
        dataset_dir=dataset_dir, project_dir=project_dir, random_seed=random_seed,
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

    return dict(
        pretrain_train_dataloader=pretrain_and_tune_train_dataloader,
        pretrain_validation_dataloader=pretrain_and_tune_validation_dataloader,
        search_train_dataloader=search_train_dataloader,
        search_validation_dataloader=search_validation_dataloader,
        tune_train_dataloader=pretrain_and_tune_train_dataloader,
        tune_validation_dataloader=pretrain_and_tune_validation_dataloader,
    )
