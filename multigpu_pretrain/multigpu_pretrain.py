"""
You should take into account all of the specifics related to distributed training setup (like using DistributedSampler,
metric synchronization, initialization of process groups, locking IO operations to master workers, etc.).

The only exception is:

<<< We synchronize model buffers and gradients in PretrainOptimizer! (As DDP does) >>>

So, you should NOT use DistributedDataParallel in pretrain loop, as we do it's functionality in our code.
For easier distributed package initialization you can use enot.utils.distributed.init_torch function.
"""

import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from tutorial_utils.dataset import create_imagenette_dataloaders
from tutorial_utils.train import WarmupScheduler
from tutorial_utils.train import accuracy

from enot.distributed import extract_data_from_queue
from enot.distributed import get_world_size
from enot.distributed import init_torch
from enot.distributed import is_dist
from enot.distributed import is_local_master
from enot.distributed import is_master
from enot.distributed import sync_model
from enot.logging import prepare_log
from enot.models import SearchSpaceModel
from enot.models.mobilenet import build_mobilenet
from enot.optimize import PretrainOptimizer

HOME_DIR = Path.home() / '.optimization_experiments'
DATASETS_DIR = HOME_DIR / 'datasets'
PROJECT_DIR = HOME_DIR / 'distributed_pretrain'

SEARCH_OPS = [
    'MIB_k=3_t=6',
    'MIB_k=5_t=6',
    'MIB_k=7_t=6',
]

N_EPOCHS = 100
N_WARMUP_EPOCHS = 10
LR = 0.06


def main():
    HOME_DIR.mkdir(exist_ok=True)
    DATASETS_DIR.mkdir(exist_ok=True)
    PROJECT_DIR.mkdir(exist_ok=True)

    prepare_log(PROJECT_DIR / 'experiments' / 'multigpu_example_2x2_config_v1')

    init_torch(cuda_optimize_for_speed=True)
    distributed = is_dist()
    n_workers = get_world_size()

    dataloaders = create_imagenette_dataloaders(
        DATASETS_DIR,
        PROJECT_DIR,
        input_size=(224, 224),
        batch_size=64,
        num_workers=4,  # Each spawned process in single node will use this number of workers.
        dist=distributed,  # Flag to use DistributedSampler to sample different images in different worker processes.
    )

    # Building search space to pretrain.
    model = build_mobilenet(
        search_ops=SEARCH_OPS,
        num_classes=10,
        blocks_out_channels=[24, 32, 64, 96, 160, 320],
        blocks_count=[2, 3, 4, 3, 3, 1],
        blocks_stride=[2, 2, 2, 1, 2, 1],
    )
    search_space = SearchSpaceModel(model).cuda()  # We do not wrap model with DistributedDataParallel.

    # Synchronize search space across workers.
    sync_model(search_space, reduce_parameters=False, reduce_buffers=False)

    # Log the total size of gradients to transfer.
    if is_local_master():
        total_gradient_bytes = sum([x.element_size() * x.nelement() for x in search_space.model_parameters()])
        total_gradient_megabytes = total_gradient_bytes / (1024 * 1024)
        logging.info(f'Gradients to transfer (in megabytes): {total_gradient_megabytes:.3f}Mb')

    train_loader = dataloaders['pretrain_train_dataloader']
    len_train_loader = len(train_loader)

    # Dataloader for master-only validation.
    if is_master():
        dataloaders = create_imagenette_dataloaders(
            DATASETS_DIR,
            PROJECT_DIR,
            input_size=(224, 224),
            batch_size=64,
            num_workers=8,  # More workers for faster validation.
            dist=False,  # We only validate in master process, so this dataloader is master-only.
        )
        validation_loader = dataloaders['pretrain_validation_dataloader']
        validation_len = len(validation_loader)

    # We should use ``search_space.model_parameters()`` parameters in pre-train phase.
    optimizer = SGD(params=search_space.model_parameters(), lr=LR * n_workers, momentum=0.9, weight_decay=1e-4)

    # Wrap regular optimizer with ``PretrainOptimizer``, and use it later.
    pretrain_optimizer = PretrainOptimizer(search_space=search_space, optimizer=optimizer)

    scheduler = CosineAnnealingLR(optimizer, T_max=len_train_loader * N_EPOCHS)
    scheduler = WarmupScheduler(scheduler, warmup_steps=len_train_loader * N_WARMUP_EPOCHS)

    metric_function = accuracy  # Change if you want.
    loss_function = nn.CrossEntropyLoss().cuda()

    if is_local_master():  # Fancy logging output.
        logging.info('')

    for epoch in range(N_EPOCHS):
        # Setting current epoch in DistributedSampler, see it's documentation:
        # https://pytorch.org/docs/stable/data.html, torch.utils.data.distributed.DistributedSampler class.
        if distributed:
            train_loader.sampler.set_epoch(epoch)

        if is_local_master():
            logging.info(f'EPOCH #{epoch}')

        search_space.train()
        train_metrics_acc = {
            'loss': 0.0,
            'accuracy': 0.0,
            'n': 0,
        }

        train_queue = extract_data_from_queue(
            train_loader,
            data_parallel=True,
            use_tqdm=is_local_master(),  # Show tqdm bar only in local master process.
        )

        for num_sample, inputs, labels in train_queue:
            with torch.no_grad():  # Initialize output distribution optimization.
                if not search_space.output_distribution_optimization_enabled:
                    search_space.initialize_output_distribution_optimization(inputs)

            pretrain_optimizer.zero_grad()

            # Executable closure with forward-backward passes.
            def closure():
                pred_labels = search_space(inputs)
                batch_loss = loss_function(pred_labels, labels)
                batch_loss.backward()
                batch_metric = metric_function(pred_labels, labels)

                train_metrics_acc['loss'] += batch_loss.item()
                train_metrics_acc['accuracy'] += batch_metric.item()
                train_metrics_acc['n'] += 1

            pretrain_optimizer.step(closure)  # Performing enot optimizer step, which internally calls closure.
            if scheduler is not None:
                scheduler.step()

        if is_local_master():  # Log training stats in each local master.
            train_loss = train_metrics_acc['loss'] / train_metrics_acc['n']
            train_accuracy = train_metrics_acc['accuracy'] / train_metrics_acc['n']

            if scheduler is not None:
                logging.info(f'lr: {scheduler.get_lr()[0]:.4f}')
            logging.info('Train metrics:')
            logging.info(f'  loss: {train_loss:.4f}')
            logging.info(f'  accuracy: {train_accuracy:.2f}')

        if is_master():  # Validate only in master process.
            search_space.eval()
            validation_loss = 0
            validation_accuracy = 0

            validation_queue = extract_data_from_queue(
                validation_loader,
                data_parallel=True,
                use_tqdm=True,
            )
            for num_sample, inputs, labels in validation_queue:
                search_space.sample_random_arch()

                with torch.no_grad():
                    val_predictions = search_space(inputs)
                    val_batch_loss = loss_function(val_predictions, labels)
                    val_batch_metric = metric_function(val_predictions, labels)

                validation_loss += val_batch_loss.item()
                validation_accuracy += val_batch_metric.item()

            validation_loss /= validation_len
            validation_accuracy /= validation_len

            logging.info('Validation metrics:')
            logging.info(f'  loss: {validation_loss:.4f}')
            logging.info(f'  accuracy: {validation_accuracy:.2f}')

        if is_local_master():  # Fancy logging output.
            logging.info('')


if __name__ == '__main__':
    main()
