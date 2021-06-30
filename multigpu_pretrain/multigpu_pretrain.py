"""
You should take into account all of the specifics related to distributed training setup (like using DistributedSampler,
 metric synchronization, initialization of process groups, locking IO operations to master worker).
The only exception is:

<<< We synchronize model buffers and gradients in EnotPretrainOptimizer! >>>

So, you should NOT use DistributedDataParallel in pretrain loop, as we do it's functionality in our code.
For easier distributed package initialization you can use enot.utils.distributed.init_torch function.
"""

from argparse import ArgumentParser
from pathlib import Path
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn as nn

from enot.utils.distributed import torch_save
from enot.utils.train import create_checkpoint
from enot.models import SearchSpaceModel
from enot.models.mobilenet import build_mobilenet
from enot.utils.distributed import init_torch
from enot.utils.distributed import extract_data_from_queue
from enot.utils.distributed import is_master
from enot.utils.distributed import sync_model
from enot.optimize import EnotPretrainOptimizer

from tutorial_utils.train import accuracy
from tutorial_utils.dataset import create_imagenette_dataloaders
from tutorial_utils.train import WarmupScheduler


def main(args):
    ENOT_HOME_DIR = Path.home() / '.enot'
    ENOT_DATASETS_DIR = ENOT_HOME_DIR / 'datasets'
    PROJECT_DIR = ENOT_HOME_DIR / 'distributed_pretrain'

    ENOT_HOME_DIR.mkdir(exist_ok=True)
    ENOT_DATASETS_DIR.mkdir(exist_ok=True)
    PROJECT_DIR.mkdir(exist_ok=True)

    init_torch(local_rank=args.local_rank, seed=args.seed)

    dataloaders = create_imagenette_dataloaders(
        ENOT_DATASETS_DIR,
        PROJECT_DIR,
        input_size=(224, 224),
        batch_size=32,
        dist=True,  # Need to create distributed dataloaders (use distributed sampler from torch.utils.data)
    )

    SEARCH_OPS = [
        'MIB_k=3_t=6',
        'MIB_k=5_t=6',
        'MIB_k=7_t=6',
    ]

    # build model
    model = build_mobilenet(
        search_ops=SEARCH_OPS,
        num_classes=10,
        blocks_out_channels=[24, 32, 64, 96, 160, 320],
        blocks_count=[2, 2, 2, 1, 2, 1],
        blocks_stride=[2, 2, 2, 1, 2, 1],
    )
    search_space = SearchSpaceModel(model)  # no DDP
    search_space.cuda()
    sync_model(search_space)

    N_EPOCHS = 3
    N_WARMUP_EPOCHS = 1

    train_loader = dataloaders['pretrain_train_dataloader']
    len_train_loader = len(train_loader)
    validation_loader = dataloaders['pretrain_validation_dataloader']
    validation_len = len(validation_loader)

    # using `search_space.model_parameters()` as optimizable variables
    optimizer = SGD(params=search_space.model_parameters(), lr=0.06, momentum=0.9, weight_decay=1e-4)
    # using `EnotPretrainOptimizer` as a default optimizer
    enot_optimizer = EnotPretrainOptimizer(search_space=search_space, optimizer=optimizer)

    scheduler = CosineAnnealingLR(optimizer, T_max=len_train_loader * N_EPOCHS, eta_min=1e-8)
    scheduler = WarmupScheduler(scheduler, warmup_steps=len_train_loader * N_WARMUP_EPOCHS)

    metric_function = accuracy
    loss_function = nn.CrossEntropyLoss().cuda()

    for epoch in range(N_EPOCHS):
        train_loader.sampler.set_epoch(epoch)
        validation_loader.sampler.set_epoch(epoch)

        if is_master():
            print(f'EPOCH #{epoch}')

        search_space.train()
        train_metrics_acc = {
            'loss': 0.0,
            'accuracy': 0.0,
            'n': 0,
        }

        train_queue = extract_data_from_queue(
            train_loader,
            data_parallel=True,
            use_tqdm=is_master(),
        )
        validation_queue = extract_data_from_queue(
            validation_loader,
            data_parallel=True,
            use_tqdm=is_master(),
        )

        for num_sample, inputs, labels in train_queue:
            if not search_space.output_distribution_optimization_enabled:
                search_space.initialize_output_distribution_optimization(inputs)

            enot_optimizer.zero_grad()

            def closure():
                pred_labels = search_space(inputs)
                batch_loss = loss_function(pred_labels, labels)
                batch_loss.backward()
                batch_metric = metric_function(pred_labels, labels)

                train_metrics_acc['loss'] += batch_loss.item()
                train_metrics_acc['accuracy'] += batch_metric.item()
                train_metrics_acc['n'] += 1

            enot_optimizer.step(closure)
            if scheduler is not None:
                scheduler.step()

        train_loss = train_metrics_acc['loss'] / train_metrics_acc['n']
        train_accuracy = train_metrics_acc['accuracy'] / train_metrics_acc['n']

        if is_master():
            print('train metrics:')
            print('  loss:', train_loss)
            print('  accuracy:', train_accuracy)

        search_space.eval()
        validation_loss = 0
        validation_accuracy = 0

        for num_sample, inputs, labels in validation_queue:
            search_space.sample_random_arch()

            pred_labels = search_space(inputs)
            batch_loss = loss_function(pred_labels, labels)
            batch_metric = metric_function(pred_labels, labels)

            validation_loss += batch_loss.item()
            validation_accuracy += batch_metric.item()

        validation_loss /= validation_len
        validation_accuracy /= validation_len

        if is_master():
            print('validation metrics:')
            print('  loss:', validation_loss)
            print('  accuracy:', validation_accuracy)

            print()

        checkpoint = create_checkpoint(epoch, search_space, optimizer)
        torch_save(checkpoint, f'{PROJECT_DIR}/checkpoint-{epoch}.pth')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--local_rank', type=int)
    commandline_arguments = parser.parse_args()

    main(commandline_arguments)
