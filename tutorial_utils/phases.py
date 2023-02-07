from typing import Optional

import numpy as np
import torch

from enot.latency import best_arch_latency
from enot.latency import initialize_latency
from enot.latency import max_latency
from enot.latency import mean_latency
from enot.latency import min_latency
from enot.models import SearchSpaceModel
from enot.optimize import PretrainOptimizer
from enot.optimize import SearchOptimizer


def tutorial_pretrain_loop(
    epochs,
    search_space,
    pretrain_optimizer,
    metric_function,
    loss_function,
    train_loader,
    validation_loader,
    scheduler=None,
):
    if not isinstance(search_space, SearchSpaceModel):
        raise TypeError('search_space must be instance of SearchSpaceModel')
    if not isinstance(pretrain_optimizer, PretrainOptimizer):
        raise TypeError('pretrain_optimizer must be instance of PretrainOptimizer')

    for epoch in range(epochs):
        print(f'EPOCH #{epoch}')

        search_space.train()
        train_metrics_acc = {
            'loss': 0.0,
            'accuracy': 0.0,
            'n': 0,
        }
        for inputs, labels in train_loader:
            if not search_space.output_distribution_optimization_enabled:
                search_space.initialize_output_distribution_optimization(inputs)

            pretrain_optimizer.zero_grad()

            def closure():
                pred_labels = search_space(inputs)
                batch_loss = loss_function(pred_labels, labels)
                batch_loss.backward()
                batch_metric = metric_function(pred_labels, labels)

                train_metrics_acc['loss'] += batch_loss.item()
                train_metrics_acc['accuracy'] += batch_metric.item()
                train_metrics_acc['n'] += 1

            pretrain_optimizer.step(closure)
            if scheduler is not None:
                scheduler.step()

        train_loss = train_metrics_acc['loss'] / train_metrics_acc['n']
        train_accuracy = train_metrics_acc['accuracy'] / train_metrics_acc['n']

        print('train metrics:')
        print('  loss:', train_loss)
        print('  accuracy:', train_accuracy)

        arch_to_test = [0] * len(search_space.search_variants_containers)
        test_model = search_space.get_network_by_indexes(arch_to_test)
        test_model.eval()

        validation_loss = 0
        validation_accuracy = 0
        with torch.no_grad():
            for inputs, labels in validation_loader:
                pred_labels = test_model(inputs)
                batch_loss = loss_function(pred_labels, labels)
                batch_metric = metric_function(pred_labels, labels)

                validation_loss += batch_loss.item()
                validation_accuracy += batch_metric.item()

        n = len(validation_loader)
        validation_loss /= n
        validation_accuracy /= n

        print('validation metrics:')
        print('  loss:', validation_loss)
        print('  accuracy:', validation_accuracy)

        print()


def tutorial_search_loop(
    epochs,
    search_space,
    search_optimizer,
    metric_function,
    loss_function,
    train_loader,
    validation_loader,
    latency_loss_weight,
    latency_type: Optional[str],
    scheduler=None,
):
    if not isinstance(search_space, SearchSpaceModel):
        raise TypeError('search_space must be instance of SearchSpaceModel')
    if not isinstance(search_optimizer, SearchOptimizer):
        raise TypeError('search_optimizer must be instance of SearchOptimizer')

    for epoch in range(epochs):
        print(f'EPOCH #{epoch}')

        search_space.train()
        train_metrics_acc = {
            'loss': 0.0,
            'accuracy': 0.0,
            'n': 0,
        }
        for inputs, labels in train_loader:
            if latency_type and search_space.latency_type is None:
                latency_container = initialize_latency(latency_type, search_space, (inputs,))
                print(f'Constant latency = {latency_container.constant_latency}')
                print(
                    f'Min, mean and max latencies of search space: '
                    f'{min_latency(latency_container)}, '
                    f'{mean_latency(latency_container)}, '
                    f'{max_latency(latency_container)}'
                )

            search_optimizer.zero_grad()

            def closure():
                pred_labels = search_space(inputs)

                batch_loss = loss_function(pred_labels, labels)
                if latency_loss_weight is not None and latency_loss_weight != 0:
                    batch_loss += search_space.loss_latency_expectation * latency_loss_weight

                batch_loss.backward()
                batch_metric = metric_function(pred_labels, labels)

                train_metrics_acc['loss'] += batch_loss.item()
                train_metrics_acc['accuracy'] += batch_metric.item()
                train_metrics_acc['n'] += 1

            search_optimizer.step(closure)
            if scheduler is not None:
                scheduler.step()

        train_loss = train_metrics_acc['loss'] / train_metrics_acc['n']
        train_accuracy = train_metrics_acc['accuracy'] / train_metrics_acc['n']
        arch_probabilities = np.array(search_space.architecture_probabilities)

        print('train metrics:')
        print('  loss:', train_loss)
        print('  accuracy:', train_accuracy)
        print('  arch_probabilities:')
        print(arch_probabilities)

        search_space.eval()
        search_space.sample_best_arch()

        validation_loss = 0
        validation_accuracy = 0
        with torch.no_grad():
            for inputs, labels in validation_loader:
                pred_labels = search_space(inputs)
                batch_loss = loss_function(pred_labels, labels)
                batch_metric = metric_function(pred_labels, labels)

                validation_loss += batch_loss.item()
                validation_accuracy += batch_metric.item()

        n = len(validation_loader)
        validation_loss /= n
        validation_accuracy /= n

        print('validation metrics:')
        print('  loss:', validation_loss)
        print('  accuracy:', validation_accuracy)
        if search_space.latency_type is not None:
            latency = best_arch_latency(search_space)
            print('  latency:', latency)

        print()


def tutorial_train_loop(
    epochs,
    model,
    optimizer,
    metric_function,
    loss_function,
    train_loader,
    validation_loader,
    scheduler=None,
):
    for epoch in range(epochs):
        print(f'EPOCH #{epoch}')

        model.train()
        train_loss = 0.0
        train_accuracy = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()

            pred_labels = model(inputs)
            batch_loss = loss_function(pred_labels, labels)
            batch_loss.backward()
            batch_metric = metric_function(pred_labels, labels)

            train_loss += batch_loss.item()
            train_accuracy += batch_metric.item()

            optimizer.step()
            if scheduler is not None:
                scheduler.step()

        n = len(train_loader)
        train_loss /= n
        train_accuracy /= n

        print('train metrics:')
        print('  loss:', train_loss)
        print('  accuracy:', train_accuracy)

        model.eval()
        validation_loss = 0
        validation_accuracy = 0
        with torch.no_grad():
            for inputs, labels in validation_loader:
                pred_labels = model(inputs)
                batch_loss = loss_function(pred_labels, labels)
                batch_metric = metric_function(pred_labels, labels)

                validation_loss += batch_loss.item()
                validation_accuracy += batch_metric.item()

        n = len(validation_loader)
        validation_loss /= n
        validation_accuracy /= n

        print('validation metrics:')
        print('  loss:', validation_loss)
        print('  accuracy:', validation_accuracy)

        print()
