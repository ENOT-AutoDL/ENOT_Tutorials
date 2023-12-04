from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import torch

Scheduler = torch.optim.lr_scheduler._LRScheduler  # pylint: disable=protected-access


def accuracy(
    prediction: torch.Tensor,
    target: torch.Tensor,
    topk: Tuple[int] = (1,),
) -> Union[torch.Tensor, List[torch.Tensor]]:
    """
    Calculates prediction accuracy.

    Parameters
    ----------
    prediction : Tensor
        Batch with logits or probabilities from model, of shape `(batch_size, num_classes)`.
    target : Tensor
        Batch with target class labels.
    topk : tuple of int's
        Tuple containing top-k variants of accuracy.

    Returns
    -------
    Tensor or list of Tensors
        Accuracy scores for each `topk` value. Returns single tensor if
        `len(topk) == 1`, and tuple with tensors otherwise.

    """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = prediction.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    result = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        result.append(correct_k.mul_(100.0 / batch_size))

    return result[0] if len(result) == 1 else result


class _ApplyAndRestoreLR:
    def __init__(self, optimizer: torch.optim.Optimizer, new_lr: List[float]):
        self.optimizer = optimizer
        self.new_lr = new_lr
        self.saved_lr = [group['lr'] for group in optimizer.param_groups]

    def _apply_lr(self, lr_to_apply: List[float]) -> None:
        for group, learning_rate in zip(self.optimizer.param_groups, lr_to_apply):
            group['lr'] = learning_rate

    def __enter__(self):
        self._apply_lr(self.new_lr)

    def __exit__(self, *args, **kwargs):
        del args, kwargs
        self._apply_lr(self.saved_lr)


class WarmupScheduler(Scheduler):
    """
    Scales original learning rate scheduler values linearly for
    `warmup_steps` iterations.

    Notes
    -----
    This should wrap original scheduler, and replace it in the training
    loop.

    """

    def __init__(
        self,
        scheduler: Scheduler,
        warmup_steps: int = 1000,
    ):
        """
        Parameters
        ----------
        scheduler : Scheduler
            Base scheduler which performs basic learning rate scaling.
        warmup_steps : int, optional
            Number of warmup iterations (number of iterations when
            learning rate scale factor < 1. Default value: 1000.

        """
        if warmup_steps < 0:
            raise ValueError('warmup_steps must be >= 0')

        self.warmup_steps = warmup_steps
        self.scheduler = scheduler

        super().__init__(self.scheduler.optimizer)

    def get_lr(self) -> List[float]:
        if self.warmup_steps == 0:
            return self.scheduler.get_last_lr()

        scale = min(self.last_epoch / self.warmup_steps, 1.0)
        scaled_lr = [lr * scale for lr in self.scheduler.get_last_lr()]

        return scaled_lr

    def step(self, epoch: Optional[int] = None) -> None:
        # we must skip self.scheduler.step() in __init__
        if self._step_count != 0:  # type: ignore
            with _ApplyAndRestoreLR(self.optimizer, self.scheduler.get_last_lr()):
                self.scheduler.step(epoch)

        super().step(epoch)


def tutorial_train_loop(
    epochs,
    model,
    optimizer,
    metric_function,
    loss_function,
    train_loader,
    validation_loader,
    scheduler=None,
) -> None:
    """Simple train loop."""
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

        n_batches = len(train_loader)
        train_loss /= n_batches
        train_accuracy /= n_batches

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

        n_batches = len(validation_loader)
        validation_loss /= n_batches
        validation_accuracy /= n_batches

        print('validation metrics:')
        print('  loss:', validation_loss)
        print('  accuracy:', validation_accuracy)

        print()
