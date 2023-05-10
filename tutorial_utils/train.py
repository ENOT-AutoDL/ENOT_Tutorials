from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import torch

# Base PyTorch scheduler.
# Added here to disable PyCharm warning everywhere.
# noinspection PyProtectedMember
Scheduler = torch.optim.lr_scheduler._LRScheduler


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
        Batch with logits or probabilities from model, of shape
        `ё`(batch_size, num_classes)`.
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
        for group, lr in zip(self.optimizer.param_groups, lr_to_apply):
            group['lr'] = lr

    def __enter__(self):
        self._apply_lr(self.new_lr)

    def __exit__(self, *args, **kwargs):
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

        # noinspection PyUnresolvedReferences
        super().__init__(self.scheduler.optimizer)

    def get_lr(self) -> List[float]:
        # This function has a lof of noinspection stuff because of the wrong
        # _LRScheduler description in the lr_scheduler.pyi PyTorch file.

        if self.warmup_steps == 0:
            return self.scheduler.get_last_lr()

        # noinspection PyUnresolvedReferences
        scale = min(self.last_epoch / self.warmup_steps, 1.0)
        # noinspection PyUnresolvedReferences
        scaled_lr = [lr * scale for lr in self.scheduler.get_last_lr()]

        return scaled_lr

    def step(self, epoch: Optional[int] = None) -> None:
        # we must skip self.scheduler.step() in __init__
        if self._step_count != 0:
            # noinspection PyUnresolvedReferences
            with _ApplyAndRestoreLR(self.optimizer, self.scheduler.get_last_lr()):
                self.scheduler.step(epoch)

        super().step(epoch)
