from typing import Optional, Any

import torch
from matplotlib import pyplot as plt
from torchmetrics.metric import Metric


class SequenceMetric(Metric):

    def __init__(self, num_steps: int, num_classes: Optional[int] = None, flattened_input: bool = True):
        super().__init__()

        self.num_time_steps = num_steps
        self.add_state("num_samples", torch.zeros((num_steps)))
        self.add_state("num_corrects", torch.zeros((num_steps)))
        self.flattened_input = flattened_input

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """

        Args:
            preds: float Tensor of shape [batch_size, time_steps, num_classes]
                    or long Tensor of  shape [batch_size, time_steps]
            target: int Tensor of shape [batch_size, time_steps]

        Returns:

        """
        if self.flattened_input:
            preds = preds.unflatten(0, (-1, self.num_time_steps))
            target = target.unflatten(0, (-1, self.num_time_steps))

        preds = preds.cpu()
        target = target.cpu()

        if preds.dim() == 3:
            preds = torch.argmax(preds, dim=2)  # [batch_size, time_steps]

        batch_size, time_steps = preds.shape
        assert self.num_time_steps == time_steps

        for idb in range(batch_size):
            batch_pred = preds[idb]  # [time_steps]
            batch_target = target[idb]  # [time_steps]
            current_length = 0
            current_class = batch_target[0]

            for idt in range(self.num_time_steps):
                # if the current class has changed, track the new class
                if current_class != batch_target[idt]:
                    current_class = batch_target[idt]
                    current_length = 0

                self.num_samples[current_length] += 1
                self.num_corrects[current_length] += (batch_pred[idt] == batch_target[idt]).int()
                current_length += 1

    def compute(self) -> torch.Tensor:
        return self.num_corrects / self.num_samples

    def plot(self, val: Optional[torch.Tensor] = None, ax: Optional[plt.Axes] = None) -> Any:
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()

        time_steps = torch.arange(self.num_time_steps).cpu().numpy()
        accuracies = val if val is not None else self.compute()
        accuracies = accuracies.cpu().numpy()
        ax.bar(time_steps, accuracies)
        ax.set_xlabel("Time Step in Sequence")
        ax.set_ylabel("Accuracy")
        ax.set_ylim([0.0, 1.0])
        return fig, ax



def test_all_correct_same_class():
    preds = torch.tensor([[0, 0, 0, 0]])
    target = torch.tensor([[0, 0, 0, 0]])
    metric = SequenceMetric(num_steps=4)
    metric.update(preds, target)
    result = metric.compute()
    torch.equal(result, torch.tensor([1.0, 1.0, 1.0, 1.0]))


def test_all_wrong():
    preds = torch.tensor([[1, 2, 3, 4]])
    target = torch.tensor([[0, 0, 0, 0]])
    metric = SequenceMetric(num_steps=4)
    metric.update(preds, target)
    result = metric.compute()
    torch.equal(result, torch.tensor([0.0, 0.0, 0.0, 0.0]))


def test_two_correct_classes():
    preds = torch.tensor([[1, 1, 2, 2]])
    target = torch.tensor([[1, 1, 2, 2]])
    metric = SequenceMetric(num_steps=4)
    metric.update(preds, target)
    result = metric.compute()

    # the third and fourth entry should be zero since
    # we've only encountered one-class-sequences of length two
    torch.equal(result, torch.tensor([1.0, 1.0, 0.0, 0.0]))


def test_multiple_batches():
    preds = torch.tensor([[1, 0, 2, 2], [0, 0, 1, 0]])
    target = torch.tensor([[1, 1, 2, 2], [0, 0, 0, 0]])
    metric = SequenceMetric(num_steps=4)
    metric.update(preds, target)
    result = metric.compute()
    torch.equal(result, torch.tensor([1.0, 0.5, 0.0, 1.0]))
