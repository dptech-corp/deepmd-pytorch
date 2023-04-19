import torch


class TaskLoss(torch.nn.Module):

    def __init__(self, **kwargs):
        """Construct loss."""
        super(TaskLoss, self).__init__()

    def forward(self, model_pred, label, natoms, learning_rate):
        """Return loss .
        """
        raise NotImplementedError

