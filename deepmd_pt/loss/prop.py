import torch
from deepmd_pt.utils.env import GLOBAL_PT_FLOAT_PRECISION
from deepmd_pt.loss import TaskLoss
from deepmd_pt.utils import env
import torch.nn.functional as F

class PropertyLoss(TaskLoss):

    def __init__(self,
                 mean: float,
                 std: float,
                 func: str = "smooth_mae",
                 metric: list = ["mae"],
                 **kwargs):
        """Construct a layer to compute loss on property."""
        super(PropertyLoss, self).__init__()
        self.mean = mean
        self.std = std
        self.func = func
        self.metric = metric
        self.beta = kwargs.get("beta", 1.00)

    def forward(self, model_pred, label, natoms, learning_rate, mae=False):
        """Return loss on loss and force.
        Args:
        - model_pred: Property prediction.
        - label: Target property.
        - natoms: Tell atom count.

        Returns:
        - loss: Loss to minimize.
        """
        loss = torch.tensor(0.0, dtype=env.GLOBAL_PT_FLOAT_PRECISION, device=env.DEVICE)
        more_loss = {}
        
        label_mean = torch.tensor(self.mean, dtype=env.GLOBAL_PT_FLOAT_PRECISION, device=env.DEVICE)
        label_std = torch.tensor(self.std, dtype=env.GLOBAL_PT_FLOAT_PRECISION, device=env.DEVICE)
        label_norm = (label['property'] - label_mean) / label_std
        
        # loss
        if self.func == "smooth_mae":
            loss += F.smooth_l1_loss(label_norm, model_pred['property'], reduction="sum", beta=self.beta)
        elif self.func == "mae":
            loss += F.l1_loss(label_norm, model_pred['property'], reduction="sum")
        elif self.func == "mse":
            loss += F.mse_loss(label_norm, model_pred['property'], reduction="sum")
        elif self.func == "rmse":
            loss += torch.sqrt(F.mse_loss(label_norm, model_pred['property'], reduction="mean"))
        else:
            raise RuntimeError(f"Unknown loss function : {self.func}")
        
        # more_loss
        prop_pred = (model_pred['property'] * label_std) + label_mean
        if "smooth_mae" in self.metric:
            more_loss["smooth_mae"] = F.smooth_l1_loss(label['property'], prop_pred, reduction="mean", beta=self.beta).detach()
        if "mae" in self.metric:
            more_loss['mae'] = F.l1_loss(label['property'], prop_pred, reduction="mean").detach()
        if "mse" in self.metric:
            more_loss['mse'] = F.mse_loss(label['property'], prop_pred, reduction="mean").detach()
        if "rmse" in self.metric:
            more_loss['rmse'] = torch.sqrt(F.mse_loss(label['property'], prop_pred, reduction="mean")).detach()

        return loss, more_loss