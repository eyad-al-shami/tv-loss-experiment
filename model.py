from typing import Any
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
import torch.nn as nn
from torch.nn import functional as F
# imprort pytorch_lightning as pl
import pytorch_lightning as pl


class SimpleMaskEstimator(pl.LightningModule):
    def __init__(self, input_dim=96, use_tv_loss=False, tv_loss_weight=0.1):
        super().__init__()
        
        self.use_tv_loss = use_tv_loss
        self.tv_loss_weight = tv_loss_weight
        self.mask_estimator = nn.Sequential(
            nn.Conv2d(input_dim, 128, kernel_size=7, padding="same"),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=5, padding="same"),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 1, kernel_size=3, padding=1)
        )
    
    def forward(self, x):
        features = self.mask_estimator(x)
        return features
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        
        if (self.use_tv_loss):
            tv_loss = self.total_variation_loss(y_hat).sum()
            loss += (tv_loss * self.tv_loss_weight)
        
        self.log('train_loss', loss, on_epoch=True, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)        
        self.log('test_loss', loss, on_epoch=True, prog_bar=True)
    
    def configure_optimizers(self) -> Any:
        optimizer = torch.optim.Adam(self.parameters(), lr=0.005)
        return optimizer
    
    def total_variation_loss(self, x):
        """Compute total variation statistics."""
        diff1 = x[..., 1:, :] - x[..., :-1, :]
        diff2 = x[..., :, 1:] - x[..., :, :-1]

        res1 = torch.square(diff1).sum([1, 2, 3])
        res2 = torch.square(diff2).sum([1, 2, 3])
        score = res1 + res2
        return score



