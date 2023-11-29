# here define a callback function where we can visualize the model's prediction (the mask) every n epochs


import torch
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import PIL

class VisualizeMaskCallback(pl.Callback):
    def __init__(self, dataset, every_n_epochs:int=10):
        self.every_n_epochs = every_n_epochs
        

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.current_epoch % self.every_n_epochs == 0:
            x, y = batch
            with torch.no_grad():
                y_hat = pl_module(x)
            plt.imshow(y_hat.cpu().squeeze().cpu(), cmap='gray')
            if pl_module.use_tv_loss:
                plt.savefig(f'./mask_epoch_{trainer.current_epoch}_tv_loss.png')
            else:
                plt.savefig(f'./mask_epoch_{trainer.current_epoch}.png')
            
            