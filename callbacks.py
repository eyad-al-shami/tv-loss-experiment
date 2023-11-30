# here define a callback function where we can visualize the model's prediction (the mask) every n epochs


import torch
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import PIL
import os
class VisualizeMaskCallback(pl.Callback):
    def __init__(self, dataset, every_n_epochs:int=10, folder_name:str='no_tv_loss'):
        self.every_n_epochs = every_n_epochs
        self.folder_name = folder_name
        

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.current_epoch % self.every_n_epochs == 0:
            x, y = batch
            with torch.no_grad():
                y_hat = pl_module(x)
            # remove any white spaceing aroung the image
            plt.axis('off')
            plt.tight_layout(pad=0)
            plt.imshow(y_hat.cpu().squeeze().cpu(), cmap='gray')
            
            if pl_module.use_tv_loss:
                image_path = os.path.join(self.folder_name, f'mask_epoch_{trainer.current_epoch}_tv_loss.png')
            else:
                image_path = os.path.join(self.folder_name, f'mask_epoch_{trainer.current_epoch}.png')
            plt.savefig(image_path)
            
            