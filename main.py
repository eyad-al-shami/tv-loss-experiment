from model import SimpleMaskEstimator
from data import RandomDataset
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from callbacks import VisualizeMaskCallback
from argparse import ArgumentParser
import matplotlib.pyplot as plt
from glob import glob
import os
import numpy as np
from PIL import Image


pl.seed_everything(42)



def show_images(folder_name, tv_loss_weight=None,  use_tv_loss=False):
       
    images = glob(os.path.join(folder_name, 'mask_*.png'))

    if use_tv_loss:
        images = [i for i in images if 'tv_loss' in i]
        images = sorted(images, key=lambda x: int(x.split('_')[-3]))
    else:
        images = sorted(images, key=lambda x: int(x.split('_')[-1].split('.')[0]))

    num_images = len(images)
    grid_size = int(np.ceil(np.sqrt(num_images)))

    plt.figure(figsize=(grid_size * 3, grid_size * 3))  # 3 inches per image

    # Loop over the image files and add them to the plot
    for i, file in enumerate(images):
        # Read the image
        print(file)
        img = Image.open(file)
        axs = plt.subplot(grid_size, grid_size, i + 1)
        plt.imshow(img, cmap='gray')
        epoch_num = file.split('_')[-3]
        axs.set_title(f"Epoch {epoch_num}", fontsize=10)
        plt.suptitle(f'{"TV" if use_tv_loss else "reconstuction"} Loss{" , weight: " + str(tv_loss_weight) if tv_loss_weight else ""}', fontsize=16)
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(folder_name, f'result_{"use_tv_loss" if use_tv_loss else ""}.png'))
    
    
    
    
    
if __name__ == '__main__':
    

    parser = ArgumentParser()
    parser.add_argument('--tv_loss', action='store_true')
    parser.add_argument('--circle', action='store_true')
    parser.add_argument('--tv_loss_weight', type=float, default=1.0)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--save_every_n_epochs', type=int, default=None, help='save the image every n epochs (has higher priority than epochs_to_check)')
    parser.add_argument('--epochs_to_check', nargs='+', type=int, default=[0, 5, 8, 10, 15, 20, 30, 40, 49], help='save the image at these epochs (has lower priority than save_every_n_epochs)')
    args = parser.parse_args()
    
    folder_name = "no_tv_loss"
    if args.tv_loss:
        folder_name = f"tv_loss_{args.tv_loss_weight}"
    
    # create a folder to save the images
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)
    
    dataset = RandomDataset(step=4, circle=args.circle)

    dataloader = DataLoader(dataset, batch_size=1, num_workers=1, persistent_workers=True)

    model = SimpleMaskEstimator(use_tv_loss=args.tv_loss, tv_loss_weight=args.tv_loss_weight)

    trainer = pl.Trainer(max_epochs=args.epochs, callbacks=[VisualizeMaskCallback(dataset, every_n_epochs=args.save_every_n_epochs, epochs_to_check= args.epochs_to_check, folder_name=folder_name)])

    trainer.fit(model, dataloader)
    
    show_images(folder_name, use_tv_loss=args.tv_loss, tv_loss_weight=args.tv_loss_weight)
    
    trainer.test(model, dataloader)