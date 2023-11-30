from model import SimpleMaskEstimator
from data import RandomDataset
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from callbacks import VisualizeMaskCallback
from argparse import ArgumentParser
import matplotlib.pyplot as plt
from glob import glob
import os


pl.seed_everything(42)



def show_images(folder_name, tv_loss_weight=None,  use_tv_loss=False):
       
    images = glob(os.path.join(folder_name, './mask_*.png'))
    if use_tv_loss:
        images = [i for i in images if 'tv_loss' in i]
        images = sorted(images, key=lambda x: int(x.split('_')[2].split('.')[0]))
    else:
        images = [i for i in images if 'tv_loss' not in i]
        images = sorted(images, key=lambda x: int(x.split('_')[-1].split('.')[0]))

    image_num = len(images)
    fig, axs = plt.subplots(1, image_num, figsize=(20, 10))
    plt.subplots_adjust(wspace=0.05, hspace=0.05)  # Adjust these values as needed
    plt.tight_layout()
    for i, img in enumerate(images):
        axs[i].imshow(plt.imread(img), cmap='gray')
        axs[i].set_title(os.path.basename(img))
        axs[i].axis('off')
        axs[i].set_xticks([])
        axs[i].set_yticks([])
        # axs[i].set_frame_on(False)
        # axs[i].set_adjustable('box')
    plt.suptitle(f'{"TV" if use_tv_loss else "reconstuction"} Loss{" with" + str(tv_loss_weight) if tv_loss_weight else ""}', fontsize=16)
    plt.savefig(os.path.join(folder_name, f'result_{"use_tv_loss" if use_tv_loss else ""}.png'))



if __name__ == '__main__':
    

    parser = ArgumentParser()
    parser.add_argument('--tv_loss', action='store_true')
    parser.add_argument('--tv_loss_weight', type=float, default=0.1)
    args = parser.parse_args()
    
    folder_name = "no_tv_loss"
    if args.tv_loss:
        folder_name = f"tv_loss_{args.tv_loss_weight}"
    
    # create a folder to save the images
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)
    
    dataset = RandomDataset(step=4)

    dataloader = DataLoader(dataset, batch_size=1, num_workers=4, persistent_workers=True)

    model = SimpleMaskEstimator(use_tv_loss=args.tv_loss, tv_loss_weight=args.tv_loss_weight)

    trainer = pl.Trainer(max_epochs=150, callbacks=[VisualizeMaskCallback(dataset, every_n_epochs=30, folder_name=folder_name)])

    trainer.fit(model, dataloader)
    
    show_images(folder_name, use_tv_loss=args.tv_loss, tv_loss_weight=args.tv_loss_weight)
    
    trainer.test(model, dataloader)