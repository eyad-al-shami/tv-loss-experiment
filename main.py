from model import SimpleMaskEstimator
from data import RandomDataset
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from callbacks import VisualizeMaskCallback
from argparse import ArgumentParser
import matplotlib.pyplot as plt

pl.seed_everything(42)



def show_images(use_tv_loss=False):
    from glob import glob
    import os
    
    images = glob('./mask_*.png')
    if use_tv_loss:
        images = [i for i in images if 'tv_loss' in i]
        images = sorted(images, key=lambda x: int(x.split('_')[2].split('.')[0]))
    else:
        images = [i for i in images if 'tv_loss' not in i]
        images = sorted(images, key=lambda x: int(x.split('_')[-1].split('.')[0]))

    image_num = len(images)
    fig, axs = plt.subplots(1, image_num, figsize=(20, 10))
    for i, img in enumerate(images):
        axs[i].imshow(plt.imread(img), cmap='gray')
        axs[i].set_title(os.path.basename(img))
        axs[i].axis('off')
        axs[i].set_xticks([])
        axs[i].set_yticks([])
        axs[i].set_frame_on(False)
        axs[i].set_adjustable('box')
    plt.tight_layout()
    plt.savefig(f'result_{"use_tv_loss" if use_tv_loss else ""}.png')



if __name__ == '__main__':
    

    parser = ArgumentParser()
    parser.add_argument('--tv_loss', action='store_true')
    args = parser.parse_args()
    
    dataset = RandomDataset(step=4)

    dataloader = DataLoader(dataset, batch_size=1, num_workers=4, persistent_workers=True)

    model = SimpleMaskEstimator(use_tv_loss=args.tv_loss)

    trainer = pl.Trainer(max_epochs=150, callbacks=[VisualizeMaskCallback(dataset, every_n_epochs=30)])

    trainer.fit(model, dataloader)
    
    show_images(use_tv_loss=args.tv_loss)
    
    trainer.test(model, dataloader)