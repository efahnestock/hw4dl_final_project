import os

from torch.utils.data import Dataset, DataLoader
import torch
import skimage.io as io
import pandas as pd
from torchvision import transforms
import numpy as np
from PIL import Image


class Map2Loc(Dataset):
    """
    Dataset for map2loc task
    """

    def __init__(self, root_dir, csv_file, transform=None):

        self.root_dir = root_dir
        self.df = pd.read_csv(os.path.join(self.root_dir, csv_file))
        self.transform = transforms.ToTensor()
        self.gaps = [(2, 4, 2, 4), (7, 12, 7, 12)]
        self.shape = (15,15)

        def polyf(x, y):
            return x ** 2 + y ** 2

        def varf(x, y):
            return 0.5 * x + 0.5 * y + 1

        self.polyf = polyf
        self.varf = varf

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.df.iloc[idx, 0])
        image = np.load(img_name)

        target = np.load(img_name.split(".npy")[0] + "_label.npy")

        if self.transform:
            image = self.transform(image)

        return image, target

    def __len__(self):
        return len(self.df)


if __name__ == '__main__':
    # create dataloader
    Data = Map2Loc(root_dir='../datasets/map2loc_prototype', csv_file='description.csv')
    dl = DataLoader(Data, batch_size=10, shuffle=True)

    # manually get a batch
    dl = iter(dl)
    batch = next(dl)
    print(batch[0].shape)
    # plot first image
    import matplotlib.pyplot as plt

    plt.imshow(transforms.ToPILImage()(batch[0][4]))

    plt.show()