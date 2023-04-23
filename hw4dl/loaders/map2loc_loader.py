import os

from torch.utils.data import Dataset, DataLoader
import torch
import skimage.io as io
import pandas as pd


class Map2Loc(Dataset):

    """
    Dataset for map2loc task
    """

    def __init__(self, root_dir, csv_file, transform=None):
        
        self.root_dir = root_dir
        self.df = pd.read_csv(os.path.join(self.root_dir, csv_file))
        self.transform = transform

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.df.iloc[idx,0])
        image = io.imread(img_name)
        
        target = self.df.iloc[idx,3]

        if self.transform:
            image = self.transform(image)

        return image, target

    def __len__(self):
        return len(self.df)

if __name__ == '__main__':

    # create dataloader
    Data = Map2Loc(root_dir='../datasets/map2loc', csv_file='description.csv')
    dl = DataLoader(Data, batch_size=32, shuffle=True)
    
    # manually get a batch
    dl = iter(dl)
    batch = next(dl)
    
    # plot first image
    import matplotlib.pyplot as plt
    plt.imshow(batch[0][0])
    plt.show()