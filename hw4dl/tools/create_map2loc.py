import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import sys
from PIL import Image
import pdb
ROOT_DIR = "/data/vision/phillipi/perception/hw4dl_final_project"

def create_map2loc(polyf, varf, dir, gaps=[], x_bounds=(-1, 1), y_bounds=(-1, 1), samples=1000, seed=1111,
                   shape=(10, 10), patch_size=2, task="patch"):
    """
    Function to create the dataset for the map2loc task.

    Parameters
    ----------
    polyf : function
        polynomial function used to produce dataset.
    varf : function
        function that represents variance as a function of x. MAKE SURE IT'S POSITIVE ON x_bounds AND y_bounds.
    dir : str
        directory to which we save images and 'description.csv' with files names and labels
    gaps : List[Tuple]
        list of tuples which define square regions to not generate samples in.
        of the form (x_lower, x_upper, y_lower, y_upper).
    x_bounds : Tuple
        tuple to establish x limits of the function. of the form (x_lower, x_upper)
    y_bounds : Tuple
        tuple to establish y limits of the function. of the form (y_lower, y_upper)
    samples : int
        number of images to generate
    seed : int
        random seed for RNG
    shape : Tuple
        tuple to establish the shape of each image in the datase.
    patch_size : int
        If task=="patch" denotes the size of the patch in each image
    task: str
        Denotes the problem setting for the dataset. Takes either "pixel" or "patch". "pixel" generates images that are
        (shape, shape) zero-filled arrays, in which one pixel is filled as a one. "patch" generates images that are (shape, shape) zero-filled
        arrays in which a (patch_size, patch_size) patch is filled with ones.
    """

    # make sure gaps are in the bounds of the function
    if len(gaps) > 0:
        assert all(gaps[i][0] >= 0 and gaps[i][1] <= shape[1] for i in
                   range(len(gaps))), "gaps in the x direction leave the function domain"
        assert all(gaps[i][2] >= 0 and gaps[i][3] <= shape[1] for i in
                   range(len(gaps))), "gaps in the y direction leave the function domain"
        assert all(
            gaps[i][1] > gaps[i][0] and gaps[i][3] > gaps[i][2] for i in range(len(gaps))), "invalid gap dimension"

    # instantiate rng
    rng = np.random.default_rng(seed)

    # make grid
    x_g = np.linspace(x_bounds[0], x_bounds[1], shape[0])
    y_g = np.linspace(y_bounds[0], y_bounds[1], shape[1])
    Y, X = np.meshgrid(y_g, x_g)
    ground_truth = polyf(X, Y)

    # dataframe to keep track of data
    df = pd.DataFrame(columns=['File', 'X', 'Y'])

    # keep track of successfull generation
    gen_count = 0

    with tqdm(total=samples) as pbar:
        while gen_count < samples:
            image = np.zeros(shape)

            if task == "patch":
                # sample a location
                x = rng.integers(low=0, high=shape[0] - patch_size)
                y = rng.integers(low=0, high=shape[1] - patch_size)
                x_grid, y_grid = np.meshgrid(list(range(x, x + patch_size + 1)), list(range(y, y + patch_size + 1)))
                # Normalize grid positions to make inputs to polynomial/variance functions
                x_inpt = (2 * x_grid) / shape[0] - 1
                y_inpt = (2 * y_grid) / shape[1] - 1
                # rejection sampling based on gaps
                if len(gaps) > 0:
                    reject = False
                    for gap in gaps:
                        if (x >= gap[1] or (x + patch_size + 1) <= gap[0]) and (y >= gap[3] or (y + patch_size + 1) <= gap[2]):
                            reject = True
                    if reject:
                        continue
                # get height on map with appropriate variance added
                var = varf(x_inpt, y_inpt)
                if (var < 0).any():
                    raise ValueError('variance function is negative in provided polynomial interval')

                z = polyf(x_inpt, y_inpt) + rng.normal(loc=0, scale=np.sqrt(var), size=var.shape)
                image[x:x + patch_size + 1, y:y + patch_size + 1] = 1

            else:
                # sample a location
                x = rng.integers(low=0, high=shape[0])
                y = rng.integers(low=0, high=shape[1])
                # Normalize grid positions to make inputs to polynomial/variance functions
                x_inpt = (2 * x) / shape[0] - 1
                y_inpt = (2 * y) / shape[1] - 1
                # rejection sampling based on gaps
                if len(gaps) > 0:
                    reject = False
                    for gap in gaps:
                        if (x >= gap[0] and x <= gap[1]) and (y >= gap[2] and y <= gap[3]):
                            reject = True
                    if reject:
                        continue
                # get height on map with appropriate variance added
                var = varf(x_inpt, y_inpt)
                if var < 0:
                    raise ValueError('variance function is negative in provided polynomial interval')

                z = polyf(x_inpt, y_inpt) + rng.normal(loc=0, scale=np.sqrt(var))
                image[x, y] = 1

            image = (image * 255).astype(np.uint8)

            # append to df
            df = pd.concat([df, pd.DataFrame({'File': [f'{gen_count}.npy'], 'X': [x], 'Y': [y], 'Z': [z]})],
                           ignore_index=True)

            np.save(os.path.join(dir, f'{gen_count}'), image)
            np.save(os.path.join(dir, f'{gen_count}_label'), z)

            # update count and progress bar
            gen_count += 1
            pbar.update(1)

    df.to_csv(os.path.join(dir, 'description.csv'), index=False)


if __name__ == '__main__':
    task = "pixel"
    dir = os.path.join(ROOT_DIR, f'hw4dl/datasets/map2loc_{task}/')
    testdir = os.path.join(ROOT_DIR, f'hw4dl/datasets/map2loc_{task}_test/')

    def polyf(x, y):
        return x ** 2 + y ** 2

    def varf(x, y):
        return 0.5 * x + 0.5 * y + 1

    create_map2loc(polyf, varf, dir, gaps=[(2, 4, 2, 4), (7, 12, 7, 12)], samples=10000, shape=(15,15), patch_size=2, task=task)
    create_map2loc(polyf, varf, testdir, gaps=[], samples=5000, shape=(15,15), patch_size=2, task=task)

    df = pd.read_csv(os.path.join(dir, 'description.csv'))
    x = df['X'].to_numpy()
    y = df['Y'].to_numpy()
    plt.figure()
    plt.plot(x, y, 'o')
    plt.show()