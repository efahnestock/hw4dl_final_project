import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from hw4dl import ROOT_DIR

def create_map2loc(polyf, varf, dir, gaps=[], x_bounds=(-1, 1), y_bounds=(-1, 1), samples=1000, seed=1111):

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
    """

    # make sure gaps are in the bounds of the function
    if len(gaps) > 0:
        assert all(gaps[i][0] >= x_bounds[0] and gaps[i][1] <= x_bounds[1] for i in range(len(gaps))), "gaps in the x direction leave the function domain"
        assert all(gaps[i][2] >= y_bounds[0] and gaps[i][3] <= y_bounds[1] for i in range(len(gaps))), "gaps in the y direction leave the function domain" 
        assert all(gaps[i][1] > gaps[i][0] and gaps[i][3] > gaps[i][2] for i in range(len(gaps))), "invalid gap dimension"

    # instantiate rng
    rng = np.random.default_rng(seed)

    # make grid
    x_g = np.linspace(x_bounds[0], x_bounds[1], 100)
    y_g = np.linspace(y_bounds[0], y_bounds[1], 100)
    Y, X = np.meshgrid(y_g, x_g)

    # dataframe to keep track of data
    df = pd.DataFrame(columns=['File', 'X', 'Y'])

    # keep track of successfull generation
    gen_count = 0

    with tqdm(total=samples) as pbar:
        while gen_count < samples:
            
            # sample a location
            x = rng.uniform(low=x_bounds[0], high=x_bounds[1])
            y = rng.uniform(low=y_bounds[0], high=y_bounds[1])

            # rejection sampling based on gaps
            if len(gaps) > 0:
                reject = False
                for gap in gaps:
                    if (x >= gap[0] and x <= gap[1]) and (y >= gap[2] and y <= gap[3]):
                        reject = True
                        break
                if reject:
                    continue
            
            # get height on map with appropriate variance added
            var = varf(x, y)
            if var < 0:
                raise ValueError('variance function is negative in provided polynomial interval')
            z = polyf(x, y) + rng.normal(loc=0, scale=np.sqrt(var))
            
            # append to df
            df = pd.concat([df, pd.DataFrame({'File' : [f'{gen_count}.jpg'], 'X' : [x], 'Y' : [y], 'Z' : [z]})], ignore_index=True)

            plt.contourf(X, Y, polyf(X,Y), 30, cmap='viridis')
            plt.scatter(x, y, s=200, marker='o', c='pink')
            plt.axis('off')
            plt.savefig(os.path.join(dir, f'{gen_count}.jpg'), bbox_inches='tight', pad_inches=0.0)
            plt.close()

            # update count and progress bar
            gen_count += 1
            pbar.update(1)
    
    df.to_csv(os.path.join(dir, 'description.csv'), index=False)

if __name__ == '__main__':

    dir = os.path.join(ROOT_DIR, 'hw4dl/datasets/map2loc/')

    def polyf(x, y):
        return x ** 2 + y ** 2
    
    def varf(x, y):
        return 0.5*x + 0.5*y + 1

    
    create_map2loc(polyf, varf, dir, gaps=[(0.1, 0.8, 0.1, 0.8)], samples=5)

    df = pd.read_csv(os.path.join(dir, 'description.csv'))
    x = df['X'].to_numpy()
    y = df['Y'].to_numpy()
    plt.figure()
    plt.plot(x, y, 'o')
    plt.show()