from torch.utils.data import Dataset, DataLoader
import torch

import numpy as np
import matplotlib.pyplot as plt
import warnings


class PolyData(Dataset):


    """
    Dataset for toy regression problem
    """

    def __init__(self, polyf, varf, gaps, lower=-1, upper=1, size=1000, seed=1111, scramble:bool=False, num_heads:int=1):

        """
        Construct dataset attricutes and construct dataset

        Parameters
        ----------
        polyf : function
            polynomial function used to produce dataset.
        varf : function
            function that represents variance as a function of x. MAKE SURE IT'S POSITIVE ON [lower, upper].
        gaps : List[Tuple]
            list of tuples which represent intervals where there are gaps in the data.
        lower : int
            lower bound of polynomial function
        upper : int
            upper bound of polynomial function
        size : int
            number of examples in dataset
        seed : int
            seed for random number generator
        scramble : bool
            whether to scramble the dataset -> changes output shape to (size, num_heads, 2)
        num_heads : int
            number of heads to use for multihead model -> only used if scramble is True
        """

        # functions for variance and polynomial
        self.polyf = polyf
        self.varf = varf

        # regions of epistemic uncertainty
        self.lower = lower
        self.upper = upper
        self.gaps = gaps
        self.gaps.sort(key=lambda x: x[1])

        self.size = size

        self.rng = np.random.default_rng(seed)

        self.scramble = scramble
        self.num_heads = num_heads

        # make sure gaps are in function range
        if len(self.gaps) == 0:
            self.use_gaps = False
        else:
            self.use_gaps = True
            assert gaps[0][0] >= lower and gaps[-1][1] <= upper, "Gap intervals must with within ['lower', 'upper']."

        self.x, self.y = self._construct_dataset()

    def sample_xy(self, total_interval_size, intervals): 
      if self.use_gaps:
          # sample intervals according to their relative size to ensure [lower, upper] is uniform
          bin = self.rng.choice(len(self.gaps) + 1, p=[(i[1] - i[0]) / total_interval_size for i in intervals])
          
          if bin == 0:
              x = self.rng.uniform(low=self.lower, high=self.gaps[bin][0])
          elif bin == len(self.gaps):
              x = self.rng.uniform(low=self.gaps[-1][1], high=self.upper)
          else:
              x = self.rng.uniform(low=self.gaps[bin-1][1], high=self.gaps[bin][0])
      else:
          x = self.rng.uniform(low=intervals[0][0], high=intervals[0][1])
      
      # check variance
      var_x = self.varf(x)
      if var_x < 0:
          raise ValueError('variance function is negative in provided polynomial interval')

      # get poly out
      y = self.polyf(x) + self.rng.normal(loc=0, scale=np.sqrt(self.varf(x)))
      
      return x,y

    def _construct_dataset(self):
        """
        Randomly construct the polynormial dataset, taking into account the gap intervals and variance function

        Returns
        -------
        x_list : array-like
            list of x values
        y_list : array-like
            list of y values
        """
        
        x_list, y_list = np.zeros((self.size,)), np.zeros((self.size,))
        if self.scramble:
          x_list = np.zeros((self.size, self.num_heads))
          y_list = np.zeros((self.size, self.num_heads)) 

        # convert gaps to valid intervals
        intervals = []
        if self.use_gaps:
            assert len(self.gaps) > 0
            intervals.append((self.lower, self.gaps[0][0]))
            for idx in range(1, len(self.gaps)):
                intervals.append((self.gaps[idx-1][1], self.gaps[idx][0]))
            intervals.append((self.gaps[-1][1], self.upper))
        else:
            intervals = [(self.lower, self.upper)]

        # sample
        total_interval_size = sum(i[1] - i[0] for i in intervals)
        for idx in range(self.size):
          if self.scramble:
            for head in range(self.num_heads):
              x, y = self.sample_xy(total_interval_size, intervals)
              x_list[idx][head] = x
              y_list[idx][head] = y
          else:
            x, y = self.sample_xy(total_interval_size, intervals)
            # append to lists
            x_list[idx] = x
            y_list[idx] = y


        return x_list, y_list

    def __getitem__(self, idx):
        return self.x[idx].astype(np.float32), self.y[idx].astype(np.float32)

    def __len__(self):
        return self.size


if __name__ == '__main__':

    # TEST
    
    # polynomial function
    def polyf(x):
        return 2*x ** 3 + 5*x ** 2 + 2

    # variance function
    def varf(x):
        return 0.5*x + 0.5

    # instantiate dataset and data loader
    Data = PolyData(polyf, varf, gaps=[])
    dl = DataLoader(Data, batch_size=32, shuffle=True)

    # get data
    x_collect, y_collect = [], []
    for x, y in dl:
        x_collect.extend(x.tolist())
        y_collect.extend(y.tolist())

    # plot
    plt.plot(x_collect, y_collect, 'o')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()