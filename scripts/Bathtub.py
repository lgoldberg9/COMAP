# As of January 31 2016 (12:30 AM)
# TODO: make changes to work with Logan's code
#       add static plots and visualization
#       incorporate with fluid simulation

import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.animation as animation

class Bathtub:
    """
    Bathtub class, for specifying the geometry and other parameters
    needed to perform diffusion.

    Every parameter is public and accessible. This class is just a way to
    organize the information for the Diffuser class.
    """

    def __init__(self, box_shape, box_dims, intervals=100, duration=60,
            temp=0.0, alpha=1e-7, material=1.0):
        """
        Bathtub class initialization method.

        Arguments:
        box_shape -- a 2-element list of integers specifying rows and cols
        box_dims -- a 2-element list of floats specifying length and width


        Keyword arguments:
        intervals -- number of time steps for the diffusion (default 100)
        duration -- number of time units for the diffusion (default 60)
        temp -- global temperature default (default 0.0)
        alpha -- diffusion parameter (default 1e-7)
        """

        self.rows = box_shape[0]
        self.cols = box_shape[1]
        self.length = box_dims[0]
        self.width = box_dims[1]
        self.intervals = intervals
        self.duration = duration
        self.temp = temp
        self.alpha = alpha
        self.material = 1.0
