# As of January 31 2016 (12:30 AM)
# TODO: make changes to work with Logan's code
#       add static plots and visualization
#       incorporate with fluid simulation

import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.animation as animation

class DiffuserSpecs:
    """
    DiffuserSpecs class, for specifying the geometry and other parameters
    needed to perform diffusion.

    Every parameter is public and accessible. This class is just a way to
    organize the information for the Diffuser class.
    """

    def __init__(self, box_shape, box_dims, intervals=100, duration=60,
            temp=0.0, alpha=1e-7):
        """
        DiffuserSpecs class initialization method.

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
        self.max_temp = max_temp
        self.alpha = alpha



class Diffuser:
    """
    A class for running simulated heat diffusion by numerically solving the
    partial differential equation
    \[
        \frac{\partial u}{\partial t} - \alpha \nabla^2 u,
    \]
    where $u : \Omega \times \mathbb{R} \to \mathbb{R}$ is a temperature 
    function and $\Omega \subseteq \mathbb{R}^2$.
    """
    def __init__(self, specs, source=False, fig=plt.figure()):
        """
        Diffuser class initialization method.

        Arguments:
        specs -- a DiffuserSpecs object
        source -- indicates whether a heat source is present (default False)
        fig -- where to send plots (default plt.figure())
        """
        self.__specs = specs
        self.__dx = self.__specs.length / self.__specs.cols
        self.__dy = self.__specs.width / self.__specs.rows
        self.__dt = self.__specs.duration / self.__specs.intervals
        self.grid = np.ones((self.__specs.rows + 1, self.__specs.cols + 1,
            self.__specs.intervals)) \
                    * self.__specs.max_temp
        self.__source = source
        self.__fig = fig

    def stable(self):
        """
        Method to determine if current specs will produce numerically stable
        results. It verifies the following inequality:
        \[
            dt \leq \frac{1}{2\alpha} \frac{(dx \ dy)^2}{(dx)^2 + (dy)^2)}
        \]
        Source: pending
        """
        dx, dy, dt = [self.__dx, self.__dy, self.__dt]
        alpha = self.__specs.alpha
        return (dt <= 1.0/(2.0 * alpha) * (dx * dy)**2 / (dx**2 + dy**2))

    def central_second_diff(next_u, current_u, prev_u, delta):
        """
        Compute central finite differentiation operation using the formula:
        \[
            \frac{\partial^2 T}{\partial x^2} \approx \frac{T_{i+1,j} 
                - 2 T_{i,j} + T_{i-1,j}}{(\Delta x)^2},
        \]
        This is O((\Delta x)^2), i.e. with quadratic error.

        Arguments:
        next_u -- T_{i+1,j} (float)
        current_u -- T_{i,j} (float)
        prev_u -- T_{i-1,j} (float)
        delta -- \Delta x (float)
        """
        num = next_u - 2 * current_u + prev_u
        den = delta * delta
        return num / den

    def diffusion_step(self, t):
        """
        Computes the next iteration of diffusion across the grid, using
        the update step
        \[
            T_{i,j,t+1} = T_{i,j,t} + \alpha \Nabla^2 T_{i,j,t},
        \]
        where 
        \[
            \Nabla^2 T = \frac{\partial^2 T}{\partial x^2} + 
            \frac{\partial^2 T}{\partial y^2}
        \]
        is the 2D Laplacian.

        Arguments:
        t -- the current time index (int)
        """
        m = self.__specs.rows
        n = self.__specs.cols
        dx = self.__dx
        dy = self.__dy
        dt = self.__dt
        alpha = self.__specs.alpha

        for i in range(1, m):
            for j in range(1, n):
                # Compute numerical approximation of the Laplacian
                xlap = Diffuser.central_second_diff(self.grid[i,j+1,t], 
                        self.grid[i,j,t], self.grid[i,j-1,t], dx)
                ylap = Diffuser.central_second_diff(self.grid[i-1,j,t], 
                        self.grid[i,j,t], self.grid[i+1,j,t], dy)
                self.grid[i,j,t+1] = self.grid[i,j,t] + alpha  * (xlap + ylap) * dt

    def animate(self, **kwargs):
        """
        Generate an animated gif of heat diffusion.

        Keyword arguments:
        passed for matplotlib customization

        TODO: make this savable.
        """
        ims = []
        for t in range(self.__specs.intervals-1):
            self.diffusion_step(t)
            im = plt.imshow(self.grid[:,:,t], **kwargs)
            ims.append([im])
        ani = animation.ArtistAnimation(self.__fig, ims,
                interval=self.__specs.intervals, blit=True)

        plt.show()
