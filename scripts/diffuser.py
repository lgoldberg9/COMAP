# As of January 31 2016 (12:30 AM)
# TODO: make changes to work with Logan's code
#       add static plots and visualization
#       incorporate with fluid simulation

import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.animation as animation

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
    def __init__(self, tub, source=False, fig=plt.figure()):
        """
        Diffuser class initialization method.

        Arguments:
        tub -- a Bathtub object
        source -- indicates whether a heat source is present (default False)
        fig -- where to send plots (default plt.figure())
        """
        self.__tub = tub
        self.__dx = self.__tub.length / self.__tub.cols
        self.__dy = self.__tub.width / self.__tub.rows
        self.__dt = self.__tub.duration / self.__tub.intervals
        self.grid = np.ones((self.__tub.rows + 1, self.__tub.cols + 1,
            self.__tub.intervals)) \
                    * self.__tub.temp
        self.__source = source
        self.__ellipse = (0,0,0,0)
        self.__fig = fig

    def stable(self):
        """
        Method to determine if current tub will produce numerically stable
        results. It verifies the following inequality:
        \[
            dt \leq \frac{1}{2\alpha} \frac{(dx \ dy)^2}{(dx)^2 + (dy)^2)}
        \]
        Source: pending
        """
        dx, dy, dt = [self.__dx, self.__dy, self.__dt]
        alpha = self.__tub.alpha
        return (dt <= 1.0/(2.0 * alpha) * (dx * dy)**2 / (dx**2 + dy**2))

    def define_boundary(self):
        pass

    def ij_to_xy(self, i, j):
        """
        Converts a grid (i,j) row-column coordinate on the meshgrid into a
        Cartesian (x,y) pair based on the geometry of the Bathtub.
        """
        return ((j/self.__tub.cols - 1/2)*self.__tub.length, (1/2 -
            i/self.__tub.rows)*self.__tub.width)

    def calculate_ellipse(self, i, j):
        x, y = self.ij_to_xy(i, j)
        h, k, a, b = self.__ellipse
        return a * (x - h)**2 + b * (y - k)**2 <= 1.0


    def setup_source(self, ellipse, temperature, source=False):
        """ 
        Add a \"faucet\" source to the first iteration of the diffusion 
        simulation. If the argument source is True, make this source permanent.

        Arguments:
        ellipse -- a list of 4 floats [h,k,a,b] which will define the inequality
        \[
            a(x  - h)^2 + b(y - k)^2 <= 1,
        \]
        which defines an elliptical disk in the region.
        temperature -- the temperature value to assign the source (float)


        Keyword Arguments:
        source -- determines if the faucet source remains on (default False)
        """
        self.__source = source
        self.__ellipse = tuple(ellipse)
        if self.__source:
            time = self.__tub.intervals
        else:
            time = 1
        print(time)
        for i in range(self.__tub.rows):
            for j in range(self.__tub.cols):
                if self.calculate_ellipse(i, j):
                    for t in range(time):
                        self.grid[i, j, t] = temperature

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
        r"""
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

        m = self.__tub.rows
        n = self.__tub.cols
        dx = self.__dx
        dy = self.__dy
        dt = self.__dt
        alpha = self.__tub.alpha

        for i in range(1, m):
            for j in range(1, n):
                # Compute numerical approximation of the Laplacian
                xlap = Diffuser.central_second_diff(self.grid[i,j+1,t], 
                        self.grid[i,j,t], self.grid[i,j-1,t], dx)
                ylap = Diffuser.central_second_diff(self.grid[i-1,j,t], 
                        self.grid[i,j,t], self.grid[i+1,j,t], dy)
                if not (self.__source and self.calculate_ellipse(i, j)):
                    self.grid[i,j,t+1] = self.grid[i,j,t] + alpha  * (xlap + ylap) * dt

    def animate(self, **kwargs):
        """
        Generate an animated gif of heat diffusion.

        Keyword arguments:
        passed for matplotlib customization

        TODO: make this savable.
        """
        ims = []
        for t in range(self.__tub.intervals-1):
            self.diffusion_step(t)
            im = plt.imshow(self.grid[:,:,t], **kwargs)
            ims.append([im])
        ani = animation.ArtistAnimation(self.__fig, ims,
                interval=self.__tub.intervals, blit=True)

        plt.show()
