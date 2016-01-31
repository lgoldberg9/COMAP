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
        dx, dy, dt = [self.__dx, self.__dy, self.__dt]
        alpha = self.__specs.alpha
        return (dt <= 1.0/(2.0 * alpha) * (dx * dy)**2 / (dx**2 + dy**2))

    def central_second_diff(next_u, current_u, prev_u, delta):
        num = next_u - 2 * current_u + prev_u
        den = delta * delta
        return num / den

    def diffusion_step(self, t):
        m = self.__specs.rows
        n = self.__specs.cols
        dx = self.__dx
        dy = self.__dy
        dt = self.__dt
        alpha = self.__specs.alpha

        for i in range(1, m):
            for j in range(1, n):
                xlap = Diffuser.central_second_diff(self.grid[i,j+1,t], 
                        self.grid[i,j,t], self.grid[i,j-1,t], dx)
                ylap = Diffuser.central_second_diff(self.grid[i-1,j,t], 
                        self.grid[i,j,t], self.grid[i+1,j,t], dy)
                self.grid[i,j,t+1] = self.grid[i,j,t] + alpha  * (xlap + ylap) * dt

    def animate(self, **kwargs):
        ims = []
        for t in range(self.__specs.intervals-1):
            self.diffusion_step(t)
            im = plt.imshow(self.grid[:,:,t], **kwargs)
            ims.append([im])
        ani = animation.ArtistAnimation(self.__fig, ims,
                interval=self.__specs.intervals, blit=True)

        plt.show()
