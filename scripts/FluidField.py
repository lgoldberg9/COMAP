# coding: utf-8
# As of January 31 2016 (12:30 AM)

import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def forward_diff(future, current, delta):
    return (future - current) / delta

def backward_diff(current, past, delta):
    return (current - past) / delta

def second_diff(future, current, past, delta):
    return (future - 2 * current + past) / (delta * delta)

class FluidField:

    def __init__(self, tub, F, rho, initial_temp):
        self.__tub = tub
        self.__u = np.zeros((tub.rows,tub.cols,tub.intervals))
        self.__v = np.zeros((tub.rows,tub.cols,tub.intervals))
        self.__p = np.ones((tub.rows,tub.cols,tub.intervals)) 
        self.__b = np.zeros((tub.rows,tub.cols,tub.intervals))
        self.__source_temperature = initial_temp
        
        # Field constants
        self.__F = F
        self.__rho = rho

    def get_u(self): return self.__u
    def get_v(self): return self.__v
    def get_p(self): return self.__p
        
    def set_conv(self, conv):
        self.__conv = conv
        
    def reinitialize(self):
        self.u = np.zeros((tub.rows,tub.cols,tub.intervals))
        self.v = np.zeros((tub.rows,tub.cols,tub.intervals))
        self.__p = np.ones((tub.rows,tub.cols,tub.intervals)) 
        self.__b = np.zeros((tub.rows,tub.cols,tub.intervals))

    def setup_b(self, ellipsoid, initial_b, t):
        h, k, a, b, p = ellipsoid
        for i in range(self.__tub.rows):
            for j in range(self.__tub.cols):              
                if self.__conv.check_ellipsoid(i, j, ellipsoid):
                    self.__b[i,j,t] = initial_b
                else:
                    self.__b[i,j,t] = 0.0

    def compute_pressure(self, t):
        dx = self.__conv.get_dx()
        dx *= dx
        dy = self.__conv.get_dy()
        dy *= dy
        p = self.__p
        for i in range(1, self.__tub.rows - 1):
            for j in range(1, self.__tub.cols - 1):
                result = (p[i+1,j, t-1] + p[i-1, j, t-1]) * dy
                result += (p[i, j+1, t-1] + p[i, j-1, t-1]) * dx
                result -= self.__b[i,j,t-1] * dx * dy
                result /= (2 * (dx + dy))
                self.__p[i, j, t] = result 
                

    def fluid_field(self, source_ellipsoid, sink_ellipsoid, t):
        u = self.__u
        v = self.__v
        rho = 1.0 / self.__rho
        alpha = self.__tub.alpha
        F = self.__F
        p = self.__p
        b = self.__b
        dx = self.__conv.get_dx()
        dy = self.__conv.get_dy()
        dt = self.__conv.get_dt() 

        self.setup_b(source_ellipsoid, 0.0, t)
        self.compute_pressure(t)
        
        for i in range(1, self.__tub.rows-1):
            for j in range(1, self.__tub.rows-1):
                if self.__conv.mask_ellipsoid(i,j):
                    p[i,j,t] = p[i,j,t-1]

                    du = -rho * forward_diff(p[i+1,j,t-1], p[i,j,t-1], dx)
                    du += alpha * second_diff(u[i+1,j,t-1], u[i,j,t-1], u[i-1,
                                                j, t-1], dx) 
                    du += alpha * second_diff(u[i,j+1,t-1], u[i,j,t-1],
                    u[i,j-1,t-1], dy)
                    du -= u[i,j,t-1] * forward_diff(u[i+1,j,t-1], u[i,j,t-1],
                    dx)
                    du -= v[i,j,t-1] * forward_diff(u[i,j+1,t-1], u[i,j,t-1],
                    dy)
                    du *= dt
                    u[i,j,t] = u[i,j,t-1] + du

                    dv = -rho * forward_diff(p[i,j+1,t-1], p[i,j,t-1], dy)
                    dv += alpha * second_diff(v[i+1,j,t-1], v[i,j,t-1], v[i-1,
                                                j, t-1], dx) 
                    dv += alpha * second_diff(v[i,j+1,t-1], v[i,j,t-1],
                    v[i,j-1,t-1], dy)
                    dv -= u[i,j,t-1] * forward_diff(v[i+1,j,t-1], v[i,j,t-1],
                    dx)
                    dv -= v[i,j,t-1] * forward_diff(v[i,j+1,t-1], v[i,j,t-1],
                    dy)
                    dv *= dt
                    v[i,j,t] = v[i,j,t-1] + dv
