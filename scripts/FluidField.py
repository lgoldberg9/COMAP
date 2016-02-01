# coding: utf-8
# As of January 31 2016 (12:30 AM)

import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.animation as animation

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

    def get_u(): return u
    def get_v(): return v
        
    def set_conv(self, conv):
        self.__conv = conv
        
    def reinitialize(self):
        self.u = np.zeros((tub.rows,tub.cols,tub.intervals))
        self.v = np.zeros((tub.rows,tub.cols,tub.intervals))
        self.__p = np.ones((tub.rows,tub.cols,tub.intervals)) 
        self.__b = np.zeros((tub.rows,tub.cols,tub.intervals))

    def buildUpB(self, rho, dt, dx, dy, u, v, t, ellipse):
        self.__b = np.zeros_like(u)
        axis1, axis2, q = ellipse
        
        print(u[0,0,0])
        
        for i in range(1, np.shape(u)[0]):
            for j in range(1, np.shape(v)[1]):
                if np.power(i / axis1, q) + np.power(j / axis2, q) <= 1:
                    self.__b[i,j,t] = rho * (1 / dt * ((u[i,j+1,t-1] - u[i,j-1,t-1]) / (2 * dx) + \
                            (v[i+1,j,t-1] - v[i-1,j,t-1]) / (2 * dy)) - \
                            np.power((u[i,j+1,t-1] - u[i,j-1,t-1]) / (2 * dx), 2.0) - 2 * \
                            (u[i+1,j,t-1] - u[i-1,j,t-1]) / (2 * dy) * \
                            (v[i,j+1,t-1] - v[i,j-1,t-1]) / (2 * dx) - \
                            np.power((v[i+1,j,t-1] - v[i-1,j,t-1]) / (2 * dy), 2.0))
        return self.__b

    def __presPoissPeriodic(self, p, dx, dy):
        pn = np.empty_like(p)
                            
                            
        for q in range(self.__tub.intervals):
            pn = p.copy()
            
            p[1:-1,1:-1,t] = ((p[1:-1,2:,t-1] + p[1:-1,0:-2,t-1]) * (dy * dy) + \
                (p[2:,1:-1,t-1] + p[0:-2,1:-1,t-1]) * (dx * dx)) / \
                2 * ((dx * dx) + (dy * dy)) - \
                (dx * dx) * (dy * dy) / (2 * ((dx * dx) + (dy * dy))) * \
                self.__b[1:-1,1:-1]
        return p

    def fluid_field(self, source_i, source_j, sink_i, sink_j):
        #self.__conv.calculate_ellipse(source_i, source_j)
        axis1, axis2, q = self.__tub.tubshape
        u = self.__u
        v = self.__v
        rho = self.__rho
        F = self.__F
        dx = self.__conv.get_dx
        dy = self.__conv.get_dy
        dt = self.__conv.get_dt
        

        for t in range(1,self.__tub.intervals):
            
            b = self.buildUpB(rho, dt, dx, dy, u, v, t, self.__tub.tubshape)
            p = self.__presPoissPeriodic(p, dx, dy)

            for i in range(1, np.shape(u)[0]):
                for j in range(1, np.shape(v)[1]):   
                    if np.power(i / axis1, q) + np.power(j / axis2, q) < 1:
                        u[i,j,t] = u[i,j,t-1] - u[i,j,t-1] * dt / dx * \
                            (u[i,j,t-1] - u[i,j-1,t-1]) - v[i,j,t-1] * \
                            dt / dy * (u[i,j,t-1] - u[i-1,j,t-1]) - \
                            dt / (2 * rho * dx) * (p[i,j+1,t-1] - p[i,j-2,t-1]) + \
                            tub.alpha * (dt / (dx * dx) * (u[i,j+1,t-1] - 2 * \
                            u[i,j,t-1] + u[i,j-1,t-1]) + dt / (dy * dy) * \
                            (u[i+1,j,t-1] - 2 * u[i,j,t-1] + u[i-1,j,t-1])) + F * dt

                        v[i,j,t] = v[i,j,t-1] - u[i,j,t-1] * \
                            dt / dx * (v[i,j,t-1] - v[i,j-1,t-1]) - \
                            v[i,j,t-1] * dt / dy * (v[i,j,t-1] - \
                            v[i-1,j,t-1]) - dt / (2 * rho * dy) * (p[i+1,j,t-1] - \
                            p[i-1,j,t-1]) + tub.alpha * (dt / (dx * dx) * (v[i,j+1,t-1] - \
                            2 * v[i,j,t-1] + v[i,j-1,t-1]) + (dt / (dy * dy) * \
                            (v[i+1,j,t-1] - 2 * v[i,j,t-1] + v[i-1,j,t-1])))
                                         
                    elif np.power(i/a,p) + np.power(j/b,p) == 1:
                        u[i,j] = u[i,j,t-1] - \
                            u[i,j,t-1] * dt / dx * (u[i,j,t-1] - u[i,j-1,t-1]) - \
                            v[i,j,t-1] * dt / dy * (u[i,j,t-1] - u[i-1,j,t-1]) - \
                            dt / (2 * rho * dx) * (p[p,j+2]-p[i,j]) + \
                            tub.alpha*(dt / np.power(dx,2) * (u[i,j+2] - 2*u[i,j+1] + u[i,j]) + \
                            dt / np.power(dy,2) * (u[i-1,j+1] - 2*u[i,j-1] + u[i-1,j+1])) + F * dt

                        v[i,j] = v[i,j] - \
                            u[i,j,t-1] * dt / dx * (v[i,j,t-1] - v[i,j-1,t-1]) - \
                            v[i,j,t-1] * dt / dy * (v[i,j,t-1] - v[i-1,j,t-1]) - \
                            dt / (2 * rho * dy) * (p[i+1,j,t-1] - p[i-1,j,t-1]) + tub.alpha * \
                            (dt / np.power(dx,2) * (v[i,0,t-1] - 2*v[i,j,t-1] + v[i,j-1,t-1]) + \
                            (dt / np.power(dy,2) * (v[i+1,j,t-1] - 2*v[i,j,t-1] + v[i-1,j,t-1])))


