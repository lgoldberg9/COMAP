# coding: utf-8
# As of January 31 2016 (12:30 AM)

import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class FluidField:

    def __init__(self, tub, conv, F, rho, initial_temp):
        self.__tub = tub
        self.__conv = conv
        self.u = np.zeros((tub.rows,tub.columns,tub.intervals))
        self.v = np.zeros((tub.rows,tub.columns,tub.intervals))
        self.__p = np.ones((tub.rows,tub.columns,tub.intervals)) 
        self.__b = np.zeros((tub.rows,tub.columns,tub.intervals))
        self.__source_temperature = initial_temp

        # Field constants
        self.F = F
        self.rho = rho

    def reinitialize(self):
        self.u = np.zeros((tub.rows,tub.columns,tub.intervals))
        self.v = np.zeros((tub.rows,tub.columns,tub.intervals))
        self.__p = np.ones((tub.rows,tub.columns,tub.intervals)) 
        self.__b = np.zeros((tub.rows,tub.columns,tub.intervals))

    def __buildUpB(self, source, rho, dt, dx, dy, u, v):
        b = np.zeros_like(u)
        b[1:-1,1:-1] = rho*(1/dt*((u[1:-1,2:]-u[1:-1,0:-2])/ \
                (2 * dx) + (v[2:,1:-1] - v[0:-2,1:-1]) / (2 * dy)) - \
                np.power((u[1:-1,2:] - u[1:-1,0:-2]) / (2 * dx), 2.0) - 2 * \
                ((u[2:,1:-1] - u[0:-2,1:-1]) / (2 * dy) * \
                (v[1:-1,2:] - v[1:-1,0:-2]) / (2 * dx)) - \
                np.power((v[2:,1:-1] - v[0:-2,1:-1]) / (2 * dy), 2.0)
        return b

    def __presPoissPeriodic(self, p, dx, dy):
        pn = np.empty_like(p)
        for q in range(nit):
            pn = p.copy()
            p[1:-1,1:-1] = ((pn[1:-1,2:] + pn[1:-1,0:-2]) * (dy * dy) + \
                (pn[2:,1:-1] + pn[0:-2,1:-1]) * (dx * dx)) / \
                (2 * ((dx * dx) + (dy * dy)) - \
                (dx * dx) * (dy * dy) / (2 * ((dx * dx) + (dy * dy)) * \
                b[1:-1,1:-1]
        return p

    def fluid_field(self, source_i, source_j, sink_i, sink_j):
        self.__conv.calculate_ellipse(source_i, source_j)
        udiff = 1
        stepcount = 0

        while udiff > .001:
            un = u.copy()
            vn = v.copy()

            b = self.__buildUpB(source, rho, self.__conv.get_dt, self.__conv.get_dx, self.__conv.get_dy, u, v)
            p = self.__presPoissPeriodic(p, self.__conv.get_dx, self.__conv.get_dy)

            u[1:-1,1:-1] = un[1:-1,1:-1] - un[1:-1,1:-1,] * dt / dx * \
                (un[1:-1,1:-1] - un[1:-1,0:-2]) - vn[1:-1,1:-1,] * \
                dt / dy * (un[1:-1,1:-1] - un[0:-2,1:-1]) - \
                dt / (2 * rho * dx) * (p[1:-1,2:] - p[1:-1,0:-2]) + \
                self.__tub.alpha * (dt / (dx * dx) * (un[1:-1,2:] - 2 * \
                un[1:-1,1:-1] + un[1:-1,0:-2]) + dt / (dy * dy) * \
                (un[2:,1:-1] - 2 * un[1:-1,1:-1] + un[0:-2,1:-1])) + F * dt

            v[1:-1,1:-1] = vn[1:-1,1:-1] - un[1:-1,1:-1] * \
                dt / dx * (vn[1:-1,1:-1] - vn[1:-1,0:-2]) - \
                vn[1:-1,1:-1] * dt / dy * (vn[1:-1,1:-1] - \
                vn[0:-2,1:-1]) - dt / (2 * rho * dy) * (p[2:,1:-1] - \
                p[0:-2,1:-1]) + self.__tub.alpha * (dt / (dx * dx) * (vn[1:-1,2:] - \
                2 * vn[1:-1,1:-1] + vn[1:-1,0:-2]) + (dt / (dy * dy) * \
                (vn[2:,1:-1] - 2 * vn[1:-1,1:-1] + vn[0:-2,1:-1])))
            udiff = (np.sum(u)-np.sum(un))/np.sum(u)
            stepcount += 1

        return u, v


