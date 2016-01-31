
# coding: utf-8

# In[ ]:

# As of January 31 2016 (12:30 AM)

import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class FluidField:
    
    def __init__(self, Bathtub, Diffuser, F, rho, initial_temp):
        
        self.__tub = Bathtub
        self.__diff = Diffuser
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
        
        
        
        b[1:-1,1:-1]=rho*(1/dt*((u[1:-1,2:]-u[1:-1,0:-2])/(2*dx)+(v[2:,1:-1]-v[0:-2,1:-1])/(2*dy))-                          ((u[1:-1,2:]-u[1:-1,0:-2])/(2*dx))**2-                          2*((u[2:,1:-1]-u[0:-2,1:-1])/(2*dy)*(v[1:-1,2:]-v[1:-1,0:-2])/(2*dx))-                          ((v[2:,1:-1]-v[0:-2,1:-1])/(2*dy))**2)
        return b
    
    def __presPoissPeriodic(self, p, dx, dy):
        pn = np.empty_like(p)
        
        for q in range(nit):
            pn = p.copy()
            p[1:-1,1:-1] = ((pn[1:-1,2:]+pn[1:-1,0:-2])*dy**2+(pn[2:,1:-1]+pn[0:-2,1:-1])*dx**2)/                (2*(dx**2+dy**2)) -                dx**2*dy**2/(2*(dx**2+dy**2))*b[1:-1,1:-1]
        return p
    
    def fluid_field(source_i, source_j, sink_i, sink_j):
    
        diff.calculate_ellipse(source_i, source_j)
    
        
    
        udiff = 1
        stepcount = 0
        
        while udiff > .001:
            un = u.copy()
            vn = v.copy()

            b = buildUpB(source, rho, diff.get_dt, diff.get_dx, diff.get_dy, u, v)
            p = presPoissPeriodic(p, diff.get_dx, diff.get_dy)

            u[1:-1,1:-1] = un[1:-1,1:-1]-                un[1:-1,1:-1,]*dt/dx*(un[1:-1,1:-1]-un[1:-1,0:-2])-                vn[1:-1,1:-1,]*dt/dy*(un[1:-1,1:-1]-un[0:-2,1:-1])-                dt/(2*rho*dx)*(p[1:-1,2:]-p[1:-1,0:-2])+                tub.alpha*(dt/dx**2*(un[1:-1,2:]-2*un[1:-1,1:-1]+un[1:-1,0:-2])+                dt/dy**2*(un[2:,1:-1]-2*un[1:-1,1:-1]+un[0:-2,1:-1]))+F*dt

            v[1:-1,1:-1] = vn[1:-1,1:-1]-                un[1:-1,1:-1]*dt/dx*(vn[1:-1,1:-1]-vn[1:-1,0:-2])-                vn[1:-1,1:-1]*dt/dy*(vn[1:-1,1:-1]-vn[0:-2,1:-1])-                dt/(2*rho*dy)*(p[2:,1:-1]-p[0:-2,1:-1])+                tub.alpha*(dt/dx**2*(vn[1:-1,2:]-2*vn[1:-1,1:-1]+vn[1:-1,0:-2])+                (dt/dy**2*(vn[2:,1:-1]-2*vn[1:-1,1:-1]+vn[0:-2,1:-1])))



            udiff = (np.sum(u)-np.sum(un))/np.sum(u)
            stepcount += 1
        return u, v
    

