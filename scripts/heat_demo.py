import numpy as np 
import matplotlib.pyplot as plt 

m = 2**4
n = 2**4 
T = 200 

max_temp = 50.0 
duration = 40.0
xlen = 50 
ylen = 50 

dx = xlen / n
dy = ylen / m
dt = duration / T

# water's alpha constant
# alpha = 1.4e-7
alpha = 1e1

u = np.ones((m+1,n+1, T)) * max_temp
u[4:12,5:15,0] = max_temp * 2.0

def central_second_diff(next_u, current_u, prev_u, delta):
    numerator = next_u - 2 * current_u + prev_u
    denominator = delta * delta
    return numerator / denominator

def heat_diffusion(u, t):
    for i in range(1, m):
        for j in range(1, n):
            xlap = central_second_diff(u[i,j+1,t], u[i,j,t], u[i,j-1,t], dx)
            ylap = central_second_diff(u[i-1,j,t], u[i,j,t], u[i+1,j,t], dy)
            u[i,j,t+1] = u[i,j,t] + alpha  * (xlap + ylap) * dt
