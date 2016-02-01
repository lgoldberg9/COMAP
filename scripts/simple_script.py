from Convector import Convector
from FluidField import FluidField
from Bathtub import Bathtub

import numpy as np
import matplotlib.pyplot as plt

rows = 2**5
cols = 2**5

length = 1.0
width = 1.0

intervals = 200
duration = 60

temperature = 20.0
initial_temp = 10.0
alpha = 1e-4
material = 1.0
F = 0.0
rho = 1.0

tub = Bathtub([rows, cols], [length,width], intervals, duration, temperature,
alpha, material)

heat = Convector(tub)
vfield = FluidField(tub, F, rho, initial_temp)

heat.set_vfield(vfield)
vfield.set_conv(heat)

boundary = (0,0,length,width, 3.0)
heat.define_boundary(boundary)

source = (-0.5, 0, 0.2, 0.2, 2.0)
heat.setup_source(source, temperature + 50.0, source=True)
vfield.setup_b(source, 25.0, 0)
vfield.compute_pressure(0)

print(heat.stable())

# for t in range(1,10):
#     vfield.fluid_field(source, None, t)
#     heat.convection_step(t)
#     plt.imshow(heat.grid[:,:,t])
#     plt.show()
#     plt.close()
# 
# 
# 
heat.animate(cmap="gray",vmin=0.0,vmax=80,interpolation="nearest")
