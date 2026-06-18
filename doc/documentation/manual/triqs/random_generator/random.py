from triqs.mc_tools import *
from triqs.plot.mpl_interface import *

r = RandomGenerator("mt19937_64", 237489)
l = []
for i in range(10000): l += [r(),]
plt.hist(l, 30, density=True)
