import numpy as np
v0=30 #desirable velocity, in m/s (default: 30)
T=1 #float, optional  safe time headway, in s (default: 1)
a=1.5 #float, optional  comfortable acceleration, in m/s2 (default: 1.5)
b=1.5 #float, optional  comfortable deceleration, in m/s2 (default: 1.5)
delta=4 #float, optional  acceleration exponent (default: 4)
s0=2 #float, optional linear jam distance, in m (default: 2)
time_delay=0.0
dt=0.1 #float, optional timestep, in s (default: 0.1)
noise=0 #float, optional  std dev of normal perturbation to the acceleration (default: 0)
fail_safe=None,
sumo_cf_params=None

#following dynamic
def IDM(deltav,headway,v):#headway, velocity, delta_velocity(v-v_lead)

    if abs(headway) < 1e-3:
        headway = 1e-3
    s_star = s0 + max(0, v * T + v*deltav/(2 * np.sqrt(a * b)))
    return a * (1 - (v / v0) ** delta - (s_star / headway)**2)



#visulization
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
fig = plt.figure()
h = np.linspace(0,60,100)
Deltav = np.linspace(-6,12,100)
Headway,DELTAV = np.meshgrid(h,Deltav)
#fix v=20m/s
xn, yn = Headway.shape
geta = np.array(Headway)

for xk in range(xn):
    for yk in range(yn):
        geta[xk,yk] = max(IDM(DELTAV[xk,yk],Headway[xk,yk],20),-8)

surf = plt.contourf( DELTAV, Headway,geta, 20, cmap=cm.coolwarm)
plt.colorbar()
C = plt.contour(DELTAV, Headway,geta, 20, colors = 'black')
#plt.clabel(C, inline = True, fontsize = 10)
plt.show()
