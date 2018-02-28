"""
dz(t)/dt = v(t)
dv(t)/dt = -g - (A*v(t) + B*v(t)^3)/m
"""

import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
from scipy.integrate import odeint

z0 = 0.0  # m
y0 = 0.0  # m

v0 = 500.0  # m/sec
alpha = pi / 4  # rad

m = 0.009  # kg
g = 9.8  # m/sec^2

A = 1.e-5  # N*sec/m
B = 1.e-8  # N*sec^3/m^3
tm = 110.0  # sec

vertical_v0 = v0 * np.sin(alpha)
horizontal_v0 = v0 * np.cos(alpha)


def windage(v):
    global m, A, B
    return (A * v + B * v ** 3) / m


def horizontal_system(f, t):
    global g
    y = f[0]
    v = f[1]
    dydt = v
    dvdt = -windage(v)
    return [dydt, dvdt]


def vertical_system(f, t):
    global g
    z = f[0]
    v = f[1]
    dzdt = v
    dvdt = -g - windage(v)
    return [dzdt, dvdt]


nt = 5000
t = np.linspace(0., tm, nt)

print("NODES:", len(t))

print()
print("ALPHA:", alpha)

horizontal_sol = odeint(horizontal_system, [y0, horizontal_v0], t)
y = horizontal_sol[:, 0]
yv = horizontal_sol[:, 1]

vertical_sol = odeint(vertical_system, [z0, vertical_v0], t)
z = vertical_sol[:, 0]
zv = vertical_sol[:, 1]

flight_time = 0
for i in range(len(z)):
    if z[i] < 0.0:
        flight_time = (t[i] + t[i - 1]) / 2.0
        print()
        print("LANDING NODE (RES):", i)
        print("FLIGHT TIME (RES):", flight_time)
        print("MAX LIFT (RES):", max(z))
        break

# plt.savefig("Lab2Graph.pdf", dpi=300)
# plt.show()
