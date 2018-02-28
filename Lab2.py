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


def system(f, t):
    global m, g, A, B
    z = f[0]
    v = f[1]
    dzdt = v
    dvdt = -g - (A * v + B * v ** 3) / m
    return [dzdt, dvdt]


nt = 1000
t = np.linspace(0., tm, nt)

print("NODES:", len(t))

print()
print("ALPHA:", alpha)

sol = odeint(system, [z0, v0], t)
res_z = sol[:, 0]
res_v = sol[:, 1]

res_flight_time = 0
for i in range(len(res_z)):
    if res_z[i] < 0.0:
        res_flight_time = (t[i] + t[i - 1]) / 2.0
        print()
        print("LANDING NODE (RES):", i)
        print("FLIGHT TIME (RES):", res_flight_time)
        print("MAX LIFT (RES):", max(res_z))
        break

free_z = np.linspace(0.,0.,nt)
free_v = np.linspace(0.,0.,nt)
for i in range(len(t)):
    free_z[i] = z0 + (v0 * t[i]) - (g * t[i] * t[i] / 2)
    free_v[i] = v0 - (g * t[i])

free_flight_time = 0
for i in range(len(free_z)):
    if free_z[i] < 0.0:
        free_flight_time = (t[i] + t[i - 1]) / 2.0
        print()
        print("LANDING NODE (FREE):", i)
        print("FLIGHT TIME (FREE):", free_flight_time)
        print("MAX LIFT (FREE):", max(free_z))
        break

plt.figure(figsize=(10, 4))

# subplot 1
plt.subplot(121)
plt.text(18,50,"RES",color='r',fontsize=14,rotation=-40)
plt.text(30,210,"FREE",color='b',fontsize=14,rotation=-40)
plt.plot(t, [0.0] * nt, 'm-', linewidth=1)
plt.plot(t, res_v, 'r-', linewidth=2)
plt.plot(t, free_v, 'b-', linewidth=2)
plt.axis([0, free_flight_time + 1, -500., 500.])
plt.grid(True)
plt.xlabel("t")
plt.ylabel("v(t)")

# subplot 2
plt.subplot(122)
plt.text(14,3700,"RES",color='r',fontsize=14,rotation=10)
plt.text(16,10250,"FREE",color='b',fontsize=14,rotation=60)
plt.plot(t, res_z, 'r-', linewidth=2)
plt.plot(t, free_z, 'b-', linewidth=2)
plt.axis([0, free_flight_time + 1., 0., 13500.])
plt.grid(True)
plt.xlabel("t")
plt.ylabel("z(t)")

plt.savefig("Lab2Graph.pdf", dpi=300)
plt.show()
