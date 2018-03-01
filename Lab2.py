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
alpha = 5 * (pi / 11)  # rad

m = 0.009  # kg
g = 9.8  # m/sec^2

w = -10  # m/sec
A = 1.e-5  # N*sec/m
B = 1.e-8  # N*sec^3/m^3

tm = 110.0  # sec

vertical_v0 = v0 * np.sin(alpha)
horizontal_v0 = v0 * np.cos(alpha)


def horizontal_windage(v):
    global w, m, A, B
    return (A * v + B * (v - w) ** 3) / m


def vertical_windage(v):
    global w, m, A, B
    return (A * v + B * v ** 3) / m


def horizontal_system(f, t):
    global g
    y = f[0]
    v = f[1]
    dydt = v
    dvdt = -horizontal_windage(v)
    return [dydt, dvdt]


def vertical_system(f, t):
    global g
    z = f[0]
    v = f[1]
    dzdt = v
    dvdt = -g - vertical_windage(v)
    return [dzdt, dvdt]


nt = 1000
t = np.linspace(0., tm, nt)
print("NODES:", len(t))

horizontal_sol = odeint(horizontal_system, [y0, horizontal_v0], t)
y = horizontal_sol[:, 0]
yv = horizontal_sol[:, 1]

vertical_sol = odeint(vertical_system, [z0, vertical_v0], t)
z = vertical_sol[:, 0]
zv = vertical_sol[:, 1]

flight_time = 0
flight_range = 0
landing_node = 0
max_lift = max(z)
for i in range(len(z)):
    if z[i] < 0.0:
        landing_node = i
        flight_time = (t[landing_node] + t[landing_node - 1]) / 2.0
        flight_range = (y[landing_node] + y[landing_node - 1]) / 2.0
        break

plt.figure(figsize=(6, 6))

# subplot 1
plt.subplot(221)

pos = 3000
delta_pos = -425
plt.text(4500, pos, "BOOM ANGLE: " + str(round(((alpha * 180) / pi), 2)) + " DEG")
pos = pos + (2 * delta_pos)
plt.text(4500, pos, "LANDING NODE: " + str(landing_node))
pos = pos + delta_pos
plt.text(4500, pos, "MAX LIFT: " + str(round(max_lift, 2)))
pos = pos + delta_pos
plt.text(4500, pos, "FLIGHT TIME: " + str(round(flight_time, 2)))
pos = pos + delta_pos
plt.text(4500, pos, "FLIGHT RANGE: " + str(round(flight_range, 2)))

plt.plot(y, z, 'b-', linewidth=2)
plt.axis([0., 3500., 0., 3500.])
plt.xlabel("Y-AXIS")
plt.ylabel("Z-AXIS")
plt.grid(True)

# subplot 3
plt.subplot(223)
plt.plot(t, yv, 'r-', linewidth=2)
plt.xlabel("t")
plt.ylabel("Vy(t)")
plt.grid(True)

# subplot 4
plt.subplot(224)
plt.plot(t, zv, 'r-', linewidth=2)
plt.xlabel("t")
plt.ylabel("Vz(t)")
plt.grid(True)

plt.savefig("Lab2Graph.pdf", dpi=300)
plt.show()
