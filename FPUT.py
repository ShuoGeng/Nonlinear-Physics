# There are 31 movable particles in total, with a total of 33 particles. 
# The positions of the first and last particles are fixed at 0.
# The initial displacement x_i of each particle is sin⁡(i\*pi/32)
# where i represents the particle index. The initial velocity of all particles is zero.
# The simulation uses a time step of 0.01, and the total number of simulation steps is 1,000,000.


import numpy as np
import matplotlib.pyplot as plt

N = 31
dt = 0.01
steps = 1000000
k0 = 1
alpha = 0.25
beta = 0.1
omega = np.zeros(N)
psi = np.zeros((N, N))
x = np.zeros((N + 2, steps))
v = np.zeros((N + 2, steps))
a = np.zeros((N + 2, steps))

# Initialization
for i in range(1, N + 1):
    omega[i - 1] = 2 * np.sin(i * np.pi / (2 * (N + 1)))
    x[i, 0] = np.sin(i * np.pi / 32)
    v[i, 0] = 0

for i in range(1, N + 1):
    for j in range(1, N + 1):
        psi[i - 1, j - 1] = np.sqrt(2 / (N + 1)) * np.sin(i * j * np.pi / (N + 1))

# Newton's law of motion
for i in range(1, N + 1):
    a[i, 0] = k0 * (x[i + 1, 0] + x[i - 1, 0] - 2 * x[i, 0]) + alpha * (
            (x[i + 1, 0] - x[i, 0]) ** 2 - (x[i, 0] - x[i - 1, 0]) ** 2) + beta * (
                          (x[i + 1, 0] - x[i, 0]) ** 3 - (x[i, 0] - x[i - 1, 0]) ** 3)
    x[i, 1] = x[i, 0] + v[i, 0] * dt + 0.5 * a[i, 0] * dt ** 2
    v[i, 1] = v[i, 0] + a[i, 0] * dt

# Verlet algorithm
for t in range(1, steps-1):
    for i in range(1, N + 1):
        a[i, t] = k0 * (x[i + 1, t] + x[i - 1, t] - 2 * x[i, t]) + alpha * (
                (x[i + 1, t] - x[i, t]) ** 2 - (x[i, t] - x[i - 1, t]) ** 2) + beta * (
                          (x[i + 1, t] - x[i, t]) ** 3 - (x[i, t] - x[i - 1, t]) ** 3)
        x[i, t + 1] = 2 * x[i, t] - x[i, t - 1] + a[i, t] * dt ** 2
        v[i, t] = (x[i, t + 1] - x[i, t - 1]) / (2 * dt)

q = np.zeros(N)
p = np.zeros(N)
E = np.zeros((N, steps))

for t in range(steps):
    q = np.dot(psi, x[1:N + 1, t])
    p = np.dot(psi, v[1:N + 1, t])
    E[:, t] = 0.5 * (p * p + omega * omega * q * q)
for i in range(8):
    plt.plot(np.arange(steps-1) * dt, E[i, 0:steps-1])
plt.xlabel('Time')
plt.ylabel('Energy')
plt.show()
