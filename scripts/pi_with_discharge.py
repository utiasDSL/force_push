import numpy as np
import matplotlib.pyplot as plt


tf = 10.0
dt = 0.0001
N = int(tf / dt)

ki = 0.1
kd = 1.1
I = 0

ts = np.array([i * dt for i in range(N)])
Is = np.zeros(N)

for i in range(N):
    t = ts[i]
    e = 1 if t < 5 else 0
    I += ki * dt * e
    # I *= kd
    # I = kd * (I + dt * e)
    Is[i] = I
    # I = 0.9 * I

plt.plot(ts, Is)
plt.grid()
plt.show()


