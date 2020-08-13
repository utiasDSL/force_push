import numpy as np
import matplotlib.pyplot as plt

a = 0.1
b = 0.1
t = np.linspace(0, 10, 100)
x = (a + b*t) * np.cos(t)
y = (a + b*t) * np.sin(t)
plt.plot(x, y)
plt.grid()
plt.show()
