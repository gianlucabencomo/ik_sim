import argparse
import numpy as np
import matplotlib.pyplot as plt
from functools import partial

def circle(t: float, r: float, h: float, k: float, f: float) -> np.ndarray:
    """Return the (x, y) coordinates of a circle with radius r centered at (h, k)
    as a function of time t and frequency f."""
    x = r * np.cos(2 * np.pi * f * t) + h
    y = r * np.sin(2 * np.pi * f * t) + k
    z = 0.2
    return np.array([x, y, z])

def trefoil(t: float, r: float, h: float, k: float, f: float) -> np.ndarray:
    x = r * np.cos(2 * np.pi * f * t) + 2 * r * np.cos(4 * np.pi * f * t) + h
    y = 2* r * np.sin(2 * np.pi * f * t) - 4 * r * np.sin(4 * np.pi * f * t) + k
    z = 2 * r * np.sin(3 * np.pi * f * t) + 0.22
    return np.array([x, y, z])

def plot(f: callable, t_f: float, dt: float):
    timesteps = np.arange(0., t_f, step=dt)
    xyzs = np.empty((timesteps.shape[0], 3))
    for i, t in enumerate(timesteps):
        xyzs[i] = f(t)

    ax = plt.figure().add_subplot(projection='3d')
    ax.plot(*xyzs.T, lw=1)
    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.set_zlabel("Z Axis")
    ax.set_xlim(-0.1, 0.1)
    ax.set_ylim(-0.1, 0.1)
    ax.set_zlim(0, 0.5)


    plt.show()

if __name__ == "__main__":
    f = partial(trefoil, r=0.04, h=0.0, k=0.0, f=0.2)
    plot(f, 30, 0.01)