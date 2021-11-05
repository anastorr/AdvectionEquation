import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from schemes import leapfrog, lax_wendroff, corner_transport_upwind, upwind

mpl.use('Qt5Agg')


def c0(x, y, x0, y0, r, sigma0):
    x_mesh, y_mesh = np.meshgrid(x, y)
    return np.where((((x_mesh-x0)**2+(y_mesh-y0)**2) > r**2), 0, np.e**(-((x_mesh-x0)**2+(y_mesh-y0)**2)/sigma0**2)).T


def c1(x, y, x0, y0, r):
    x_mesh, y_mesh = np.meshgrid(x, y)
    return np.where(((abs(x_mesh-x0) < r)&(abs(y_mesh-y0) < r)), 1, 0).T


def c2(x, y, x0, y0, r, sigma0, sigma1):
    x_mesh, y_mesh = np.meshgrid(x, y)
    return np.where((((x_mesh-x0)**2/sigma0**2+(y_mesh-y0)**2/sigma1**2) > r**2), 0, np.e**(-((x_mesh-x0)**2/sigma0**2+(y_mesh-y0)**2/sigma1**2))).T


def plot_solution_3d(x, y, solution):
    X, Y = np.meshgrid(x, y)
    ax1 = plt.axes(projection='3d')
    im1 = ax1.plot_surface(X, Y, solution, cmap='viridis', alpha=0.9)
    plt.savefig('leapfrog_t5_h100.png')


def plot_solution_lvl(x, y, solution):
    X, Y = np.meshgrid(x, y)
    ax2 = plt.axes()
    im = ax2.contourf(X, Y, solution, cmap='viridis')
    plt.colorbar(im)
    plt.show()


if __name__ == '__main__':
    x0 = 2500
    y0 = 0
    r = 1000
    sigma0 = 500
    sigma1 = 700
    omega = 0.0005
    L = 5000
    h = 100
    tau = 5
    T = 2*np.pi/omega
    x = np.arange(-L, L, h)
    y = np.arange(-L, L, h)
    t = np.arange(0, T, tau)
    c = np.zeros((t.shape[0], x.shape[0], y.shape[0]))
    c[0] = c0(x, y, x0, y0, r, sigma0)
    u = -omega*y
    v = omega*x
    v, u = np.meshgrid(v, u)
    c = leapfrog(c, tau, h, u, v)
    plot_solution_3d(x, y, np.round(c[-1], 6))

