import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.use('Qt5Agg')


def plot_results(x, plots):
    for plot in plots:
        plt.plot(x, plot[0], label=plot[1])
    plt.legend()
    plt.show()


def lax_wendroff(c, u, h, tau):
    for n in range(1, c.shape[0]):
        c[n, 1:-1] = (c[n - 1, 1:-1] - u / 2 / h * tau * (c[n - 1, 2:] - c[n - 1, :-2])
                      + u ** 2 * tau ** 2 / 2 / h ** 2 * (c[n - 1, 2:] - 2 * c[n - 1, 1:-1] + c[n - 1, :-2]))
    return c


def minmod(r):
    return np.maximum(np.zeros(r.shape[0]), np.minimum(np.ones(r.shape[0]), r))


def van_leer(r):
    return (r + abs(r)) / (1 + abs(r))


def mc(r):
    return np.maximum(np.zeros(r.shape[0]), np.minimum(np.ones(r.shape[0]) * 2, 2 * r, (1 + r) / 2))


def superbee(r):
    return np.maximum(np.zeros(r.shape[0]), np.minimum(np.ones(r.shape[0]), 2 * r),
                      np.minimum(np.ones(r.shape[0]) * 2, r))


def tvd(c, u, h, tau, limiter):
    phi = np.copy(c)
    for n in range(1, phi.shape[0]):
        r1 = (phi[n - 1, 1:-1] - phi[n - 1, :-2]) / np.where(abs(phi[n - 1, 2:] - phi[n - 1, 1:-1]) > 0,
                                                             phi[n - 1, 2:] - phi[n - 1, 1:-1], 1)
        r0 = (phi[n - 1, :-2] - np.insert(phi[n - 1, :-3], 0, 0)) / np.where(
            abs(phi[n - 1, 1:-1] - phi[n - 1, :-2]) > 0, phi[n - 1, 1:-1] - phi[n - 1, :-2], 1)

        psi1 = limiter(r1)
        psi0 = limiter(r0)
        mu = u * tau / h
        phi[n, 1:-1] = phi[n - 1, 1:-1] - mu * (1 - 0.5 * (1 - mu) * psi0) * (phi[n - 1, 1:-1] - phi[n - 1, :-2]) \
                       - 0.5 * mu * (1 - mu) * psi1 * (phi[n - 1, 2:] - phi[n - 1, 1:-1])
    return phi


if __name__ == '__main__':
    u = 0.1
    L = 1
    x0 = 0.5
    r = 0.2
    h = 0.0001
    tau = 0.001
    sigma = 0.05
    x = np.arange(0, L, h)
    t = np.arange(0, (L - x0 - r) / u - 1.5, tau)
    c = np.zeros((t.shape[0], x.shape[0]))
    c[0] = np.where(abs(x0 - x) <= r, 1, 0)
    c1 = tvd(c, u, h, tau, mc)
    c2 = tvd(c, u, h, tau, minmod)
    c3 = tvd(c, u, h, tau, van_leer)
    c4 = tvd(c, u, h, tau, superbee)
    c_lw = lax_wendroff(c, u, h, tau)
    np.save('tvd_1d/data/mc_h{}t{}'.format(h, tau), c1)
    np.save('tvd_1d/data/minmod_h{}t{}'.format(h, tau), c2)
    np.save('tvd_1d/data/superbee_h{}t{}'.format(h, tau), c4)
    np.save('tvd_1d/data/lw_h{}t{}'.format(h, tau), c_lw)
    T = (L - x0 - r) / u - 1.5
    theor = np.where(abs(x - x0 - T * u) <= r, 1, 0)
    np.save('tvd_1d/data/theor_h{}'.format(h), theor)
    plot_results(x, [
        (c4[-1], 'tvd with superbee limiter'),
        (c3[-1], 'tvd with Van-Leer limiter'),
        (c2[-1], 'tvd with minmod limiter'),
        (c1[-1], 'tvd with MC limiter'),
        (c_lw[-1], 'Lax-Wendroff'),
        (theor, 'true solution')])
