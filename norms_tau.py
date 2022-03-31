import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from advection_1d import lax_wendroff, minmod, van_leer, mc, superbee, tvd
from norms import err_c, err_l2

mpl.use('Qt5Agg')


def error_from_tau_tvd_1d(c0, theor, x, u, h, tau_list, T, limiter):
    errors_c = np.zeros(tau_list.size)
    errors_l2 = np.zeros(tau_list.size)
    for i in range(tau_list.size):
        t = np.arange(0, T, tau_list[i])
        c = np.zeros((t.shape[0], x.shape[0]))
        c[0] = c0
        c = tvd(c, u, h, tau_list[i], limiter)
        print('tau {}: Success!'.format(i))
        errors_c[i] = err_c(c[-1], theor)
        errors_l2[i] = err_l2(c[-1], theor)
    return errors_c, errors_l2


def error_from_tau_lw(c0, theor, x, u, h, tau_list, T):
    errors_c = np.zeros(tau_list.size)
    errors_l2 = np.zeros(tau_list.size)
    for i in range(tau_list.size):
        t = np.arange(0, T, tau_list[i])
        c = np.zeros((t.shape[0], x.shape[0]))
        c[0] = c0
        c = lax_wendroff(c, u, h, tau_list[i])
        print('tau {}: Success!'.format(i))
        errors_c[i] = err_c(c[-1], theor)
        errors_l2[i] = err_l2(c[-1], theor)
    return errors_c, errors_l2


if __name__  == "__main__":
    u = 0.1
    L = 1
    x0 = 0.5
    r = 0.1
    h = 0.0025
    T = (L - x0 - r) / u - 1.5
    tau_list = np.linspace(0.00001, 0.05, 100)
    x = np.arange(0, L, h)
    c0 = np.where(abs(x0 - x) <= r, 1, 0)
    theor = np.where(abs(x - x0 - T * u) <= r, 1, 0)
    errors_c, errors_l2 = error_from_tau_tvd_1d(c0, theor, x, u, h, tau_list, T, minmod)
    plt.plot(1/tau_list, errors_l2)
    plt.scatter(1/tau_list, errors_l2)
    plt.show()
