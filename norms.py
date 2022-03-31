import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.use('Qt5Agg')


def err_l2(c, c_analyt):
    return (np.sum((c - c_analyt)**2))**0.5/np.size(c)


def err_c(c, c_analyt):
    return np.max(np.abs(c-c_analyt))


tau = 0.001
method = 'lw'

c1 = np.load('tvd_1d/data/{}_h0.0001t{}.npy'.format(method, tau))
c2 = np.load('tvd_1d/data/{}_h0.00025t{}.npy'.format(method, tau))
c3 = np.load('tvd_1d/data/{}_h0.0005t{}.npy'.format(method, tau))
c4 = np.load('tvd_1d/data/{}_h0.00075t{}.npy'.format(method, tau))
c5 = np.load('tvd_1d/data/{}_h0.001t{}.npy'.format(method, tau))


theor1 = np.load('tvd_1d/data/theor_h0.0001.npy')
theor2 = np.load('tvd_1d/data/theor_h0.00025.npy')
theor3 = np.load('tvd_1d/data/theor_h0.0005.npy')
theor4 = np.load('tvd_1d/data/theor_h0.00075.npy')
theor5 = np.load('tvd_1d/data/theor_h0.001.npy')

d1 = err_l2(c1[-1], theor1)
d2 = err_l2(c2[-1], theor2)
d3 = err_l2(c3[-1], theor3)
d4 = err_l2(c4[-1], theor4)
d5 = err_l2(c5[-1], theor5)

plt.plot(np.log(np.array([0.0001, 0.00025, 0.0005, 0.00075, 0.001])), np.log(np.array([d1, d2, d3, d4, d5])))
plt.scatter(np.log(np.array([0.0001, 0.00025, 0.0005, 0.00075, 0.001])), np.log(np.array([d1, d2, d3, d4, d5])))
# plt.scatter(np.array([0.001, 0.0025, 0.005, 0.01, 0.02]), np.array([d1, d2, d3, d4, d5]))
plt.show()


# c1 = np.load('leapfrog/data/h100t5.npy')
# c2 = np.load('leapfrog/data/h200t5.npy')
# c3 = np.load('leapfrog/data/h400t5.npy')
#
#
# d1 = div_c(c1[-1], c1[0])
# d2 = div_c(c2[-1], c2[0])
# d3 = div_c(c3[-1], c3[0])
#
# plt.plot(np.array([100, 200, 400]), np.array([d1, d2, d3]))
# plt.show()