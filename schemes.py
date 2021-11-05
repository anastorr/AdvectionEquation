import numpy as np


def direction(u):
    return np.where(u>0, u, 0)


def upwind(c, tau, h, u, v):
    for n in range(1, c.shape[0]):
        c[n, 1:-1, 1:-1] = (c[n-1, 1:-1, 1:-1] - tau*direction(v[1:-1, 1:-1])*(c[n-1, 1:-1, 1:-1]-c[n-1, :-2, 1:-1])/h
                            + tau*direction(-v[1:-1, 1:-1])*(c[n-1, 2:, 1:-1]-c[n-1, 1:-1, 1:-1])/h
                            - tau*direction(u[1:-1, 1:-1])*(c[n-1, 1:-1, 1:-1]-c[n-1, 1:-1, :-2])/h
                            + tau*direction(-u[1:-1, 1:-1])*(c[n-1, 1:-1, 2:]-c[n-1, 1:-1, 1:-1])/h)
    return c


def leapfrog(c, tau, h, u, v):
    c[1] = upwind(c[:2, ...], tau, h, u, v)[1]
    for n in range(2, c.shape[0]):
        c[n, 1:-1, 1:-1] = (c[n-2, 1:-1, 1:-1] - v[1:-1, 1:-1]*tau/h*(c[n-1, 2:, 1:-1] - c[n-1, :-2, 1:-1])
                            - u[1:-1, 1:-1]*tau/h*(c[n-1, 1:-1, 2:] - c[n-1, 1:-1, :-2]))
    return c


def lax_wendroff(c, tau, h, u, v):
    for n in range(1, c.shape[0]):
        c[n, 1:-1, 1:-1] = (c[n-1, 1:-1, 1:-1] - v[1:-1, 1:-1]*tau/2/h*(c[n-1, 2:, 1:-1] - c[n-1, :-2, 1:-1])
                            + v[1:-1, 1:-1]**2*tau**2/2/h**2*(c[n-1, 2:, 1:-1] - 2*c[n-1, 1:-1, 1:-1] + c[n-1, :-2, 1:-1])
                            - u[1:-1, 1:-1]*tau/2/h*(c[n-1, 1:-1, 2:] - c[n-1, 1:-1, :-2])
                            + u[1:-1, 1:-1]**2*tau**2/2/h**2*(c[n-1, 1:-1, 2:] - 2*c[n-1, 1:-1, 1:-1] + c[n-1, 1:-1, :-2])
                            + u[1:-1, 1:-1]*v[1:-1, 1:-1]*tau**2/4/h**2*(c[n-1, 2:, 2:] - c[n-1, :-2, 2:] - c[n-1, 2:, :-2]
                                                                         + c[n-1, :-2, :-2]))
    return c


def corner_transport_upwind(c, tau, h, u, v):
    for n in range(1, c.shape[0]):
        c[n, 1:-1, 1:-1] = (c[n-1, 1:-1, 1:-1] - tau*direction(v[1:-1, 1:-1])*(c[n-1, 1:-1, 1:-1] - c[n-1, :-2, 1:-1])/h
                            + tau * direction(-v[1:-1, 1:-1]) * (c[n - 1, 2:, 1:-1] - c[n - 1, 1:-1, 1:-1]) / h
                            - tau * direction(u[1:-1, 1:-1]) * (c[n - 1, 1:-1, 1:-1] - c[n - 1, 1:-1, :-2]) / h
                            + tau * direction(-u[1:-1, 1:-1]) * (c[n - 1, 1:-1, 2:] - c[n - 1, 1:-1, 1:-1]) / h
                            + tau**2*u[1:-1, 1:-1]*v[1:-1, 1:-1]*(c[n-1, 1:-1, 1:-1] - c[n-1, :-2, 1:-1]
                                                                  - c[n-1, 1:-1, :-2] + c[n-1, :-2, :-2])/h**2)
    return c


def flux_limiter(c, tau, h, u, v):
    for n in range(1, c.shape[0]):
        f = np.zeros(c.shape[1:])
        g = np.zeros(c.shape[1:])
