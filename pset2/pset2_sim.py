import numpy as np
import matplotlib.pyplot as plt

def f(k, l, a, b, x, y):
    return k*x - a*x*y, -l*y + b*x*y

def euler(f, k, l, a, b, x0, y0, dt, T):
    n = int(T / dt)
    x, y = np.zeros(n), np.zeros(n)
    x[0], y[0] = x0, y0

    for i in range(n-1):
        dx, dy = f(k, l, a, b, x[i], y[i])
        x[i+1] = x[i] + dt * dx
        y[i+1] = y[i] + dt * dy

    return x, y

def rk(f, k, l, a, b, x0, y0, dt, T):
    n = int(T / dt)
    x, y = np.zeros(n), np.zeros(n)
    x[0], y[0] = x0, y0

    for i in range(n-1):
        xk1, yk1 = f(k, l, a, b, x[i], y[i])
        xk2, yk2 = f(k, l, a, b, x[i] + 0.5*dt*xk1, y[i] + 0.5*dt*yk1)
        xk3, yk3 = f(k, l, a, b, x[i] + 0.5*dt*xk2, y[i] + 0.5*dt*yk2)
        xk4, yk4 = f(k, l, a, b, x[i] + dt*xk3, y[i] + dt*yk3)

        x[i+1] = x[i] + (dt/6)*(xk1 + 2*xk2 + 2*xk3 + xk4)
        y[i+1] = y[i] + (dt/6)*(yk1 + 2*yk2 + 2*yk3 + yk4)

    return x, y

def plot(rk, euler, f, k, l, a, b, dt, T, initials):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for x0, y0 in initials:
        x, y = euler(f, k, l, a, b, x0, y0, dt, T)
        axes[0].plot(x, y)

    axes[0].set_title("Euler Method")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")

    for x0, y0 in initials:
        x, y = rk(f, k, l, a, b, x0, y0, dt, T)
        axes[1].plot(x, y)

    axes[1].set_title("Runge–Kutta 4 Method")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    dt = 0.01
    initials = [
        (3,3),
        (3,6),
        (6,9),
        (6,7),
        (1,2),
        (2,3),
        (4,5),
        (4,9),
        (5,1)
        ]
    k, l, a, b = 1.2, 1.0, 0.5, 0.4
    T = 10
    plot(rk, euler, f, k, l, a, b, dt, T, initials)
