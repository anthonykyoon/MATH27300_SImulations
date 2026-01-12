import pandas as pd
import math


def euler_method(f, x0, t0, T, h):
    """
    f: f(t, x)
    x0 : initial value (can be vector or scalar)
    t0: initial time
    T: final time
    h: step size

    Returns ONLY the final value x(T).
    Uses only two pointers (t_n, x_n).
    """
    t = t0
    x = x0

    while t < T:
        x = x + h * f(t, x)
        t = t + h

    return x

def midpoint_method(f, x0, t0, T, h):
    """
    f: function f(t, x)
    x0 : initial value (scalar or vector)
    t0 : initial time
    T : final time
    h : step size

    Returns ONLY the final value x(T).
    Uses only two pointers (t_n, x_n).
    """
    t = t0
    x = x0

    while t < T:
        k1 = f(t, x)
        k2 = f(t + h / 2, x + h / 2 * k1)
        x = x + h * k2
        t = t + h

    return x


def runge_kutta(f, x0, t0, T, h):
    """
    f: f(t, x)
    x0: initial value
    t0: initial time
    T: final time
    h: time step

    Returns ONLY the final value x(T).
    Uses only two pointers (t_n, x_n).
    """
    t = t0
    x = x0

    while t < T:
        k1 = f(t, x)
        k2 = f(t + h / 2, x + h / 2 * k1)
        k3 = f(t + h / 2, x + h / 2 * k2)
        k4 = f(t + h, x + h * k3)

        x = x + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        t = t + h

    return x


def foc(t, x):
    """
    ODE x' = x.
    """
    return x


def make_table(f, method, x0, t0, T):
    """
    Constructs a table for the convergence experiment.

    Column 1: N = 2^k, where k = 10, ..., 20 and h = 1/N
    Column 2: numerical approximation of x(1)
    Column 3: absolute error |x(1) - e|
    """
    rows = []

    for k in range(10, 21):
        N = 2 ** k
        h = 1 / N

        x_num = method(f, x0, t0, T, h)
        err = abs(x_num - math.e)

        rows.append((k, x_num, err))

    return pd.DataFrame(
        rows,
        columns=["N = 2^k", "Numerical x(1)", "Error |x(1) - e|"]
    )

if __name__ == "__main__":
    print("Simulating \dot\{x\} = x where solution is equal to x(t) = e^t")
    print_out = input("Do you want tables outputted in terminal (Y or N):   ")
    csv = input("Do you want export as CSV? The CSVs will be saved in this directory (Y or N):    ")
    csv = csv.lower()
    print_out = print_out.lower()

    if csv != "y" and print_out != "y":
        raise ValueError("Did not want anything to be returned ")

    x0 = 1.0
    t0 = 0.0
    T = 1.0

    table_euler = make_table(foc, euler_method, x0, t0, T)
    table_midpoint = make_table(foc, midpoint_method, x0, t0, T)
    table_rk = make_table(foc, runge_kutta, x0, t0, T)

    if csv == "y" and print_out == "y":
        print("Euler Method")
        print(table_euler)
        print("\nMidpoint Method")
        print(table_midpoint)
        print("\nRunge–Kutta 4")
        print(table_rk)
        print("exporting tables as CSVs")
        table_euler.to_csv("euler_table.csv", index = False, float_format = "%.12e")
        table_midpoint.to_csv("midpoint_table.csv", index = False, float_format="%.12e")
        table_rk.to_csv("range_kuta.csv", index = False, float_format="%.12e" )
    elif csv == "y" and print_out != "y":
        print("exporting tables as CSVs")
        table_euler.to_csv("euler_table.csv", index = False, float_format = "%.12e")
        table_midpoint.to_csv("midpoint_table.csv", index = False, float_format="%.12e")
        table_rk.to_csv("range_kuta.csv", index = False, float_format="%.12e" )
    elif csv != "y" and print_out == "y":
        print("Euler Method")
        print(table_euler)
        print("\nMidpoint Method")
        print(table_midpoint)
        print("\nRunge–Kutta 4")
        print(table_rk)