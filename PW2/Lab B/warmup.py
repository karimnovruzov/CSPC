"""
PW2 Lab B Part 2 -- three routes to a minimum.

Compare gradient descent, Newton, and SLSQP on two functions:
  2A: f(x) = (x-3)**2 + 1          (easy, one minimum at x=3)
  2B: g(x) = x**4 - 3*x**2 + x + 5 (harder, several stationary points)
Run:  python warmup.py
"""
import numpy as np
from scipy.optimize import newton, minimize



def f(x):   return (x-3)**2 + 1
def df(x):  return 2*(x-3)
def d2f(x): return 2.0

import numpy as np
from scipy.optimize import minimize, newton

def f(x):

    return (x - 3)**2 + 1


def df(x):

    return 2*x - 6

def d2f(x):

    return 2


x = 0
step = 0.1
for i in range(100):
    x = x - step * df(x)
print("gd result:", x)


res_newton = newton(df, 0, fprime=d2f)
print("newton result:", res_newton)



res_slsqp = minimize(f, 0, method="SLSQP")
print("slsqp result:", res_slsqp.x[0])


def g(x):   return x**4 - 3*x**2 + x + 5
def dg(x):  return 4*x**3 - 6*x + 1
def d2g(x): return 12*x**2 - 6


def g(x):
    return x**4 - 3*x**2 + x + 5

def dg(x):
    return 4*x**3 - 6*x + 1

def d2g(x):
    return 12*x**2 - 6


for x0 in [0, 2]:
    print(" x0 =", x0)
    
    
    x = float(x0)
    step = 0.01
    for i in range(1000):
        x = x - step * dg(x)
    print("gd:", x)



    
    res_newt = newton(dg, x0, fprime=d2g)
    second_deriv = d2g(res_newt)
    
    if second_deriv > 0:
        kind = "min"
    else:
        kind = "max"
    print("newton:", res_newt, "category:", kind)



    res_slsqp = minimize(g, x0, method="SLSQP")
    print("slsqp:", res_slsqp.x[0])
