"""
PW2 Lab B Part 4 -- chemical equilibrium via the equilibrium constant K.

Reaction  H2 + I2 <=> 2 HI, starting from 1 mol H2 and 1 mol I2.
As the reaction proceeds by an extent x:  H2 = 1-x,  I2 = 1-x,  HI = 2x.
At equilibrium the composition satisfies the equilibrium constant
        K = [HI]^2 / ([H2][I2]) = (2x)^2 / ((1-x)(1-x)).
Given K, find the extent x. Solve it TWO ways and compare.
Run:  python equilibrium.py
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import newton, minimize

K = 15.6
a = b = 1.0

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, newton



K = 16.0 

def k_imbalance(x):
 
    return ((2*x)**2) / ((1-x)*(1-x)) - K

x_newton = newton(k_imbalance, 0.5)

def obj_func(x):
    return k_imbalance(x)**2


res_slsqp = minimize(obj_func, 0.5, method="SLSQP", bounds=[(0, 0.99)])
x_slsqp = res_slsqp.x[0]

print("Newton x:", x_newton)
print("SLSQP x:", x_slsqp)



h2_eq = 1 - x_slsqp
i2_eq = 1 - x_slsqp
hi_eq = 2 * x_slsqp

print("H2 eq:", h2_eq)
print("I2 eq:", i2_eq)
print("HI eq:", hi_eq)


x_vals = np.linspace(0, 0.9, 100)
h2_vals = 1 - x_vals
i2_vals = 1 - x_vals
hi_vals = 2 * x_vals

plt.figure()
plt.plot(x_vals, h2_vals, label="H2")
plt.plot(x_vals, i2_vals, label="I2")
plt.plot(x_vals, hi_vals, label="HI")
plt.axvline(x=x_slsqp, color='r', linestyle='--', label="Equilibrium")
plt.xlabel("Extent x")
plt.ylabel("Moles")
plt.legend()
plt.savefig("equilibrium.png")