"""
PW2 Lab B Part 3 -- fit a reaction's rate constant to measured data.

A first-order reaction decays as  C(t) = C0 * exp(-k*t).  You have noisy
concentration-vs-time measurements; find the k that best matches them.
Complete the TODOs. Run:  python kinetics.py
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize



data = np.loadtxt("kinetics.csv", delimiter=",", skiprows=1)
time = data[:, 0]
conc = data[:, 1]
c0 = conc[0]



def total_error(k):
    pred = c0 * np.exp(-k * time)
    err = np.sum((conc - pred)**2)
    return err



res = minimize(total_error, 0.5, method="SLSQP", bounds=[(0, 5)])
best_k = res.x[0]

print("k predicted as:", best_k)



plt.figure()
plt.plot(time, conc, 'o', label="data")
plt.plot(time, c0 * np.exp(-best_k * time), label="fit")
plt.xlabel("time")
plt.ylabel("concentration")
plt.legend()
plt.savefig("kinetics.png")
