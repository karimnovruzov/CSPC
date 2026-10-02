"""
PW2 Lab B Part 5 (bonus) -- find a titration's equivalence point.

titration.csv holds a titration curve: pH versus the volume of base added.
The equivalence point is the volume where the pH changes fastest (the steep
jump). Numerically, that is where the SLOPE of the pH curve is largest.
Run:  python titration.py
"""
import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt



data = np.loadtxt("titration.csv", delimiter=",", skiprows=1)
v_base = data[:, 0]
ph = data[:, 1]



dp_dv = np.gradient(ph, v_base)



max_idx = np.argmax(dp_dv)
eq_vol = v_base[max_idx]


print("equivalence point volume:", eq_vol, "mL")



plt.figure(figsize=(10, 4))



plt.subplot(1, 2, 1)
plt.plot(v_base, ph, label="pH curve")
plt.axvline(x=eq_vol, color='r', linestyle='--', label=f"Eq point ({eq_vol} mL)")
plt.xlabel("Volume of Base (mL)")
plt.ylabel("pH")
plt.legend()


plt.subplot(1, 2, 2)
plt.plot(v_base, dp_dv, color='orange', label="Slope (dpH/dV)")
plt.axvline(x=eq_vol, color='r', linestyle='--', label=f"Max slope ({eq_vol} mL)")
plt.xlabel("Volume of Base (mL)")
plt.ylabel("Slope")
plt.legend()





plt.tight_layout()
plt.savefig("titration.png")
