import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid

# 1. Downloading the data


data = np.loadtxt('freefall.csv', delimiter=',', skiprows=1)
t = data[:, 0] # time
y = data[:, 1] # height

# 2. We are finding the velocity and acceleration using numerical differentiation



v = np.gradient(y, t)
a = np.gradient(v, t)

# We are printing the mean and standard deviation of acceleration
print(f"Mean acceleration: {np.mean(a):.2f} m/s^2")
print(f"Std of acceleration: {np.std(a):.2f} m/s^2")



# 3. We are recovering the position from acceleration using numerical integration


v_rec = cumulative_trapezoid(a, t, initial=0) + v[0]
y_rec = cumulative_trapezoid(v_rec, t, initial=0) + y[0]

# We are calculating the maximum difference between the original and recovered position


max_diff = np.max(np.abs(y - y_rec))

print(f"Max difference in recovered position: {max_diff:.2f} m")

# 4. We are plotting the position, velocity, and acceleration


fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

# First graph - Position



ax1.plot(t, y, label='Position', color='blue')
ax1.set_ylabel('Position (m)')
ax1.grid(True)
ax1.legend()



# Second graph - Velocity


ax2.plot(t, v, label='Velocity', color='orange')
ax2.set_ylabel('Velocity (m/s)')
ax2.grid(True)
ax2.legend()



# Third graph - Acceleration


ax3.plot(t, a, label='Acceleration', color='red')
ax3.axhline(-9.81, color='black', linestyle='--', label='Theoretical g (-9.81)')
ax3.set_xlabel('Time (s)')
ax3.set_ylabel('Acceleration (m/s^2)')
ax3.grid(True)
ax3.legend()

plt.tight_layout()
plt.savefig('motion.png')
print("Graphs saved as 'motion.png'")