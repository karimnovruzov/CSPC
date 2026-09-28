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







#BONUS TASK: 2D Trajectory Analysis 


# 1. Read trajectory.csv (columns: time, x, y)



traj_data = np.loadtxt('trajectory.csv', delimiter=',', skiprows=1)
t_2d = traj_data[:, 0]
x = traj_data[:, 1]
y_2d = traj_data[:, 2]

# 2. Compute velocity in x and y directions using np.gradient


vx = np.gradient(x, t_2d)

vy = np.gradient(y_2d, t_2d)

# Compute overall speed sqrt(vx^2 + vy^2)


speed = np.sqrt(vx**2 + vy**2)

# 3. Plot the 2D path (x vs y) and speed over time


fig_bonus, (ax_path, ax_speed) = plt.subplots(1, 2, figsize=(12, 5))

# Path plot (x vs y)


ax_path.plot(x, y_2d, label='2D Path', color='purple')
ax_path.set_xlabel('X Position (m)')
ax_path.set_ylabel('Y Position (m)')
ax_path.set_title('Object Trajectory (X vs Y)')
ax_path.grid(True)
ax_path.legend()

# Speed plot


ax_speed.plot(t_2d, speed, label='Speed', color='green')
ax_speed.set_xlabel('Time (s)')
ax_speed.set_ylabel('Speed (m/s)')
ax_speed.set_title('Speed over Time')
ax_speed.grid(True)
ax_speed.legend()

plt.tight_layout()
plt.savefig('trajectory.png')
print("trajectory.png successfully created!")