import numpy as np
import matplotlib.pyplot as plt

# TODO 1: Read decay_observed.csv into arrays t and observed



data = np.loadtxt('decay_observed.csv', delimiter=',', skiprows=1)
t = data[:, 0]
observed = data[:, 1]

# TODO 2: Set N0 to the first observed value and build analytical curve



LAMBDA = 0.3
N0 = observed[0]
analytical = N0 * np.exp(-LAMBDA * t)

# TODO 3: Make 1x2 subplot with shared x and y axes



fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(10, 5))
ax1.scatter(t, observed, color='blue', label='Observed')
ax1.set_title('Observed Data')
ax1.set_xlabel('Time')
ax1.set_ylabel('Count')
ax1.grid(True)
ax2.plot(t, analytical, color='red', label='Analytical')
ax2.set_title('Analytical Law')
ax2.set_xlabel('Time')
ax2.grid(True)
plt.tight_layout()

# TODO 4: Save figure as figure.png




plt.savefig('figure.png')
print("figure.png created succesfully!")