import numpy as np
import matplotlib.pyplot as plt

# Parameters
dt = 0.1
tau_v = 0.6
v_f_cmd = 1.0

# Discrete-time filter coefficient
alpha = np.exp(-dt / tau_v)

# Simulation over 100 steps
steps = 200
vf = 0.0
vf_history = np.zeros(steps + 1)
time = np.linspace(0, steps*dt, steps+1)

for i in range(1, steps + 1):
    vf = alpha * vf + (1.0 - alpha) * v_f_cmd
    vf_history[i] = vf

# Plot
plt.figure()
plt.plot(time, vf_history)
plt.xlabel("Time [s]")
plt.ylabel("vf [m/s]")
plt.title("Discrete-time First-order Response\n(?=0.6?s, dt=0.1?s, step to 1?m/s)")
plt.grid(True)
plt.show()
