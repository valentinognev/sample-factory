import json
import numpy as np
import matplotlib.pyplot as plt

# 1) Load your JSON
with open('recorded_actions.json', 'r') as f:
    actions = json.load(f)

# 2) Turn into a NumPy array
arr = np.array(actions)

# 3) Flatten any extra dimensions so you end up with shape (n_steps, n_dims)
#    This works no matter how many extra singleton or agent dims you had.
arr = arr.reshape(arr.shape[0], -1)

# 4) Now unpack safely
n_steps, n_dims = arr.shape
time = np.arange(n_steps)

# 5) Plot each action component vs. time
for i in range(n_dims):
    plt.figure()
    plt.plot(time, arr[:, i])
    plt.title(f'Action Dimension {i} Over Time')
    plt.xlabel('Step')
    plt.ylabel(f'Action[{i}]')
    plt.tight_layout()

plt.show()
