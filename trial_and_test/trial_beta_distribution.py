import numpy as np
import matplotlib.pyplot as plt

data = []

for i in range(10000):
    data.append(np.random.beta(0.1, 0.9))
    
plt.figure()
plt.hist(data, bins=100)
plt.show()