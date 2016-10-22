import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# y = ax + b

x = np.array([0, 1, 2, 3])
y = np.array([2, 3, 5, 5])

A = np.vstack([x, np.ones(len(x))]).T

a, b = np.linalg.lstsq(A, y)[0]

unknown = 4
print(a*unknown + b)

'''
plt.plot(x, y, '.', label='Original data', markersize=10)
plt.plot(x, a*x + b, '-', label='Fitted line')
plt.ylabel('Price')
plt.xlabel('Distance')
plt.axis([0, 7, 0, 7])
plt.legend()
plt.show()
'''

dummies = pd.get_dummies([7, 2, 8, 3, 3, 7, 1])

dummies2 = pd.get_dummies(['jozo', 'fero', 'stevo'])

print(dummies)
print(dummies2)

