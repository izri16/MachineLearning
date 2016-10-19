import numpy as np
import scipy.stats as st
import sklearn.linear_model as lm
import matplotlib.pyplot as plt

# matplotlib.pyplot makes matplotlib work like matlab
# be aware that python uses half open intervals

# clear all figures
plt.close("all")

# e^x or e^x for each x in array
f = lambda x: np.exp(3*x)

# Return evenly spaced numbers over a specified interval.
# evenly == rovnomerne
# start, stop, total_count
x_tr = np.linspace(0., 2, 200)
X_tr = np.vstack([np.ones(len(x_tr)), x_tr]).T

# y values will be exponencial
# x_tr and y_tr are training data
y_tr = f(x_tr)

# generate data points
# .1 is shortcut for 0.1
x = np.array([0, .1, .2, .5, .8, .9, 1])
A = np.vstack([np.ones(len(x)), x]).T

# we add gaussian nose
y = (f(x) + np.random.randn(len(x)))

# create the model
lr = lm.LinearRegression()

# We train the model on our training dataset.
lr.fit(A, y)

theta_values = lr.coef_
print('theta values', theta_values)

# Now, we predict points with our trained model.
y_lr = lr.predict(X_tr)

# 3rd argument of plot is color and line type
# b-        blue solid line
# ro        red circles
plt.figure(1, figsize=(6,3))
plt.subplot(111)
plt.plot(x_tr, y_tr, '--k', linewidth=2.0, label='Real data')
plt.plot(x_tr, y_lr, 'g', label='Fitted from training')
plt.plot(x, y, 'ok', ms=10, label='Training') # ms = markersize
plt.legend(loc=2)
#plt.xlim(0, 1) # limit for x-axis
#plt.ylim(y.min()-1, y.max()+1) # limit for y-axis
plt.title('Regression')
plt.ylabel('Y-values')
plt.xlabel('X-values')
#plt.axis([0, 6, 0, 20])
plt.show()

# Using polynomial regression
# Vandermonde matrix to add polynomials
plt.figure(2)
plt.plot(x_tr, y_tr, '--k');
for deg, s in zip([2, 5], ['-', '.']):
    lr.fit(np.vander(x, deg + 1), y)
    y_lrp = lr.predict(np.vander(x_tr, deg + 1))
    plt.plot(x_tr, y_lrp, s, label='degree ' + str(deg))
    plt.legend(loc=2)
    plt.xlim(0, 1.4)
    plt.ylim(-10, 40)
plt.plot(x, y, 'ok', ms=10)
plt.title("Linear regression")
plt.show()

# Using Ridge class we set regularization term by ourselves
# Using RigdeCV in set regularization term based on cross-validation
ridge = lm.RidgeCV()
plt.figure(figsize=(6,3));
plt.plot(x_tr, y_tr, '--k');

for deg, s in zip([2, 5], ['-', '.']):
    ridge.fit(np.vander(x, deg + 1), y);
    y_ridge = ridge.predict(np.vander(x_tr, deg + 1))
    plt.plot(x_tr, y_ridge, s, label='degree ' + str(deg));
    plt.legend(loc=2);
    plt.xlim(0, 1.5);
    plt.ylim(-5, 80);
plt.plot(x, y, 'ok', ms=10);
plt.title("Ridge regression");
plt.show()

