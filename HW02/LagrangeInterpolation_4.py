import numpy as np
import scipy.interpolate as interpolate
import matplotlib.pyplot as plt
from numpy.polynomial.polynomial import Polynomial
def log_mean_squared_error_fun(y,ypred):
    LMSE = np.mean((np.log(y)-np.log(ypred))**2)
    return LMSE

seed_value = 50
np.random.seed(seed_value)

#generate the training set
a, b = np.pi/4,3*np.pi/4
n = 100
x = np.random.uniform(a, b, n)
y = np.sin(x)

fitDegree = 4
#building the Lagrange interpolation model
lagrange_model = interpolate.lagrange(x[0:fitDegree], y[0:fitDegree])

#generate the test set
x_test = np.random.uniform(a, b, n)
y_test = np.sin(x_test)

# plt.figure()
# plt.hist(x)

#compute train and test errors
y_train_predicted = lagrange_model(x)
# y_train_predicted = Polynomial(lagrange_model.coef[::-1])(x)
train_error = log_mean_squared_error_fun(y, y_train_predicted)
y_test_predicted = lagrange_model(x_test)
test_error = log_mean_squared_error_fun(y_test, y_test_predicted)

print("Train Error:", train_error)
print("Test Error:", test_error)

#Gaussian noise
std_devs = [0.01,0.05, 0.2]  #standard deviations for noise

for std_dev in std_devs:
    #add Gaussian noise 
    x_noisy = x + np.random.normal(0, std_dev, n)
    y_noisy = np.sin(x_noisy)
    x_test_noisy = x_test + np.random.normal(0, std_dev, n)
    y_test_noisy = np.sin(x_test_noisy)

    #build the Lagrange interpolation model
    lagrange_model_noisy = interpolate.lagrange(x_noisy[0:fitDegree], y_noisy[0:fitDegree])

    # Error calculation
    y_train_predicted_noisy = lagrange_model_noisy(x_noisy)
    train_error_noisy = log_mean_squared_error_fun(y_noisy, y_train_predicted_noisy)
    y_test_predicted_noisy = lagrange_model_noisy(x_test_noisy)
    test_error_noisy = log_mean_squared_error_fun(y_test_noisy, y_test_predicted_noisy)

    print(f"Train Error (Noise Std Dev = {std_dev}):", train_error_noisy)
    print(f"Test Error (Noise Std Dev = {std_dev}):", test_error_noisy)
