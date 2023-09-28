import numpy as np
import scipy.interpolate as interpolate

def log_mean_squared_error_fun(y,ypred):
    LMSE = np.mean((np.log(y)-np.log(ypred))**2)
    return LMSE
#generate the training set
a, b = 0, 2 * np.pi
n = 100
x = np.random.uniform(a, b, n)
y = np.sin(x)

#building the Lagrange interpolation model
lagrange_model = interpolate.lagrange(x, y)

#generate the test set
x_test = np.random.uniform(a, b, n)
y_test = np.sin(x_test)

#compute train and test errors
y_train_predicted = lagrange_model(x)
train_error = log_mean_squared_error_fun(y, y_train_predicted)
y_test_predicted = lagrange_model(x_test)
test_error = log_mean_squared_error_fun(y_test, y_test_predicted)

print("Train Error:", train_error)
print("Test Error:", test_error)

#Gaussian noise
std_devs = [0.1, 0.5, 1.0]  #standard deviations for noise

for std_dev in std_devs:
    #add Gaussian noise 
    x_noisy = x + np.random.normal(0, std_dev, n)
    y_noisy = np.sin(x_noisy)
    x_test_noisy = x_test + np.random.normal(0, std_dev, n)
    y_test_noisy = np.sin(x_test_noisy)

    #build the Lagrange interpolation model
    lagrange_model_noisy = interpolate.lagrange(x_noisy, y_noisy)

    # Error calculation
    y_train_predicted_noisy = lagrange_model_noisy(x_noisy)
    train_error_noisy = log_mean_squared_error_fun(y_noisy, y_train_predicted_noisy)
    y_test_predicted_noisy = lagrange_model_noisy(x_test_noisy)
    test_error_noisy = log_mean_squared_error_fun(y_test_noisy, y_test_predicted_noisy)

    print(f"Train Error (Noise Std Dev = {std_dev}):", train_error_noisy)
    print(f"Test Error (Noise Std Dev = {std_dev}):", test_error_noisy)
