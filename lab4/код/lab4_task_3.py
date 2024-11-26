# Файл: LR_3_task_3.py

import numpy as np
from sklearn import linear_model
import sklearn.metrics as sm
from sklearn.preprocessing import PolynomialFeatures

# Вхідний файл з даними
input_file = 'data_multivar_regr.txt'

# Завантаження даних
try:
    data = np.loadtxt(input_file, delimiter=',')
except OSError:
    raise FileNotFoundError(f"Файл {input_file} не знайдено. Переконайтесь, що файл існує.")

X, y = data[:, :-1], data[:, -1]

# Розбивка даних на навчальний та тестовий набори
num_training = int(0.8 * len(X))
num_test = len(X) - num_training

X_train, y_train = X[:num_training], y[:num_training]
X_test, y_test = X[num_training:], y[num_training:]

# Лінійна регресія
linear_regressor = linear_model.LinearRegression()
linear_regressor.fit(X_train, y_train)
y_test_pred = linear_regressor.predict(X_test)

# Метрики для лінійного регресора
print("Linear Regressor performance:")
print("Mean absolute error =", round(sm.mean_absolute_error(y_test, y_test_pred), 2))
print("Mean squared error =", round(sm.mean_squared_error(y_test, y_test_pred), 2))
print("Median absolute error =", round(sm.median_absolute_error(y_test, y_test_pred), 2))
print("Explained variance score =", round(sm.explained_variance_score(y_test, y_test_pred), 2))
print("R2 score =", round(sm.r2_score(y_test, y_test_pred), 2))

# Поліноміальна регресія
polynomial = PolynomialFeatures(degree=10)
X_train_transformed = polynomial.fit_transform(X_train)
X_test_transformed = polynomial.transform(X_test)

poly_linear_model = linear_model.LinearRegression()
poly_linear_model.fit(X_train_transformed, y_train)
y_test_pred_poly = poly_linear_model.predict(X_test_transformed)

# Прогноз для вибіркової точки
datapoint = [[7.75, 6.35, 5.56]]
poly_datapoint = polynomial.transform(datapoint)

print("\nLinear regression prediction for datapoint:\n", linear_regressor.predict(datapoint))
print("\nPolynomial regression prediction for datapoint:\n", poly_linear_model.predict(poly_datapoint))

# Порівняння метрик для поліноміальної регресії
print("\nPolynomial Regressor performance:")
print("Mean absolute error =", round(sm.mean_absolute_error(y_test, y_test_pred_poly), 2))
print("Mean squared error =", round(sm.mean_squared_error(y_test, y_test_pred_poly), 2))
print("Median absolute error =", round(sm.median_absolute_error(y_test, y_test_pred_poly), 2))
print("Explained variance score =", round(sm.explained_variance_score(y_test, y_test_pred_poly), 2))
print("R2 score =", round(sm.r2_score(y_test, y_test_pred_poly), 2))
