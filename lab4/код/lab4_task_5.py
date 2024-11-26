import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Генерація випадкових даних
m = 100
X = np.linspace(-3, 3, m).reshape(-1, 1)  # Розтягування вектору X в матрицю
y = np.sin(X).ravel() + np.random.uniform(-0.5, 0.5, m)  # Генерація y

# Побудова графіка даних
plt.scatter(X, y, color='blue', label='Дані')

# Лінійна регресія
lin_reg = LinearRegression()
lin_reg.fit(X, y)
y_lin_pred = lin_reg.predict(X)
plt.plot(X, y_lin_pred, color='green', label='Лінійна регресія')

# Поліноміальна регресія (степінь 2)
poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X)
poly_reg = LinearRegression()
poly_reg.fit(X_poly, y)
y_poly_pred = poly_reg.predict(X_poly)

# Побудова графіка для поліноміальної регресії
plt.plot(X, y_poly_pred, color='red', label='Поліноміальна регресія')

# Підпис графіків
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()

# Виведення коефіцієнтів
print("Коефіцієнти лінійної регресії:", lin_reg.coef_)
print("Перехоплення лінійної регресії:", lin_reg.intercept_)
print("Коефіцієнти поліноміальної регресії:", poly_reg.coef_)
