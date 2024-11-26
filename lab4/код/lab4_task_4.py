import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split

diabetes = datasets.load_diabetes()
X = diabetes.data
y = diabetes.target

# Поділяємо дані на навчальну та тестову вибірки
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.5, random_state=0)

# Створюємо модель лінійної регресії та натренуємо її
regr = linear_model.LinearRegression()
regr.fit(Xtrain, ytrain)

# Зробимо прогноз по тестовій вибірці
ypred = regr.predict(Xtest)

# Розрахунок коефіцієнтів регресії та різних показників якості
print('Коефіцієнти регресії:', regr.coef_)
print('Перехоплення (intercept):', regr.intercept_)
print('R^2 (коефіцієнт детермінації):', r2_score(ytest, ypred))
print('Середня абсолютна помилка (MAE):', mean_absolute_error(ytest, ypred))
print('Середньоквадратична помилка (MSE):', mean_squared_error(ytest, ypred))

# Побудова графіка
fig, ax = plt.subplots()
ax.scatter(ytest, ypred, edgecolors=(0, 0, 0))
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
ax.set_xlabel('Виміряно')
ax.set_ylabel('Передбачено')
plt.show()
