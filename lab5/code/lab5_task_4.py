import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_squared_error, explained_variance_score
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# Завантаження набору даних California Housing
housing_data = fetch_california_housing()

# Перемішування даних
X, y = shuffle(housing_data.data, housing_data.target, random_state=7)

# Розбиття даних на навчальний та тестовий набори
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)

# Створення та навчання регресора AdaBoost з використанням дерева рішень
regressor = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4), n_estimators=400, random_state=7)
regressor.fit(X_train, y_train)

# Оцінка ефективності регресора
y_pred = regressor.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
evs = explained_variance_score(y_test, y_pred)

print("\nADABOOST REGRESSOR")
print("Mean squared error:", round(mse, 2))
print("Explained variance score:", round(evs, 2))

# Вилучення важливості ознак
feature_importances = regressor.feature_importances_
feature_names = housing_data.feature_names

# Нормалізація значень важливості ознак
feature_importances = 100.0 * (feature_importances / max(feature_importances))

# Сортування важливості ознак
index_sorted = np.flipud(np.argsort(feature_importances))

# Розміщення міток уздовж осі X для побудови стовпчастої діаграми
pos = np.arange(index_sorted.shape[0]) + 0.5

# Переконайтесь, що ви використовуєте правильний формат для міток
sorted_feature_names = [feature_names[i] for i in index_sorted]

plt.figure()
plt.bar(pos, feature_importances[index_sorted], align='center')
plt.xticks(pos, sorted_feature_names, rotation=90)  # Використовуємо правильно відсортовані назви ознак
plt.ylabel('Relative Importance')
plt.title('Оцінка важливості ознак з використанням регресора AdaBoost')
plt.show()
