# Імпортуємо необхідні бібліотеки
from sklearn.cluster import KMeans  # Для кластеризації методом k-means
from sklearn.datasets import load_iris  # Для завантаження набору даних Iris
import matplotlib.pyplot as plt  # Для візуалізації
import numpy as np  # Для роботи з масивами

# Завантажуємо набір даних Iris
iris = load_iris()
X = iris['data']  # Вибираємо дані (4 атрибути для кожного зразка)
y = iris['target']  # Вибираємо цільові значення (класи квітів)

# Створюємо об'єкт KMeans
# Вказуємо кількість кластерів (n_clusters=3, оскільки маємо три класи квітів)
kmeans = KMeans(n_clusters=3, init='k-means++', n_init=10, max_iter=300, random_state=42)

# Навчаємо модель на даних X
kmeans.fit(X)

# Прогнозуємо мітки кластерів для всіх точок у наборі даних
y_kmeans = kmeans.predict(X)

# Візуалізація результатів
# Відображаємо точки даних на площині за двома першими ознаками (довжина та ширина чашолистка)
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')

# Відображаємо центроїди кластерів
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75, label='Centroids')

# Додаємо легенду та підписи до осей
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.title('K-means Clustering of Iris Dataset')
plt.legend(['Clusters', 'Centroids'])
plt.grid(True)
plt.show()
