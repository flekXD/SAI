import matplotlib.pyplot as plt

def visualize_classifier(classifier, X, y, title=None):
    # Проводимо класифікацію
    plt.figure()
    # Точки даних, припустимо, ви хочете показати їх на графіку
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu)
    if title:
        plt.title(title)  # Додаємо заголовок, якщо він переданий
    plt.show()