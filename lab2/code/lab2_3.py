from sklearn.datasets import load_iris

# Завантаження набору даних
iris_dataset = load_iris()

# Ключі об'єкта iris_dataset
print("Ключі iris_dataset: \n{}".format(iris_dataset.keys()))

# Опис набору даних
print("Опис набору даних:\n", iris_dataset['DESCR'][:193] + "\n...")

# Назви відповідей (класів)
print("Назви відповідей: {}".format(iris_dataset['target_names']))

# Назва ознак
print("Назва ознак: \n{}".format(iris_dataset['feature_names']))

# Тип масиву data
print("Тип масиву data: {}".format(type(iris_dataset['data'])))

# Форма масиву data
print("Форма масиву data: {}".format(iris_dataset['data'].shape))

# Виведення значень ознак для перших п'яти прикладів
print("Перші п'ять прикладів:\n{}".format(iris_dataset['data'][:5]))

# Тип масиву target
print("Тип масиву target: {}".format(type(iris_dataset['target'])))

# Виведення цільових значень
print("Відповіді:\n{}".format(iris_dataset['target']))
