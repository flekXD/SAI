import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from matplotlib import pyplot as plt

data_path = "income_data.txt"
columns = ['age', 'workclass', 'education', 'education-num', 'marital-status',
           'occupation', 'relationship', 'race', 'sex', 'hours-per-week',
           'native-country', 'income']
dataset = pd.read_csv(data_path, header=None, names=columns)


dataset = pd.get_dummies(dataset, drop_first=True)

X = dataset.drop('income_ >50K', axis=1)  # Вхідні ознаки
y = dataset['income_ >50K']  # Мітки класів

X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)

models = []
models.append(('LR', LogisticRegression(solver='liblinear')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))

results = []
names = []

print("\nРезультати моделей:")
for name, model in models:
    kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

plt.boxplot(results, labels=names)
plt.title('Порівняння алгоритмів класифікації')
plt.show()

model = SVC(gamma='auto') 
model.fit(X_train, Y_train)
predictions = model.predict(X_validation)

print("\nТочність:", accuracy_score(Y_validation, predictions))
print("Матриця помилок:")
print(confusion_matrix(Y_validation, predictions))
print("\nЗвіт про класифікацію:")
print(classification_report(Y_validation, predictions))
