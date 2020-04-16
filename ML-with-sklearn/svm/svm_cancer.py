import sklearn
from sklearn import datasets
from sklearn import svm

cancer = datasets.load_breast_cancer()

# print(dir(cancer))
# print(cancer.feature_names)
# print(cancer.target_names)


X = cancer.data
y = cancer.target

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2)

print(X_train,y_train)

classes = ['malignant' 'benign']


clf = svm.SVC(kernel='linear')

clf.fit(X_train, y_train)


y_predict = clf.predict(X_test)

acc = sklearn.metrics.accuracy_score(y_test, y_predict)

print(acc)