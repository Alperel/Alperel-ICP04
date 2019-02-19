from sklearn import svm,datasets
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split

#Question 1
iris = datasets.load_iris()
x = iris.data
y = iris.target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.4, random_state = 0)

model = GaussianNB()
model.fit(x_train,y_train)
print('Accuracy of Naive Bayes classifier on test set: %.2f' % (model.score(x_test,y_test)).round(2))

#Question 2
linear_svm = svm.SVC(kernel = 'linear', C = 1).fit(x_train, y_train)
print('Accuracy of SVM classifier with linear kernel on training test set: %.2f' % (linear_svm.score(x_test, y_test)).round(2))

#Question 3
RBF_svm = svm.SVC(kernel = 'rbf', C = 1).fit(x_train, y_train)
print('Accuracy of SVM classifier with RBF kernel on training test set: %.2f' % (RBF_svm.score(x_test, y_test)).round(2))