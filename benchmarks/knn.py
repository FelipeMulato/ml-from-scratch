from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier as KNN
from src.models.KNN import KNN as myKNN
from src.utils.metric import acurracy

iris = datasets.load_breast_cancer()

X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X,y, train_size=0.8, random_state=1234)

predictions1 =  KNN(n_neighbors=3).fit(X_train,y_train).predict(X_test)
my = myKNN(k=3)
my.fit(X_train,y_train)
predictions2 =  my.predict(X_test)
print("Compare my model with sklearn")
print("Accuracy of sklearn model:", acurracy(predictions1, y_test))
print("Accuracy of my model:", acurracy(predictions2, y_test))

print("Compare sklearn vs my predictions | True labels")
for i in range(len(predictions1)):
    print(f"sklearn: {predictions1[i]} | my model: {predictions2[i]} | true label: {y_test[i]}")
