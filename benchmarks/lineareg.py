from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression 
from src.models.LinearRegression import LinearRegression as myLinearRegression
from src.utils.metric import MSE

X,y = datasets.make_regression(n_samples=100,n_features=1,noise=20,random_state=4)



X_train, X_test, y_train, y_test = train_test_split(X,y, train_size=0.8, random_state=1234)

predictions1 =  LinearRegression().fit(X_train,y_train).predict(X_test)
my = myLinearRegression(lr=0.1,n_iters=1000)
my.fit(X_train,y_train)
predictions2 =  my.predict(X_test)
print("Compare my model with sklearn")
print("Mean MSE of sklearn model:", MSE(predictions1, y_test))
print("Mean MSE of my model:", MSE(predictions2, y_test))

print("Compare sklearn vs my predictions | True labels")
for i in range(len(predictions1)):
    print(f"sklearn: {predictions1[i]} | my model: {predictions2[i]} | true label: {y_test[i]}")