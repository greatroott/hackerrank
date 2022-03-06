'''MultipleLinearRegression(Sklearn version)'''
from sklearn.linear_model import LinearRegression 
import numpy as np 
from sklearn.metrics import mean_squared_error as mse
import sys 

# X = np.array([[0.18,0.89],[1.0, 0.26],[0.92,0.11],[0.07,0.37],[0.85,0.16],[0.99,0.41],[0.87,0.47]])
# y = np.array([109.85,155.72,137.66,76.17,139.75,162.6,151.77])

# lr = LinearRegression(fit_intercept = True,normalize=False)
# lr.fit(X,y)
# test_X = np.array([[0.49,0.18],[0.57,0.83],[0.56,0.64],[0.76,0.18]])
# print("training의 결과는 {:.3f}".format(mse(lr.predict(X),y)))
# print("test 결과는 {}".format([round(i,2) for i in lr.predict(test_X)]))


# '''sysStdIn'''
from sklearn.linear_model import LinearRegression 
import numpy as np 
import sys 

lr = LinearRegression()
 
feature_, n = list(map(int, sys.stdin.readline().strip().split()))
train_X = []
y = []
for i in range(n):
    data = list(map(float,sys.stdin.readline().strip().split()))
    train_X.append([float(i) for i in data[:feature_]])
    y.append([data[feature_]])
lr.fit(np.array(train_X),np.array(y))
test_n = int(sys.stdin.readline().strip())
for i in range(test_n):
    test_X = np.array([list(map(float,sys.stdin.readline().strip().split()))])
    print(round(lr.predict(test_X)[0][0],2))



