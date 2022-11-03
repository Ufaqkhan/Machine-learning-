#LOGISTIC REGRESSION USING DATA SET FROM MACHINE LEARNING COURSE OF COURSERA.ORG
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

#loading data from comma seperated file and displaying it
data = pd.read_csv("dataset/ex2data1.txt" , header = 0).values
print(data)

#Features Exam1,Exam2 and Admission(target) 
X = data[:,0:2]
Y = data[:,2]

#Display Feature varivale X and target variable Y
print(X)
m = np.size(Y)
Y = np.transpose([Y])
print(Y)

#Seperating zero and one examples for plotting with different
#markers for better visualization  
pos_index = np.where(Y == 1)[0]
neg_index = np.where(Y == 0)[0]
positive = X[pos_index, :2]
negative = X[neg_index, :2]

#Plotting the scatter plot
plt.scatter(positive[:, 0], positive[:, 1], color="b",s = 70, marker="+")
plt.scatter(negative[:, 0],negative[:, 1], color="r", s = 60,marker="o")
plt.xlabel("Exam1")
plt.ylabel("Exam2")
plt.show()

#Adding bais/intersept to the feature variable 
ones = np.ones((m,1,), dtype = float)
X = np.concatenate((ones,X),axis=1)

#Theta initialized
theta = np.zeros((3,1), dtype= int)

#Sigmoid function
def sigmoid(z):
	g  = 1./(1 + np.exp(-z))
	return g
#Cost function
def costFunction(X, Y, theta):
	sig = np.matmul(X,theta)
	sigmoid_value = sigmoid(sig)
	J = np.sum(1/m * (-Y * np.log(sigmoid_value) - (1-Y) * np.log(1 - sigmoid_value )))
	error = sigmoid_value - Y
	print(J)
	return J


################################################################################
#testing values 
costFunction(X,Y,[[0],[0],[0]])
print("Expected cost 0.6931")       
costFunction(X,Y,[[-24], [0.2],[0.2]])
print("Expected cost 0.2183")
################################################################################

#Setting learning rate and iteration
alpha = 0.01
iteration = 1000000

#Gradient Decent for finding gradient
def gradientDecent(X, Y, theta, alpha, iteration):

	for i in range(iteration):
		sig = np.matmul(X, theta)
		sigmoid_value = sigmoid(sig)
		error = sigmoid_value - Y
		gradient = 1/m*(np.matmul(np.transpose(X),error))
		theta = theta - alpha * gradient
		#For displaying cost minimization
		if(i%99 == 0):
			costFunction(X,Y,theta)
	return theta

#Predicting the accuracy
def predict(X, theta):
	count =0
	p = sigmoid(np.dot(X,theta))
	p = np.round(p)
	for i in range(m):
		print(p[i],"",Y[i])
		if p[i] == Y[i]:
			count = count+1
	print("Accuracy:",count/m*100,"%")			   

#Final theta 

print("Press Enter to train the model")
input()
print("Gradient Decent Running.......................")
theta = gradientDecent(X, Y, theta, alpha, iteration)

#checking accuracy
predict(X,theta)
print("theta",theta,"cost",costFunction(X, Y, theta))

#plotting the line
plt.plot(X[:,1],np.matmul(X[:,1].reshape(m,1),theta.T))
plt.show()
