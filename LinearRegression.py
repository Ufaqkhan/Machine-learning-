'''THE GRAPH IS PLOTTED MULTIPLE TIMES TO BETTER UNDERSTAND THE WORKING 
OF THE LINEAR REGRESSION ALGORITHM'''
print("......LINEAR REGRESSION STARTING......")
#Importing useful packadges 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Parsing data from csv file into ndarray
#Data set used is included in the directory(ex1data1.txt)
file = pd.read_csv("ex1data1.txt" , header = None).values
print(file)

#Slicing the data into feature vector and target vector
X = file[:,0]
Y = file[:,1]
print(type(X))
input("Press ENTER to continue")

#Plotting the data 
plt.scatter(X,Y, marker= "^", color= "r")
plt.xlabel("Profit in $10,000s")
plt.ylabel("Population of city in 10,000s")
plt.show  

#Calculate cost and run gradient desent
print("Shape of feature value array X ",np.shape(X))
print("Shape of target value  array Y ",np.shape(Y))


#Function for Plotting line on the graph
def plotFunction(X, Y, varTheta):
	plt.plot(X[:,1], np.matmul(X,varTheta))


total_data_items  = len(Y)
ones   = np.ones((total_data_items,), dtype=int)
X = np.transpose([ones,X])
Y = np.transpose([Y])
theta =  np.zeros((2,1), dtype=int)

#Learning rate alpha and iteration 
alpha = 0.02
iteration = 1500


print("......COST FUNCTION RUNNING.....")
def costFunction(X, Y, theta):
	J = 0
	error = np.square(np.matmul(X,theta) - Y)
	J = 1/(2*total_data_items) * np.sum(error)
	return J

print("Initial cost when theta is zero",costFunction(X, Y, theta))
#This portion is specfic to this dataset to check correct working of this algo	
########################################################################
#checking cost at theta = 0,0
J = costFunction(X, Y, theta)
print("expected cost 32.07 = ",J)

#Plot scatter graph
plotFunction(X, Y, theta)


#Checking cost at theta = -1,2
J = costFunction(X, Y, [[-1],[2]])
print("expected cost 54.24 = ",J)

#Plot scatter graph
plotFunction(X, Y, theta)

########################################################################
#Plot line one the graph

print("........GRADIENT DECENT........")
input("Press ENTER to start Trainning")
#Gradent decent (cost minimizer)
def gradientDecent(X, Y, theta, alpha, iteration):
	for start in range(iteration):
		J = costFunction(X, Y, theta)
		print("COST ",J)
		error = np.matmul(X,theta) - Y 
		derivative = np.matmul(np.transpose(X),error)
		theta = theta - alpha * 1/total_data_items * derivative
	
		if(start%30==0):
			plotFunction(X, Y, theta)
		print("Running gradient decent",theta)
	J = costFunction(X, Y, theta)
	print("FINAL COST: ",J)	
	return theta
		

#Best cost  
theta = gradientDecent(X, Y, theta, alpha, iteration)
print("Best values for theta0 and theta1 obtained\n",theta)


#Plot scatter graph
plt.show()
plotFunction(X, Y, theta)
plt.scatter(X[:,1],Y, marker= "^", color= "r")
plt.xlabel("Profit in $10,000s")
plt.ylabel("Population of city in 10,000s")
plt.plot(X[:,1], np.matmul(X,theta))
plt.title("BEST FIT LINE")
plt.show()
print("SUCCESS")





