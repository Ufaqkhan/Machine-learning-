from random import shuffle

filename = 'data.txt'
#hyperparameters of the algorithm
n_folds =20
max_depth = 10
min_size = 5

#splits the data in k folds and returns the concatenated data
def k_fold_cross_validation(items, randomize=False):
	k=n_folds
	if randomize:
		items = list(items)
		shuffle(items)
	slices = [items[i::k] for i in xrange(k)]
	return slices

#loading the data as well as converting the data into float value. 
def data(fname):
	X,Z=list(),list()
	with open(fname,'r') as f:
		contents = f.readlines()
		Z+= contents
	for i in Z:
		my_list = i.strip("\n").strip("\r").split(",")
		X.append(my_list)
	X = [[float(column) for column in row] for row in X]
	return X
 
# Calculate the Gini index for a split dataset
def gini(groups,class_values):
	gini_value = 0.0
	for class_value in class_values:#for each class_value 
		for group in groups:
			number=0.0
			size = len(group)
			if size == 0:
				continue
			for row in group:
				if(row[-1]==class_value):
					number+=1 #number of rows having the same class_value
			ratio= number/float(len(group))
			#ratio of the particular class value over the total set of the group
			gini_value += (ratio* (1.0 - ratio))#fourmula to calculate the gini index
	return gini_value
 

'''the data is split in accordance with the k_fold_cross_validation 
here the data that is already divided into folds is divided into testing and training data
with every fold getting the chance of getting the testing data when the other folds are the training data.
'''
def calculate_accuracy(dataset):
	folds=k_fold_cross_validation(dataset)
	scores = list()
	for i in range(n_folds):
		fold_at_i = folds[i]
		training=list()
		for j in range(n_folds):
			if j!=i:
				training.append(folds[j])
		training = sum(training, [])#removes the error for unhashable types
		testing = list()#make a test data list
		for row in fold_at_i:
			row_copy = list(row)
			row_copy[-1] = None#MAKE THE LABEL NONE AND THEN APPEND
			testing.append(row_copy)#make the test set
		predicted = decision_tree(training, testing)
		actual = [row[-1] for row in fold_at_i]
		correct_values = 0# Calculate accuracy percentage. pretty straightforward
		for i in range(len(actual)):
			if (actual[i] == predicted[i]):
				correct_values += 1
		accuracy= correct_values / float(len(actual)) * 100.0
		scores.append(accuracy)
	return scores
 
# Split a dataset based on an attribute and an attribute value
def test_split(index, value, dataset):
	left, right = list(), list()
	for row in dataset:
		if row[index] < value:
			left.append(row)
		else:
			right.append(row)
	return left, right
 

# Select the best split point for a dataset
def splitting(dataset):
	class_values = list(set(row[-1] for row in dataset))
	node_index=999
	node_value=999
	node_score=999
	node_groups =None
	for index in range(len(dataset[0])-1):
	#all index values for all the attributes
		count=0.0
		count_row=0.0
		for row in dataset:#here we take the average of the values in a attribute to split the data 
			count+=1
			count_row=row[index]+count_row
		count_row=count_row/count
		groups=test_split(index,count_row,dataset)
		gini_value=gini(groups,class_values)
		if gini_value < node_score:#store the values based on which the gini value is the lowest and the splitting is done
		#on the basis of that attribute
				node_index, node_value, node_score, node_groups = index, row[index], gini_value, groups
	return {'i':node_index, 'value':node_value, 'div':node_groups}#return the dictionary containing the details 
	#on the basis of which the splitting is done
 
# Create a terminal node value contains highest frequency label from groups
def terminal_node(group):
	outcomes = [row[-1] for row in group]#this returns all the labels for the data and stores in the outcome list
	classify=[0,0]
	for out in outcomes:
		classify[(int(out))]+=1
	if(classify[0]>classify[1]):
		return 0
	else:
		return 1

# Create child splits for a node or make terminal
def node_split(node,depth):
	left, right = node['div']
	del(node['div'])
	# check if there is no split and the data is coherent. If all the values in the attribute is greater or less that the particular avg value of the attribute
	if not left or not right:
		node['left'] = node['right'] = terminal_node(left + right)
		return
	# check for max depth is achieved
	if depth >= max_depth:
		node['left'], node['right'] = terminal_node(left), terminal_node(right)
		return
	# process left child
	if len(left) <= min_size:#if the mininmum records in the data is less than or equal to the min_size. No splitting
		node['left'] = terminal_node(left)
	else:
		node['left'] = splitting(left)#otherwise split the data further keeping in mind the min_size and the max_depth
		node_split(node['left'], depth+1)
	# process right child same as above
	if len(right) <= min_size:
		node['right'] = terminal_node(right)
	else:
		node['right'] = splitting(right)
		node_split(node['right'], depth+1)

 
# Make a prediction from the tree
def output_from_tree(node, row):
	if row[node['i']] < node['value']:
		if (type(node['left'])== dict):#if there exists a node['left'] of type dict
			return output_from_tree(node['left'], row)
		else:
			return node['left']#otherwise return the label depicted by the terminal node
	else:#same for the right side of the subtree
		if (type(node['right'])== dict):
			return output_from_tree(node['right'], row)
		else:
			return node['right']
 
# Classification and Regression Tree Algorithm
def decision_tree(train, test):
	root= splitting(train)
	node_split(root,1)
	predictions = list()
	for row in test:
		prediction = output_from_tree(root, row)
		predictions.append(prediction)
	return(predictions)


dataset = data(filename)
scores = calculate_accuracy(dataset)
print('Mean Accuracy: %.3f%%' % (sum(scores)/(len(scores))))
