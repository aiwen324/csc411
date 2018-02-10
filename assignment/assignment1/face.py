from rgb2gray import *
from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import random
import time
from scipy.misc import imread
from scipy.misc import imresize
from scipy.misc import imsave
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.image as mpimg
import os
from scipy.ndimage import filters
import urllib
from shutil import copy2
import pickle


SINGLE_VALUE = 0
VECTOR = 1

male_actors = ['baldwin', 'hader', 'radcliffe', 'butler', 'vartan', 'carell']
actresses = ['bracco', 'chenoweth', 'drescher', 'ferrera', 'gilpin', 'harmon']

# Linear regression for part3
# combine all data from training set
# Result matrix is in the size: n*(32*32+1)
"""
|img1 1|
|img2 1|
|img3 1|
|img4 1|
...  ...
|imgn 1|
"""
def img_matrix(actors, b_flag=1, set_flag=0, gender_flag=0):
	# indicator for combine bias to the matrix
	if b_flag == 1:
		data_matrix = np.empty((0, 32*32+1), dtype=np.float64)
	else:
		data_matrix = np.empty((0, 32*32), dtype=np.float64)
	n = len(actors)
	result_value_vector = np.array([])
	if n == 2:
		actor0, actor1 = actors[0], actors[1]
	# loop through the dataset, combine all data into matrix
	for actor in actors:
		for actor_set in os.listdir("sep_dataset"):
			if actor in actor_set:
				if set_flag == 0:
					dir_name = "sep_dataset/"+actor_set+"/training_set/"
				elif set_flag == 1:
					dir_name = "sep_dataset/"+actor_set+"/validation_set/"
				else:
					dir_name = "sep_dataset/"+actor_set+"/test_set/"
				for filename in os.listdir(dir_name):
					im = imread(dir_name+filename)
					im = np.true_divide(im, 255)
					im = im.flatten()
					if b_flag == 1:
						im = np.append(im, [1])
					data_matrix = np.vstack((data_matrix, im))
					if gender_flag == 1:
						if actor in male_actors:
							result_value_vector = np.append(result_value_vector, [1])
						if actor in actresses:
							result_value_vector = np.append(result_value_vector, [-1])

					else:
						if actor == actor0:
							result_value_vector = np.append(result_value_vector, [1])
						elif actor == actor1:
							result_value_vector = np.append(result_value_vector, [-1])
						else:
							raise NotImplementedError
	"""
	print data_matrix.shape
	print result_value_vector.shape
	print data_matrix
	print result_value_vector
	"""
	return data_matrix, result_value_vector

# sum of squares cost function
def sum_of_squares(data, theta, real_value, type=VECTOR):
	if type == VECTOR:
		distance_vector = np.dot(data, theta.T) - real_value
		#print distance_vector
		#print distance_vector.shape
		return 0.5*np.inner(distance_vector, distance_vector)

# derivative function of sum_of_squares
def df_sum_of_squares(data, theta, real_value):
	predict = np.dot(data, theta.T)
	dCost_dy = predict-real_value
	dy_dx = data
	return np.dot(dy_dx.T, dCost_dy.T)

# gradient descent, code is from csc411_linreg notebook with little modifies
def grad_descent(f, df, x, init_theta, real_value, alpha, max_iter=40000, EPS=1e-5):
	prev_theta = init_theta-10*EPS
	theta = init_theta.copy()
	iter = 0
	while norm(theta-prev_theta) > EPS and iter < max_iter:
		prev_theta = theta.copy()
		theta = theta - alpha*df(x, theta, real_value)
		# some code for showing process
		print "Iter", iter
		print "theta", theta, "cost", f(x, theta, real_value)
		print "Gradient: ", df(x, theta, real_value), "\n"
		# increase iter
		iter += 1
	return theta

# calculate accuracy over
def accuracy(actors, data, theta, real_value, flag, gender_flag=0):
	if flag == 0:
		# calculate accuracy over training set
		predict = np.dot(data, theta.T)
		predict = np.array([-1 if i < 0 else 1 for i in predict])
		r = predict - real_value
		total_num = r.shape[0]
		print "total_num: ", total_num
		correct_num = 0
		for i in r:
			if i == 0:
				correct_num += 1
		return np.true_divide(correct_num, total_num)
	elif flag == 1:
		# calculate accuracy over validation set
		data_matrix, result_value_vector = img_matrix(actors, b_flag=1, set_flag=flag, gender_flag=gender_flag)
		predict = np.dot(data_matrix, theta.T)
		predict = np.array([-1 if i < 0 else 1 for i in predict])
		r = predict - result_value_vector
		total_num = r.shape[0]
		correct_num = 0
		for i in r:
			if i == 0:
				correct_num += 1
		return np.true_divide(correct_num, total_num)
	else:
		# calculate accuracy over test set
		data_matrix, result_value_vector = img_matrix(actors, b_flag=1, set_flag=flag, gender_flag=gender_flag)
		predict = np.dot(data_matrix, theta.T)
		predict = np.array([-1 if i < 0 else 1 for i in predict])
		r = predict - result_value_vector
		total_num = r.shape[0]
		correct_num = 0
		for i in r:
			if i == 0:
				correct_num += 1
		return np.true_divide(correct_num, total_num)

# Algorithm for using gradient descent to get the best theta
def linear_reg_solver(act_list, x, real_value, t=0, gender_flag=0):
	# Use random number to get the initial theta
	if t == 2:
		init_theta = np.array([1 for i in range(1025)])
	else:
		np.random.seed(t)
		init_theta = np.random.rand(x.shape[1])
	print init_theta
	# Write a list of learning rates
	alpha_list = [1e-4, 1e-5, 1e-6, 1e-8]
	# Write a list of maximum iteration
	max_iter = 4e4
	# Use accuracy to judge if the data is overfitting or not
	performance = np.empty((0, 3), dtype=np.float64)
	report = dict()

	if gender_flag == 1:
		alpha = 1e-5
		theta = init_theta.copy()
		theta = grad_descent(sum_of_squares, df_sum_of_squares, x, theta, real_value, alpha, max_iter=30000)
		val_accuracy = accuracy(act_list, None, theta, None, 1, gender_flag=gender_flag)
		tra_accuracy = accuracy(act_list, x, theta, real_value, 0, gender_flag=gender_flag)
		# Make this into a matrix
		performance = np.vstack((performance, np.array([alpha, 30000, val_accuracy])))
		tra_cost = sum_of_squares(x, theta, real_value)
		val_data,real_value2 = img_matrix(act_list, b_flag=1, set_flag=1, gender_flag=gender_flag)
		val_cost = sum_of_squares(val_data, theta, real_value2)
		report[(alpha, 30000)] = (theta, tra_cost, tra_accuracy, val_cost, val_accuracy)
	else:
		for alpha in alpha_list:
			i = 0
			theta = init_theta.copy()
			while i*1e4 < max_iter:
				# Do gradient descent with max_iter = 1e5 and return theta
				theta = grad_descent(sum_of_squares, df_sum_of_squares, x, theta, real_value, alpha, max_iter=10000)
				i += 1
				val_accuracy = accuracy(act_list, None, theta, None, 1, gender_flag=gender_flag)
				tra_accuracy = accuracy(act_list, x, theta, real_value, 0, gender_flag=gender_flag)
				# Make this into a matrix
				performance = np.vstack((performance, np.array([alpha, i*1e4, val_accuracy])))
				tra_cost = sum_of_squares(x, theta, real_value)
				val_data,real_value2 = img_matrix(act_list, b_flag=1, set_flag=1, gender_flag=gender_flag)
				val_cost = sum_of_squares(val_data, theta, real_value2)
				report[(alpha, i*1e4)] = (theta, tra_cost, tra_accuracy, val_cost, val_accuracy)
	return performance, report

# Parse out the best theta, save theta and performance and report to files if needed
def linear_reg_handler(performance, report, ind):
	m = np.argmax(performance, axis=0)
	max_data = performance[m[-1], :]
	result = report[(max_data[0], max_data[1])]
	best_theta = result[0]
	# Save best_theta and performance and max_data
	np.save("tmp/part5_theta"+str(ind), best_theta)
	np.save("tmp/part5_performance"+str(ind), performance)
	np.save("tmp/part5_max_data"+str(ind), max_data)
	pickle_out = open("tmp/part5_report.pickle"+str(ind), "w")
	pickle.dump(report, pickle_out)
	pickle_out.close()
	return best_theta, result


# ====================================== Part 3 =============================================
def part3():
	actors = ['baldwin', 'carell']
	x, real_value = img_matrix(actors)
	performance, report = linear_reg_solver(actors, x, real_value)
	m = np.argmax(performance, axis=0)
	max_data = performance[m[-1], :]
	result = report[(max_data[0], max_data[1])]
	best_theta = result[0]
	test_accuracy = accuracy(actors, None, best_theta, None, 2)
	print max_data
	print performance
	print m[-1]
	print "Best theta we trained is: ", best_theta
	print "The training cost and training accuracy on this theta is: ", result[1], result[2]
	print "The validation cost and validation accuracy on this that is: ", result[3], result[4]
	print "The testing accuracy on this theta is: ", test_accuracy
	string = "Best theta we trained is: "+str(best_theta)+'\n'
	string += "The training cost and training accuracy on this theta is: "+str(result[1])+" "+str(result[2])+'\n'
	string += "The validation cost and validation accuracy on this that is: "+str(result[3])+" "+str(result[4])+'\n'
	string += "The testing accuracy on this theta is: "+str(test_accuracy)
	f = open("report.txt","w+")
	f.write(string)
	f.close()
	# Save the data to make it more convenient to use
	np.save("tmp/part3_theta", best_theta)
	np.save("tmp/part3_performance", performance)
	pickle_out = open("tmp/part3_report.pickle", "w")
	pickle.dump(report, pickle_out)
	pickle_out.close()
	return best_theta, report

# ======================================= Part 4 ======================================
def part4(theta, actors):
	# get the data from X
	# Set up an initial theta
	# Do gradient descent for such theta
	X, real_value = img_matrix(actors, b_flag=1)
	X = X[[0,1,70,71],:]
	print X.shape
	real_value = [real_value[0],real_value[1],real_value[70],real_value[71]]
	performance, report = linear_reg_solver(actors, X, real_value, t=2)
	m = np.argmax(performance, axis=0)
	max_data = performance[m[-1], :]
	result = report[(max_data[0], max_data[1])]
	best_theta = result[0]
	# Get rid of the bias
	theta = theta.copy()[:1024]
	best_theta = best_theta.copy()[:1024]
	theta.resize(32, 32)
	best_theta.resize(32, 32)
	pickle_out = open("tmp/part4_report.pickle", "w")
	pickle.dump(report, pickle_out)
	pickle_out.close()
	imshow(theta)
	show()
	imshow(best_theta)
	imsave("report_photo/theta.jpg", theta)
	imsave("report_photo/theta2.jpg", best_theta)
	print "best_theta is", best_theta
	print "maxdata is:", max_data
	show()
	return best_theta, (max_data[0], max_data[1])

def part4_b():
	pickle_in = open("tmp/part3_report.pickle", "r")
	part3_report = pickle.load(pickle_in)
	pickle_in.close()
	pickle_in = open("tmp/part4_report.pickle", "r")
	part4_report = pickle.load(pickle_in)
	pickle_in.close()
	k = 0
	for tuple in part3_report:
		theta = part3_report[tuple][0].copy()[:1024]
		theta.resize(32, 32)
		print "dealing with (alpha, iteration): ", tuple, part3_report[tuple]
		imsave("report_photo/"+"alltheta_"+str(k)+".jpg", theta)
		k += 1
	j = 0
	for tuple in part4_report:
		theta = part4_report[tuple][0].copy()[:1024]
		theta.resize(32, 32)
		print "dealing with (alpha, iteration): ", tuple, part4_report[tuple]
		imsave("report_photo/"+"4theta_"+str(j)+".jpg", theta)
		j += 1

# ===================================== Part 5 ==========================================
def part5():
	act = ['Lorraine Bracco', 'Peri Gilpin', 'Angie Harmon', 'Alec Baldwin', 'Bill Hader', 'Steve Carell']
	act = [i.split()[1].lower() for i in act]
	x, real_value = img_matrix(act, b_flag=1, set_flag=0, gender_flag=1)
	# Get n = 240, 270, 300, 330, 360, 390, 420
	# n = 240
	indices1 = np.array([i for i in range(40)])
	vector1 = indices1.copy()
	for i in range(5):
		vector1 += 70
		indices1 = np.append(indices1, vector1)
	x1 = x[indices1,:]
	real_value1 = real_value.copy()[indices1]
	performance1, report1 = linear_reg_solver(act, x1, real_value1, gender_flag=1)
	best_theta1, result1 = linear_reg_handler(performance1, report1, 1)
	# n = 270
	indices2 = np.array([i for i in range(45)])
	vector2 = indices2.copy()
	for i in range(5):
		vector2 += 70
		indices2 = np.append(indices2, vector2)
	x2 = x[indices2,:]
	real_value2 = real_value.copy()[indices2]
	performance2, report2 = linear_reg_solver(act, x2, real_value2, gender_flag=1)
	best_theta2, result2 = linear_reg_handler(performance2, report2, 2)
	# n = 300
	indices3 = np.array([i for i in range(50)])
	vector3 = indices3.copy()
	for i in range(5):
		vector3 += 70
		indices3 = np.append(indices3, vector3)
	x3 = x[indices3,:]
	real_value3 = real_value.copy()[indices3]
	performance3, report3 = linear_reg_solver(act, x3, real_value3, gender_flag=1)
	best_theta3, result3 = linear_reg_handler(performance3, report3, 3)
	# n = 330
	indices4 = np.array([i for i in range(55)])
	vector4 = indices4.copy()
	for i in range(5):
		vector4 += 70
		indices4 = np.append(indices4, vector4)
	x4 = x[indices4,:]
	real_value4 = real_value.copy()[indices4]
	performance4, report4 = linear_reg_solver(act, x4, real_value4, gender_flag=1)
	best_theta4, result4 = linear_reg_handler(performance4, report4, 4)
	# n = 360
	indices5 = np.array([i for i in range(60)])
	vector5 = indices5.copy()
	for i in range(5):
		vector5 += 70
		indices5 = np.append(indices5, vector5)
	x5 = x[indices5,:]
	real_value5 = real_value.copy()[indices5]
	performance5, report5 = linear_reg_solver(act, x5, real_value5, gender_flag=1)
	best_theta5, result5 = linear_reg_handler(performance5, report5, 5)
	# n = 390
	indices6 = np.array([i for i in range(65)])
	vector6 = indices6.copy()
	for i in range(5):
		vector6 += 70
		indices6 = np.append(indices6, vector6)
	x6 = x[indices6,:]
	real_value6 = real_value.copy()[indices6]
	performance6, report6 = linear_reg_solver(act, x6, real_value6, gender_flag=1)
	best_theta6, result6 = linear_reg_handler(performance6, report6, 6)
	# n = 420
	x7 = x.copy()
	real_value7 = real_value.copy()
	performance7, report7 = linear_reg_solver(act, x7, real_value7, gender_flag=1)
	best_theta7, result7 = linear_reg_handler(performance7, report7, 7)
	sizes = np.array([240, 270, 300, 330, 360, 390, 420])
	perform_trains = np.array([result1[2], result2[2], result3[2], result4[2], result5[2], result6[2], result7[2]])
	perform_valids = np.array([result1[4], result2[4], result3[4], result4[4], result5[4], result6[4], result7[4]])
	print perform_trains
	print perform_valids
	np.save("tmp/perform_trains", perform_trains)
	np.save("tmp/perform_valids", perform_valids)


def part5_b(actors):
	# Get the best theta from saved file part5_theta5.npy
	theta = np.load("tmp/part5_theta5.npy")
	data, real_value = img_matrix(actors, b_flag=1, set_flag=0, gender_flag=1)
	tra_accuracy = accuracy(actors, data, theta, real_value, 0, gender_flag=1)
	val_accuracy = accuracy(actors, None, theta, None, 1, gender_flag=1)
	test_accuracy = accuracy(actors, None, theta, None, 2, gender_flag=1)
	f = open("report.txt", "a+")
	string = "Part5:\n"
	string += "training set accuracy: " + str(tra_accuracy) + "\n"
	string += "validation set accuracy: " + str(val_accuracy) + "\n"
	string += "test set accuracy: " + str(test_accuracy) + "\n"
	f.write(string)
	f.close()
	print string

# ======================================== Part 6 ===========================================
# Define the cost function for part6
def J(X, theta, Y):
	cost_matrix = np.dot(theta.T, X) - Y
	cost_matrix = cost_matrix.T
	#print cost_matrix
	cost_sqr_matrix = np.array([np.dot(i, i) for i in cost_matrix])
	#print cost_sqr_matrix
	return np.sum(cost_sqr_matrix)

# Define the df function of J
def df_J(X, theta, Y):
	matrix1 = np.dot(theta.T, X) - Y
	matrix2 = np.dot(X, matrix1.T)
	matrix2 = 2*matrix2
	return matrix2

# t is the scalar we use
def manual_df_J(X, theta, Y, p, q, t):
	new_theta = theta.copy()
	new_theta[p][q] += t
	print new_theta
	print theta
	df = J(X, new_theta, Y) - J(X, theta, Y)
	df = np.true_divide(df, t)
	return df

# t is scalar for directional vector, default is 1e-4
def mat_man_df_J(X, theta, Y, t=1e-4):
	size = theta.shape
	df_matrix = np.empty(size, dtype=np.float64)
	for i in range(size[0]):
		for j in range(size[1]):
			df_matrix[i][j] = manual_df_J(X, theta, Y, i, j, t)
	return df_matrix


# Some test cases for part 6
"""
X = np.arange(0, 4, 1)
X.resize(2, 2)
Y = np.arange(0, 8, 1)
Y.resize(4, 2)
theta = np.array([1. for i in range(8)])
theta.resize(2, 4)
result = mat_man_df_J(X, theta, Y, t=1e-10)
result2 = df_J(X, theta, Y)
print result
print result2
"""
# Get output:
"""
[[  6.0001   2.0001  -1.9999  -5.9999]
 [ 26.0013   6.0013 -13.9987 -33.9987]]
[[  6.   2.  -2.  -6.]
 [ 26.   6. -14. -34.]]
"""


#====================================    Part 7   ===========================================
def img_matrix2(actors, set_flag=0):
	X = np.empty((0, 32*32+1), dtype=np.float64)
	k = len(actors)
	Y = np.empty((0, k), dtype=np.float64)
	for actor in actors:
		actor_index = actors.index(actor)
		actor_array = np.array([0 for i in range(k)])
		actor_array[actor_index] = 1
		for actor_set in os.listdir("sep_dataset"):
			if actor in actor_set:
				if set_flag == 0:
					dir_name = "sep_dataset/" + actor_set + "/training_set/"
				elif set_flag == 1:
					dir_name = "sep_dataset/" + actor_set + "/validation_set/"
				else:
					dir_name = "sep_dataset/" + actor_set + "/test_set/"
				for filename in os.listdir(dir_name):
					im = imread(dir_name + filename)
					im = np.true_divide(im, 255)
					im = im.flatten()
					im = np.append(im, [1])
					X = np.vstack((X, im))
					Y = np.vstack((Y, actor_array))
	X = X.T
	Y = Y.T
	print X.shape, Y.shape
	return X, Y

def evaluate(X, theta):
	if len(X.shape) > 1:
		# matrix is k*N
		result = np.dot(theta.T, X)
		# Find the index of largest number in each column
		m = np.argmax(result, axis=0)
		predict = np.zeros((theta.shape[1], X.shape[1]))
		if m.shape[0] != X.shape[1]:
			raise NotImplementedError
		for i in range(X.shape[1]):
			predict[m[i]][i] = 1
		return predict
	else:
		print "dealing with single array X"
		result = np.dot(theta.T, X.T)
		m = np.argmax(result)
		return m

def count_match(predict, Y):
	total_number = predict.shape[1]
	count = 0
	for i in range(total_number):
		if np.array_equal(predict[:, i], Y[:, i]):
			count += 1
	print total_number
	print count
	return np.true_divide(count, total_number)


def accuracy2(actors, X, theta, Y, flag):
	if flag == 0:
		predict = evaluate(X, theta)
		total_number = predict.shape[1]
		count = 0
		for i in range(total_number):
			if np.array_equal(predict[:, i], Y[:, i]):
				count += 1
	elif flag == 1:
		X_1, Y_1 = img_matrix2(actors, set_flag=1)
		predict = evaluate(X_1, theta)
		total_number = predict.shape[1]
		count = 0
		for i in range(total_number):
			if np.array_equal(predict[:, i], Y_1[:, i]):
				count += 1
	else:
		X_2, Y_2 = img_matrix2(actors, set_flag=2)
		predict = evaluate(X_2, theta)
		total_number = predict.shape[1]
		count = 0
		for i in range(total_number):
			if np.array_equal(predict[:,i], Y_2[:,i]):
				count +=1

	return np.true_divide(count, total_number)



# Code to run
"""
actors = ['Lorraine Bracco', 'Peri Gilpin', 'Angie Harmon', 'Alec Baldwin', 'Bill Hader', 'Steve Carell']
actors = [i.split()[1].lower() for i in actors]


X, Y = img_matrix2(actors)
# initialize a theta with 1s
init_theta = np.ones(shape=(1025, len(actors)))
#init_theta = np.random.rand(1025, len(actors))
alphas = [1e-5, 5e-5, 1e-6, 5e-6, 1e-7]
alphas = [1e-6]
#alphas = [1e-6, 5e-6]
i = 0
for alpha in alphas:
	theta = grad_descent(J, df_J, X, init_theta, Y, alpha)
	tra_acc = accuracy2(actors, X, theta, Y, 0)
	val_acc = accuracy2(actors, None, theta, None, 1)
	#test_acc = accuracy2(actors, X, theta, Y, 2)
	np.save("tmp/part7_theta_"+str(alpha), theta)
	string = "part7\n"
	string += "with alpha: "+ str(alpha)+"\n"
	string += "traing set accuracy: " + str(tra_acc) + "\n"
	string += "validation set accuracy: " + str(val_acc) + "\n"
	#string += "test set accuracy: " + str(test_acc) + "\n"
	f = open("report.txt", "a+")
	f.write(string)
	f.close()
	i += 1

# Find alpha = 1e-6 works greate, maybe need more iterations to get higher performance
theta = np.load("tmp/part7_theta_1e-6.npy")
test_acc = accuracy2(actors, None, theta, None, 2)
print "accuracy on test set: " + str(test_acc)
string = "with best performance theta on validation set, get test set accuracy: " + str(test_acc) + "\n"
f = open("report.txt", "a+")
f.write(string)
f.close()
"""
# ================================= Part 8 ==============================================
"""
theta = np.load("tmp/part7_theta_1e-6.npy")
actors = ['Lorraine Bracco', 'Peri Gilpin', 'Angie Harmon', 'Alec Baldwin', 'Bill Hader', 'Steve Carell']
actors = [i.split()[1].lower() for i in actors]
# for i in range(k), theta size: n*k
for i in range(theta.shape[1]):
	new_theta = theta.copy()[:theta.shape[0]-1,i]
	print new_theta
	# new_theta1 is my new flatten image
	new_theta1 = np.append(new_theta, [1])
	print new_theta1
	index = evaluate(new_theta1, theta)
	actor_name = actors[index]
	# new_theta is the one to save image
	print new_theta.shape
	new_theta = np.resize(new_theta, (32, 32))
	print new_theta.shape
	imsave("report_photo/part8_theta"+str(i)+actor_name+".jpg", new_theta)
"""

# ============================= Helper function for convert images and separate images to dataset
# Algorithm to pick training set, validation set and testing set
def sep_img(actors=None,force=None):
	# if no actor is specified, we will do sep_img to all actors
	if actors is None:
		for act in os.listdir("cropped"):
			newdir_name = "sep_dataset/" + act + "_seprated_set/"
			if os.path.exists(newdir_name) and os.path.isdir(newdir_name):
				print ("Already seperate the dataset for actor: " + act)
				if force is None:
					continue
			else: 
				os.mkdir(newdir_name)
			# make directory
			if not os.path.exists(newdir_name+"training_set/"):
				os.mkdir(newdir_name+"training_set/")
			if not os.path.exists(newdir_name+"validation_set/"):
				os.mkdir(newdir_name+"validation_set/")
			if not os.path.exists(newdir_name+"test_set/"):
				os.mkdir(newdir_name+"test_set/")
			# iterate through the dataset
			count = 0
			print ("seperating set for actor: " + act + "...........")
			for filename in os.listdir("cropped/" + act):
				if count < 70:
					# cp file to training set
					copy2("cropped/" + act + "/" + filename, newdir_name + "training_set/"+filename)

				elif count >= 70 and count < 80:
					# cp file to validation set
					copy2("cropped/" + act + "/" + filename, newdir_name + "validation_set/"+filename)

				else:
					# cp file to testing set
					copy2("cropped/" + act + "/" + filename, newdir_name + "test_set/"+filename)
				count += 1
	# for some specific actor
	else:
		for actor in actors:
			print ("seperating set for actor: " + actor + "...........")
			newdir_name = "sep_dataset/" + actor + "_seprated_set/"
			if os.path.exists(newdir_name) and os.path.isdir(newdir_name):
				print ("Already seperate the dataset for actor: " + actor)
			else:
				os.mkdir(newdir_name)
			if not os.path.exists(newdir_name+"training_set/"):
				os.mkdir(newdir_name+"training_set/")
			if not os.path.exists(newdir_name+"validation_set/"):
				os.mkdir(newdir_name+"validation_set/")
			if not os.path.exists(newdir_name+"test_set/"):
				os.mkdir(newdir_name+"test_set/")
			count = 0
			for filename in os.listdir("cropped/" + actor):
				if count < 70:
					# cp file to training set
					copy2("cropped/" + actor + "/" + filename, newdir_name + "training_set/"+filename)

				elif count >= 70 and count < 80:
					# cp file to validation set
					copy2("cropped/" + actor + "/" + filename, newdir_name + "validation_set/"+filename)

				else:
					# cp file to testing set
					copy2("cropped/" + actor + "/" + filename, newdir_name + "test_set/"+filename)
				count += 1


# crop all images, convert to gray, resize to 32x32, and save it to directory "cropped/actor_name"
def conver_img():
	for actor in os.listdir("uncropped"):
		act_bd_box_file = open(actor + ".txt")
		if not os.path.exists("cropped/" + actor) or not os.path.isdir("cropped/" + actor):
			os.mkdir("cropped/" + actor)
		for filename in os.listdir("uncropped/" + actor):
			print ("dealing with " + filename)
			try:
				im = imread("uncropped/" + actor + "/" + filename)
				# move file pointer to 0
				act_bd_box_file.seek(0)
				for line in act_bd_box_file:
					if filename in line:
						print ("find match for " + filename)
						bd_box = line.split()[3].split(',')
						bd_box = [int(a) for a in bd_box]
						cropped_im = im[bd_box[1]:bd_box[3], bd_box[0]:bd_box[2]]
						# Check if we the image is 2D or 3D
						if len(cropped_im.shape) != 2:
							cropped_im = rgb2gray(cropped_im)
						# Resize the image
						cropped_im = imresize(cropped_im, [32, 32])
						imsave("cropped/" + actor + "/" + filename, cropped_im)
						break
			# If detect some image cannot be opened in the system, report error
			except IOError:
				f = open(actor + "_err.txt", "a")
				f.write(filename + '\n')
				f.close()





# ================================================= Execute Part =================================
if __name__ == '__main__':

# ============================ Part 1 ================
	#conver_img()
# ============================ Part 2 ================
	#sep_img(force=1)
# ============================ Part 3 ================
	part3()
# ============================ Part 4 ================
	actors = ['baldwin', 'carell']
	load_theta = np.load("tmp/part3_theta.npy")
	part4(load_theta, actors)
	part4_b()
# ============================ Part 5 ================
	part5()
	remaining_actors = ['Daniel Radcliffe', 'Gerard Butler', 'Michael Vartan', 'Kristin Chenoweth', 'Fran Drescher', 'America Ferrera']
	remaining_actors2 = [i.split()[1].lower() for i in remaining_actors]
	part5_b(remaining_actors2)
	y1s = np.load("tmp/perform_trains.npy")
	y2s = np.load("tmp/perform_valids.npy")
	zs = [240, 270, 300, 330, 360, 390, 420]
	plt.plot(zs, y1s, label="performance on training set")
	plt.xlabel("size")
	plt.ylabel("performance")
	plt.legend()
	plt.show()
	
	plt.plot(zs, y2s, label="performance on validation set")
	plt.xlabel("size")
	plt.ylabel("performance")
	plt.legend()
	plt.show()
# ============================ Part 6 ================
	X = np.arange(0, 4, 1)
	X.resize(2, 2)
	Y = np.arange(0, 8, 1)
	Y.resize(4, 2)
	theta = np.array([1. for i in range(8)])
	theta.resize(2, 4)
	result = mat_man_df_J(X, theta, Y, t=1e-10)
	result2 = df_J(X, theta, Y)
	print result
	print result2

# ============================ Part 7 ================
	actors = ['Lorraine Bracco', 'Peri Gilpin', 'Angie Harmon', 'Alec Baldwin', 'Bill Hader', 'Steve Carell']
	actors = [i.split()[1].lower() for i in actors]
	X, Y = img_matrix2(actors)
	# initialize a theta with 1s
	init_theta = np.ones(shape=(1025, len(actors)))
	#init_theta = np.random.rand(1025, len(actors))
	alphas = [1e-5, 5e-5, 1e-6, 5e-6, 1e-7]
	alphas = [1e-6]
	#alphas = [1e-6, 5e-6]
	i = 0
	for alpha in alphas:
		theta = grad_descent(J, df_J, X, init_theta, Y, alpha)
		tra_acc = accuracy2(actors, X, theta, Y, 0)
		val_acc = accuracy2(actors, None, theta, None, 1)
		#test_acc = accuracy2(actors, X, theta, Y, 2)
		np.save("tmp/part7_theta_"+str(alpha), theta)
		string = "part7\n"
		string += "with alpha: "+ str(alpha)+"\n"
		string += "traing set accuracy: " + str(tra_acc) + "\n"
		string += "validation set accuracy: " + str(val_acc) + "\n"
		#string += "test set accuracy: " + str(test_acc) + "\n"
		f = open("report.txt", "a+")
		f.write(string)
		f.close()
		i += 1

	# Find alpha = 1e-6 works greate, maybe need more iterations to get higher performance
	theta = np.load("tmp/part7_theta_1e-06.npy")
	test_acc = accuracy2(actors, None, theta, None, 2)
	print "accuracy on test set: " + str(test_acc)
	string = "with best performance theta on validation set, get test set accuracy: " + str(test_acc) + "\n"
	f = open("report.txt", "a+")
	f.write(string)
	f.close()

# ================================ Part 8 ================
	theta = np.load("tmp/part7_theta_1e-06.npy")
	actors = ['Lorraine Bracco', 'Peri Gilpin', 'Angie Harmon', 'Alec Baldwin', 'Bill Hader', 'Steve Carell']
	actors = [i.split()[1].lower() for i in actors]
	# for i in range(k), theta size: n*k
	for i in range(theta.shape[1]):
		new_theta = theta.copy()[:theta.shape[0]-1,i]
		print new_theta
		# new_theta1 is my new flatten image
		new_theta1 = np.append(new_theta, [1])
		print new_theta1
		index = evaluate(new_theta1, theta)
		actor_name = actors[index]
		# new_theta is the one to save image
		print new_theta.shape
		new_theta = np.resize(new_theta, (32, 32))
		print new_theta.shape
		imsave("report_photo/part8_theta"+str(i)+actor_name+".jpg", new_theta)


# ================================================= Debug Part ====================================
"""
# debug code
act_bd_box_file = open("gilpin" + ".txt")
if not os.path.exists("cropped/"+"gilpin") or not os.path.isdir("cropped/"+"gilpin"):
	os.mkdir("cropped/"+"gilpin")
im = imread("uncropped/harmon/"+"harmon50.jpg")
imshow(im)
show()
bd = [336,121,450,235]
im = im[bd[1]:bd[3], bd[0]:bd[2]]
if len(im.shape) == 2:
	print "get 2D image"
	im = imresize(im, [32, 32])
else:
	print "get 3D image"
	print im.shape
imshow(im)
show()
imsave("report_photo/crp_harmon50.jpg", im)
imshow(im)
show()

# Test sum_of_squares
a = np.array([[1, 1, 2], [2, 2, 3]])
b = np.array([3, 2, 4])
print np.dot(a, b)
t = np.array([4, 5])
print sum_of_squares(a, b, t)

# test grad_descent
actors = ['baldwin', 'carell']
data, result = img_matrix(actors)
theta = np.array([1 for i in range(1025)])
min_theta = grad_descent(sum_of_squares, df_sum_of_squares, data, theta, result, 0.0000010)
print min_theta
print accuracy(actors, data, min_theta, result, 0)
print accuracy(actors, None, min_theta, None, 1)
print accuracy(actors, None, min_theta, None, 2)
"""

