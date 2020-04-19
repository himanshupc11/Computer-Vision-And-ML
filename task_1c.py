
'''
*****************************************************************************************
*
*        		===============================================
*           		Rapid Rescuer (RR) Theme (eYRC 2019-20)
*        		===============================================
*
*  This script is to implement Task 1C of Rapid Rescuer (RR) Theme (eYRC 2019-20).
*  
*  This software is made available on an "AS IS WHERE IS BASIS".
*  Licensee/end user indemnifies and will keep e-Yantra indemnified from
*  any and all claim(s) that emanate from the use of the Software or 
*  breach of the terms of this agreement.
*  
*  e-Yantra - An MHRD project under National Mission on Education using ICT (NMEICT)
*
*****************************************************************************************
'''


# Team ID:			RR-6202
# Author List:		[ Names of team members worked on this file separated by Comma: Name1, Name2, ... ]
# Filename:			task_1c.py
# Functions:		computeSum
# 					[ Comma separated list of functions in this file ]
# Global variables:	None
# 					[ List of global variables defined in this file ]


# Import necessary modules
import cv2
import numpy as np
import os
import sys
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Flatten, Dropout
from keras.layers.advanced_activations import LeakyReLU



#############	You can import other modules here	#############



#################################################################


# Function Name:	computeSum
# Inputs: 			img_file_path [ file path of image ]
# 					shortestPath [ list of coordinates of shortest path from initial_point to final_point ]
# Outputs:			digits_list [ list of digits present in the maze image ]
# 					digits_on_path [ list of digits present on the shortest path in the maze image ]
# 					sum_of_digits_on_path [ sum of digits present on the shortest path in the maze image ]
# Purpose: 			the function takes file path of original image and shortest path in the maze image
# 					to return the list of digits present in the image, list of digits present on the shortest
# 					path in the image and sum of digits present on the shortest	path in the image
# Logic:			[ write the logic in short of how this function solves the purpose ]
# Example call: 	digits_list, digits_on_path, sum_of_digits_on_path = computeSum(img_file_path, shortestPath)

def computeSum(img_file_path, shortestPath):

	"""
	Purpose:
	---
	the function takes file path of original image and shortest path as argument and returns list of digits, digits on path and sum of digits on path

	Input Arguments:
	---
	`img_file_path` :		[ str ]
		file path of image
	`shortestPath` :		[ list ]
		list of coordinates of shortest path from initial_point to final_point

	Returns:
	---
	`digits_list` :	[ list ]
		list of all digits on image
	`digits_on_path` :	[ list ]
		list of digits adjacent to the path from initial_point to final_point
	`sum_of_digits_on_path` :	[ int ]
		sum of digits on path

	Example call:
	---
	original_binary_img = readImage(img_file_path)

	"""

	digits_list = []
	digits_on_path = []
	sum_of_digits_on_path = 0

	#############  Add your Code here   ###############
	img = cv2.imread(img_file_path)
	gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	ret , binary_img = cv2.threshold(gray_img,255//2,255,cv2.THRESH_BINARY)
	copy = binary_img
	dim = (40,40)
	temp = np.zeros(shape = dim,dtype = np.uint8)

	
	x_test_28 , digit_containing_cells = input_data_converter(binary_img)    #28*28 cell
	#digitOnPath(shortestPath,digit_containing_cells)

	

	# for i in range(len(x_test_28)):
	# 	cv2.imshow('Value',x_test_28[i])
	# 	cv2.waitKey(0)
	# 	cv2.destroyWindow('Value')

	
	""" mnist = tf.keras.datasets.mnist
	(x_train_28, y_train),(x_test_28, y_test) = mnist.load_data()                      #Default data of 28*28 pixels

	x_train = np.zeros(shape = (6000,40,40),dtype = np.uint8)
	x_test = np.zeros(shape = (10000,40,40),dtype = np.uint8)

	for i in range(6000):
		x_train[i] = cv2.resize(x_train_28[i],dim)                                   #Converting to 40*40 pixels

	for i in range(10000):
		x_test[i] = cv2.resize(x_train_28[i],dim)    """
	

	mnist = tf.keras.datasets.mnist
	(x_train, y_train), (x_test, y_test) = mnist.load_data()   # 28x28 numbers of 0-9
	x_train = tf.keras.utils.normalize(x_train, axis=1).reshape(x_train.shape[0], -1)

	

	
	x_test_28 = tf.keras.utils.normalize(x_test_28, axis=1).reshape(x_test_28.shape[0], -1)
	x_test = tf.keras.utils.normalize(x_test, axis=1).reshape(x_test.shape[0], -1)

	my_model = build_model(x_train,y_train)



	example_result = my_model.predict(x_test_28)
	for i in range(len(example_result)):
		digits_list.append(np.argmax(example_result[i]))
	# cv2.imshow('Value',x_test[0])
	# cv2.waitKey(0)
	# cv2.destroyWindow('Value')

	digits_on_path	= digitOnPath(shortestPath,digit_containing_cells,copy)

 


	###################################################

	return digits_list, digits_on_path, sum_of_digits_on_path


#############	You can add other helper functions here		#############

def cell_with_numbers(binary_img):

	cells = set()
	sum = 0

	for i in range(0,400,40):
		for j in range(0,400,40):
			for check_x in range(i+ 10,i + 30,1):
				for check_y in range(j+ 10,j + 30,1):
					if binary_img[check_x][check_y] == 0:
						cells.add((i//40,j//40))
						break

	required_list = list(cells)
	return required_list

def input_data_converter(binary_img):

	digit_containing_cells = cell_with_numbers(binary_img)
	temp = np.zeros(shape = (40,40),dtype = np.uint8)
	x_t = np.zeros(shape = (len(digit_containing_cells),40,40),dtype = np.uint8)
	x_tf = np.zeros(shape = (len(digit_containing_cells),28,28),dtype = np.uint8)
	count = 0

	for elements in digit_containing_cells:
		row_x , col_y = elements
		for i in range(row_x*40,row_x*40 + 40,1):
			for j in range(col_y*40,col_y*40 + 40,1):
				temp[i%40][j%40] =  255 - binary_img[i][j]
				
		x_t[count] = temp
		count += 1

	for i in range(len(digit_containing_cells)):           #Cleansing the input and converting it to 28*28 size
		x_tf[i] = cv2.resize(x_t[i],(28,28))
		for j in range(3):
			for k in range(28):
				x_tf[i][j][k] = 0
		for j in range(25,28,1):
			for k in range(28):
				x_tf[i][j][k] = 0
		for j in range(28):
			for k in range(3):
				x_tf[i][j][k] = 0
		for j in range(28):
			for k in range(25,28,1):
				x_tf[i][j][k] = 0
		
				
			

	# for i in range(len(digit_containing_cells)):
	# 	cv2.imshow('Values',x_tf[i])
	# 	cv2.waitKey(0)
	# 	cv2.destroyWindow('Values')

	return x_tf , digit_containing_cells

def build_model(x_train,y_train):
	model = tf.keras.models.Sequential()
	model.add(tf.keras.layers.Flatten())   
	model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu, input_shape= x_train.shape[1:]))
	model.add(tf.keras.layers.Dense(64, activation=tf.nn.sigmoid))
	model.add(tf.keras.layers.Flatten())  
	model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

	model.compile(optimizer='adam',  
              loss='sparse_categorical_crossentropy',  
              metrics=['accuracy'])  

	model.fit(x_train,y_train,epochs = 1)

	return model

	

	

def digitOnPath(shortestPath,digit_containing_cells,copy):
	toSum = set()
	for cells in shortestPath:
		row_cell , col_cell = cells

		#To get original coordinates in the image
		row_cell = row_cell*40
		col_cell = col_cell*40

		if col_cell > 0 and copy[row_cell + 40//2 ][col_cell - 1] == 255:               #Checking if bot can go left 
			if (row_cell/40 - 1,col_cell/40) in digit_containing_cells:
				toSum.add((row_cell/40 - 1,col_cell/40))
				print()
				print(row_cell + 40//2 ,col_cell - 1)

				print()
				print(cells)
				print('left add kiya ')
				print((row_cell/40 - 1,col_cell/40))
				print()



		if copy[row_cell + 40//2 ][col_cell + 39] == 255:                       		 #Checking if bot can go right 
			if (row_cell/40 + 1,col_cell/40) in digit_containing_cells:
				toSum.add((row_cell/40 + 1,col_cell/40))
				print('right add kiya ')
				print((row_cell/40 + 1,col_cell/40))
			

		if row_cell < 9 and copy[row_cell + 40 + 1][col_cell + 40//2 ] == 255:                       		 #Checking if bot can go down
			if (row_cell/40,col_cell/40 + 1) in digit_containing_cells:
				toSum.add((row_cell/40,col_cell/40 + 1))
				print('down add kiya ')
				print((row_cell/40 ,col_cell/40 + 1))
			
		if  copy[row_cell + 1][col_cell + 40//2] == 255:                                       #Checking if bot can go up
			if (row_cell/40,col_cell/40 - 1) in digit_containing_cells:
				toSum.add((row_cell/40,col_cell/40 - 1))
				print('up add kiya ')
				print((row_cell/40 ,col_cell/40 - 1))
			

	print(toSum)		
	return toSum
		


		
			

#########################################################################


# NOTE:	YOU ARE NOT ALLOWED TO MAKE ANY CHANGE TO THIS FUNCTION
# 
# Function Name:	main
# Inputs:			None
# Outputs: 			None
# Purpose: 			the function first takes 'maze00.jpg' as input and solves the maze by calling computeSum
# 					function, it then asks the user whether to repeat the same on all maze images
# 					present in 'task_1c_images' folder or not

if __name__ != '__main__':
	
	curr_dir_path = os.getcwd()

	# Importing task_1a and image_enhancer script
	try:

		task_1a_dir_path = curr_dir_path + '/../../Task 1A/codes'
		sys.path.append(task_1a_dir_path)

		import task_1a
		import image_enhancer

	except Exception as e:

		print('\ntask_1a.py or image_enhancer.pyc file is missing from Task 1A folder !\n')
		exit()

if __name__ == '__main__':
	
	curr_dir_path = os.getcwd()
	img_dir_path = curr_dir_path + '/../task_1c_images/'				# path to directory of 'task_1c_images'
	
	file_num = 0
	img_file_path = img_dir_path + 'maze0' + str(file_num) + '.jpg'		# path to 'maze00.jpg' image file

	# Importing task_1a and image_enhancer script
	try:

		task_1a_dir_path = curr_dir_path + '/../../Task 1A/codes'
		sys.path.append(task_1a_dir_path)

		import task_1a
		import image_enhancer

	except Exception as e:

		print('\n[ERROR] task_1a.py or image_enhancer.pyc file is missing from Task 1A folder !\n')
		exit()

	# modify the task_1a.CELL_SIZE to 40 since maze images
	# in task_1c_images folder have cell size of 40 pixels
	task_1a.CELL_SIZE = 40

	print('\n============================================')

	print('\nFor maze0' + str(file_num) + '.jpg')

	try:
		
		original_binary_img = task_1a.readImage(img_file_path)
		height, width = original_binary_img.shape

	except AttributeError as attr_error:
		
		print('\n[ERROR] readImage function is not returning binary form of original image in expected format !\n')
		exit()

	
	no_cells_height = int(height/task_1a.CELL_SIZE)					# number of cells in height of maze image
	no_cells_width = int(width/task_1a.CELL_SIZE)					# number of cells in width of maze image
	initial_point = (0, 0)											# start point coordinates of maze
	final_point = ((no_cells_height-1),(no_cells_width-1))			# end point coordinates of maze

	try:

		shortestPath = task_1a.solveMaze(original_binary_img, initial_point, final_point, no_cells_height, no_cells_width)

		if len(shortestPath) > 2:

			img = image_enhancer.highlightPath(original_binary_img, initial_point, final_point, shortestPath)
			
		else:

			print('\n[ERROR] shortestPath returned by solveMaze function is not complete !\n')
			exit()
	
	except TypeError as type_err:
		
		print('\n[ERROR] solveMaze function is not returning shortest path in maze image in expected format !\n')
		exit()

	print('\nShortest Path = %s \n\nLength of Path = %d' % (shortestPath, len(shortestPath)))

	digits_list, digits_on_path, sum_of_digits_on_path = computeSum(img_file_path, shortestPath)

	print('\nDigits in the image = ', digits_list)
	print('\nDigits on shortest path in the image = ', digits_on_path)
	print('\nSum of digits on shortest path in the image = ', sum_of_digits_on_path)

	print('\n============================================')

	cv2.imshow('canvas0' + str(file_num), img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

	choice = input('\nWant to run your script on all maze images ? ==>> "y" or "n": ')

	if choice == 'y':

		file_count = len(os.listdir(img_dir_path))

		for file_num in range(file_count):

			img_file_path = img_dir_path + 'maze0' + str(file_num) + '.jpg'		# path to 'maze00.jpg' image file

			print('\n============================================')

			print('\nFor maze0' + str(file_num) + '.jpg')

			try:
				
				original_binary_img = task_1a.readImage(img_file_path)
				height, width = original_binary_img.shape

			except AttributeError as attr_error:
				
				print('\n[ERROR] readImage function is not returning binary form of original image in expected format !\n')
				exit()

			
			no_cells_height = int(height/task_1a.CELL_SIZE)					# number of cells in height of maze image
			no_cells_width = int(width/task_1a.CELL_SIZE)					# number of cells in width of maze image
			initial_point = (0, 0)											# start point coordinates of maze
			final_point = ((no_cells_height-1),(no_cells_width-1))			# end point coordinates of maze

			try:

				shortestPath = task_1a.solveMaze(original_binary_img, initial_point, final_point, no_cells_height, no_cells_width)

				if len(shortestPath) > 2:

					img = image_enhancer.highlightPath(original_binary_img, initial_point, final_point, shortestPath)
					
				else:

					print('\n[ERROR] shortestPath returned by solveMaze function is not complete !\n')
					exit()
			
			except TypeError as type_err:
				
				print('\n[ERROR] solveMaze function is not returning shortest path in maze image in expected format !\n')
				exit()

			print('\nShortest Path = %s \n\nLength of Path = %d' % (shortestPath, len(shortestPath)))

			digits_list, digits_on_path, sum_of_digits_on_path = computeSum(img_file_path, shortestPath)

			print('\nDigits in the image = ', digits_list)
			print('\nDigits on shortest path in the image = ', digits_on_path)
			print('\nSum of digits on shortest path in the image = ', sum_of_digits_on_path)

			print('\n============================================')

			cv2.imshow('canvas0' + str(file_num), img)
			cv2.waitKey(0)
			cv2.destroyAllWindows()

	else:

		print('')


