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
import matplotlib.image as mpimg
import os
from scipy.ndimage import filters
import urllib
import re
from shutil import copy2

def linear_reg():
	

# Algorithm to pick training set, validation set and testing set
def sep_img(actor=None,force=None):
	# if no actor is specified, we will do sep_img to all actors
	if actor is None:
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
# sep_img(actor=None, force=1)


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
"""
