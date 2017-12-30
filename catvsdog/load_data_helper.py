import os
import cv2
import numpy as np
import random
from scipy import misc
from scipy import ndimage
import cv2
IMAGE_HEIGHT = IMAGE_WIDTH = 64
# from scipy import skimage.transform.resize

def load_cat_vs_dog_data(data_dir, file_count, shuffle=True):
	path = os.getcwd() + data_dir
	all_images = os.listdir(path)
	random.shuffle(all_images)
	all_images = all_images[:file_count]

	images = np.zeros((file_count, 64*64*3))
	#reading images and making X matrix
	for i in range(file_count):
		imgnd_origin = cv2.imread(path+all_images[i])
		imgnd_resized = cv2.resize(imgnd_origin, (IMAGE_HEIGHT, IMAGE_WIDTH), interpolation=cv2.INTER_CUBIC)
		imgnd_flatten = imgnd_resized.reshape(-1,1).T
		images[i] = imgnd_flatten

	images = images/255.0
	## labels from filenames
	labels_list = ["dog" in name for name in all_images] #dog=0 cat=1
	labels = np.array(labels_list, dtype='int8').reshape(file_count, 1)

	if shuffle:
		permutation = list(np.random.permutation(labels.shape[0]))
		labels = labels[permutation, :]
		images = images[permutation, :]

	# print(permutation)
	return images.T, labels.T

def load_cat_vs_dog_test_data(data_dir, file_count):
	path = os.getcwd() + data_dir
	all_images = os.listdir(path)
	random.shuffle(all_images)
	all_images = all_images[:file_count]
	id_list = []
	images = np.zeros((file_count, 64*64*3))
	#reading images and making X matrix
	for idx, i in enumerate(range(file_count)):
		imgnd_origin = cv2.imread(path+all_images[i])
		imgnd_resized = cv2.resize(imgnd_origin, (IMAGE_HEIGHT, IMAGE_WIDTH), interpolation=cv2.INTER_CUBIC)
		imgnd_flatten = imgnd_resized.reshape(-1,1).T
		images[i] = imgnd_flatten
		id_list.append(idx+1)

	images = images/255.0


	# print(permutation)
	return images.T, id_list


# TRAIN_DIR = '\\input\\train'
# TEST_DIT = '\\input\\test'
# IMAGE_HEIGHT = IMAGE_WIDTH = 64

# def load_cat_vs_dog_data(data_dir = TRAIN_DIR, file_count=1000, shuffle=True):
# 	path = os.getcwd() + data_dir;
# 	all_filenames = os.listdir(path)

# 	random.shuffle(all_filenames)
# 	filenames = all_filenames[:file_count]

# 	#preprocessing for images
# 	images = np.zeros((file_count, IMAGE_HEIGHT*IMAGE_WIDTH*3))
# 	for i in range(file_count):
# 		imgnd_origin = cv2.imread(path+filenames[i])
# 		imgnd_resized = cv2.resize(imgnd_origin, (IMAGE_HEIGHT, IMAGE_WIDTH), interpolation=cv2.INTER_CUBIC)
# 		imgnd_flatten = imgnd_resized.reshape(1,-1)
# 		images[i] = imgnd_flatten

# 	## labels from filenames
# 	labels_list = ["dog" in filename for filename in filenames]
# 	labels = np.array(labels_list, dtype='int8').reshape(file_count, 1)

# 	## shuffle
# 	if shuffle:
# 		permutation = list(np.random.permutation(labels.shape[0]))
# 		labels = labels[permutation, :]
# 		images = images[permutation, :]

# 	## normalization
# 	images = images/255.0

# 	return images.T, labels.T




