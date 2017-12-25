
import numpy as np
import os
from lr_utils import load_dataset


class CatNonCat():
	@staticmethod
	def load_data():
		# Loading the data (cat/non-cat)
		train_set_x_orig, Y_train, test_set_x_orig, test_set_y, classes = load_dataset()
		m_train = Y_train.shape[1]
		m_test = test_set_y.shape[1]
		num_px = train_set_x_orig.shape[1]
		train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
		test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T
		X = train_set_x_flatten/255.
		X_test = test_set_x_flatten/255.
		return X, Y_train