import os
import numpy as np
import scipy
import matplotlib.pyplot as plt
from scipy import ndimage
from prediction import predict


def image_predict(params, classes, num_px):
	my_image = "my_image.jpg"  
	path = os.getcwd() + "\\images\\" + my_image
	image = np.array(ndimage.imread(path, flatten=False))
	my_image = scipy.misc.imresize(image, size=(num_px,num_px)).reshape((1, num_px*num_px*3)).T
	my_predicted_image = predict(params['w'], params['b'], my_image)
	# plt.imshow(image)
	print("y = " + str(np.squeeze(my_predicted_image)) + ", Algorithm predicts a \"" + 
			classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\" picture.")
	