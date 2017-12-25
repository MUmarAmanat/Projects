from inti_param import InitParam
from optimization import  Optimization
import numpy as np
from image_prediction import image_predict
import matplotlib.pyplot as plt


learning_rate = 0.005

from cat_noncat import CatNonCat
X, Y, classes, num_px = CatNonCat.load_data()
w,b = InitParam.initialize_params(X.shape[0])
params, grads, cost= Optimization.optimize(w, b, X, Y, 2000, learning_rate, False)
image_predict(params, classes, num_px)
costs = np.squeeze(cost)
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations (per hundreds)')
plt.title("Learning rate =" + str(learning_rate))

# y_predict = predict(params['w'], params['b'], X)
# print("train accuracy: {} %".format(100 - np.mean(np.abs(y_predict - Y)) * 100))





































# #Common Model Algorithms
# from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process
# #Common Model Helpers
# from sklearn.preprocessing import OneHotEncoder, LabelEncoder
# from sklearn import feature_selection
# from sklearn import model_selection
# from sklearn import metrics
# #Visualization
# import matplotlib as mpl
# import matplotlib.pyplot as plt
# import matplotlib.pylab as pylab
# import seaborn as sns
# from pandas.tools.plotting import scatter_matrix
# #Configure Visualization Defaults
# #%matplotlib inline = show plots in Jupyter Notebook browser
# # %matplotlib inline
# import os
# mpl.style.use('ggplot')
# sns.set_style('white')
# pylab.rcParams['figure.figsize'] = 12,8
# import pandas as pd
# data_raw = pd.read_csv(os.getcwd()+ "\\input\\train.csv")
# data_val = pd.read_csv(os.getcwd()+ "\\input\\train.csv")
# data1 = data_raw.copy(deep=True)
# data_cleaner = [data1, data_val]
# data_raw.describe(include='all')
# print(data1['Age'].fillna(data1['Age'].median(), inplace=True))
# print(data1.isnull().sum())