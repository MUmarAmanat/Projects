from inti_param import InitParam
from optimization import  Optimization
from load_data_helper import load_cat_vs_dog_data
from load_data_helper import load_cat_vs_dog_test_data
import pandas as pd
# import matplotlib.pyplot as plt
from prediction import predict
learning_rate = 0.005

TRAIN_DATA = '\\input\\train\\'
TEST_DATA = '\\input\\test\\'
FILE_COUNT = 10000

# Train on dataset
print('Training..........')
X_train, Y_train= load_cat_vs_dog_data(TRAIN_DATA, FILE_COUNT, shuffle=True)
# X_train = X_train.T
# Y_train = Y_train.T
w,b = InitParam.initialize_params(X_train.shape[0])
params, grads, cost= Optimization.optimize(w, b, X_train, Y_train, 500, learning_rate, False)


FILE_COUNT = 12500
print('Predicting..........')
X_test, id_list = load_cat_vs_dog_test_data(TEST_DATA, FILE_COUNT)
Y_Predict = predict(params['w'], params['b'], X_test)

my_solution = pd.DataFrame(Y_Predict.T, id_list, columns = ["Id, Label"])
my_solution.to_csv("my_solution_one.csv", index_label = ["Id"])
# print(my_solution)































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