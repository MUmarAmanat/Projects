import pandas as pd
import numpy as np

class TitanicData:
    @staticmethod
    def load_data():
        data_raw = pd.read_csv(os.getcwd() + "\\input\\train.csv")
        train = data_raw.copy()
        data_val = pd.read_csv(os.getcwd() + "\\input\\train.csv")
        test = data_val.copy()
        train.Sex[train.Sex == 'male'] = 0
        train.Sex[train.Sex == 'female'] = 1
        train.Age.fillna(train.Age.median(), inplace=True)
        remove_column = ['Ticket', 'Cabin', 'Name', 'PassengerId']
        train.drop(remove_column, axis=1, inplace=True)
        train["Embarked"] = train.Embarked.fillna('S')
        train["Embarked"][train["Embarked"] == "S"] = 0
        train.Embarked[train.Embarked == "C"] = 1
        train.Embarked[train.Embarked == "Q"] = 2
        Y = train.Survived.values
        # Y = Y.reshape((len(Y), 1))
        train.drop(['Survived'], axis=1, inplace=True)
        Y = np.array(Y)
        X = np.array(train.values).astype(float)
        X = X.T
        return (X, Y)