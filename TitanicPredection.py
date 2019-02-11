from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import accuracy_score


dataset =pd.read_csv('TitanicPreprocessed.csv')
print(dataset.head())

y = dataset['Survived']
X = dataset.drop(['Survived'], axis = 1)

train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.25, random_state=0)

parameters = {'bootstrap': True,
              'min_samples_leaf': 3,
              'n_estimators': 50,
              'min_samples_split': 10,
              'max_features': 'sqrt',
              'max_depth': 6,
              'max_leaf_nodes': None}

RF_model = RandomForestClassifier(**parameters)

RF_model.fit(train_X, train_y)

feat_imps=pd.DataFrame(list(zip(train_X.columns,RF_model.feature_importances_)),columns=["feature","Importance"])
feat_imps.sort_values(by="Importance",ascending=False,inplace=True)
print(feat_imps)
#print(RF_model.feature_importances_)
RF_predictions = RF_model.predict(test_X)

score = accuracy_score(test_y ,RF_predictions)
print(score)
