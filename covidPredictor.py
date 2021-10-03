# Importing essential libraries
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Loading the dataset
covid = pd.read_csv('Covid Dataset.csv')

label_encoder = LabelEncoder()
encode_covid = covid.iloc[:,0:23]

for i in encode_covid:
  encode_covid[i]=label_encoder.fit_transform(encode_covid[i])

#model building
X = encode_covid.drop(['Case Id','NIC','COVID-19'],axis=1)
y = encode_covid['COVID-19']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# Creating Random Forest Model
classifier = RandomForestClassifier(n_estimators=20)
classifier.fit(X_train, y_train)

# Creating a pickle file for the classifier
filename = 'covid-prediction-rfc-model.pkl'
pickle.dump(classifier, open(filename, 'wb'))

