from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

import joblib

data = pd.read_csv(
    "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")

target = data['Survived'].copy()
data.drop(['Survived'], axis=1, inplace=True)

features_numericas = ['Age', 'Fare', 'SibSp', 'Parch']
features_categoricas = ['Embarked', 'Sex', 'Pclass']
features_para_remover = ['Name', 'Cabin', 'Ticket', 'PassengerId']

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', MinMaxScaler())])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder())])

preprocessor = ColumnTransformer(
    transformers=[
        ('Features numericas', numeric_transformer, features_numericas),
        ('Features categoricas', categorical_transformer, features_categoricas),
        ('Feature para remover', 'drop', features_para_remover)
    ])


model = Pipeline([('preprocessor', preprocessor),
                 ('clf', LogisticRegression(solver='liblinear')),
                  ])

model.fit(data, target)

joblib.dump(model, "titanic_sklearn_pipelines.joblib")
