import pandas as pd
import sklearn
import numpy as np
# from prettytable import PrettyTable
# import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer
import seaborn as sns
from sklearn.linear_model import LogisticRegression
import joblib
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_recall_curve, f1_score, recall_score, confusion_matrix
import pickle
import logging
log = logging.getLogger(__name__)
logging.basicConfig( format='%(asctime)s %(levelname)-8s %(message)s', level= logging.DEBUG)
print("INIZIATO")

log.info("Importing file")
df = pd.read_csv("data/mushroom_overload.csv",nrows=10000)

print(df.groupby("class").count())
log.info("Importing OK")
numeric_features = ["cap-diameter", "stem-height", "stem-width"]
numeric_transformer = Pipeline(
    steps=[ ("imputer",SimpleImputer(missing_values=np.nan, strategy='mean')),
           ("scaler", MinMaxScaler())]
)
categorical_features = df.select_dtypes("object").columns.tolist()
categorical_features
categorical_transformer = Pipeline(
    steps=[
        ("imputer",SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder())        
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)
clf = Pipeline(
    steps=[("preprocessor", preprocessor),       
            ("logreg", LogisticRegression())]
)
X=df#.drop(columns= ["class"])
y=df["class"]
x_train, x_test, y_train, y_test = train_test_split(X,y,test_size=0.2,stratify=y)
clf.fit(x_train, y_train)
print("model score: %.3f" % clf.score(x_test, y_test))

pickle.dump(clf, open("pipeline.pkl","wb"))