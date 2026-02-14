
import pandas as pd
import numpy as np
import yaml
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
import os
import mlflow

import warnings
warnings.filterwarnings('ignore')

os.chdir(r"D:\1.Work\WorkStation\WorkSpace\Projects\Customer-Churn-Prediction")  # Set the working directory to the project folder to avoid issues with file paths
cwd = os.getcwd()
yaml_path = os.path.join(cwd,"config.yaml")


with open(yaml_path) as f:
    config = yaml.safe_load(f)

data_path = os.path.join(cwd,'data','processed','preprocessed.csv')

data = pd.read_csv(data_path)
X = data.iloc[:,:-1]
y = data.iloc[:,[-1]]

mlflow.set_experiment("Customer_Churn_Prediction")

with mlflow.start_run():

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=config['test_train_split']['test_size'])
    mlflow.log_params(config.get('test_train_split',{}))

    if config.get('model',{}) == 'random_forest':
        mlflow.log_params(config.get('models',{}).get('random_forest',{}))

        model = RandomForestClassifier(**config['models']['random_forest'])
    
    elif config.get('model',{}) == 'xgboost':
        mlflow.log_params(config.get('models',{}).get('xgboost',{}))

        model = GradientBoostingClassifier(**config['models']['xgboost'])
    
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="macro")
    recall = recall_score(y_test, y_pred, average="macro")

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)

    mlflow.sklearn.log_model(model,config.get('model'))

