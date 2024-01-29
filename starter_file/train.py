
import argparse
import os
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LogisticRegression
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from azureml.core.run import Run
from azureml.data.dataset_factory import TabularDatasetFactory
from azureml.core import Workspace, Dataset

from azureml.core import Workspace, Experiment
from azureml.core import Environment, ScriptRunConfig




def clean_data(data):
    
    
    x_df = data.drop(["HeartDisease"], axis=1)
    x_df["Sex"] = x_df.Sex.apply(lambda s: 1 if s == "M" else 0)
    x_df["ChestPainType"] = x_df.ChestPainType.apply(lambda s: 0 if s == "ASY" else (1 if s == "ATA" else (2 if s == "NAP" else 3)))
    x_df["RestingECG"] = x_df.RestingECG.apply(lambda s: 0 if s == "Normal"  else (1 if s == "ST" else 2))
    x_df["ExerciseAngina"] = x_df.ExerciseAngina.apply(lambda s: 1 if s == "Y"  else 0)
    x_df["ST_Slope"] = x_df.ST_Slope.apply(lambda s: 1 if s == "UP"  else(2 if s == "DOWN" else  0))
    
    
    y_df = data["HeartDisease"]
    
    # Clean and one hot encode data
    #x_df = data.drop(HeartDisease)
    

    #y_df = x_df.pop("HeartDisease")
    return x_df, y_df

def main():
    # Add arguments to script
    parser = argparse.ArgumentParser()

    parser.add_argument('--C', type=float, default=1.0, help="Inverse of regularization strength. Smaller values cause stronger regularization")
    parser.add_argument('--max_iter', type=int, default=100, help="Maximum number of iterations to converge")

    args = parser.parse_args()

    run = Run.get_context()

    run.log("Regularization Strength:", float(args.C))
    run.log("Max iterations:", int(args.max_iter))

    # Create TabularDataset using TabularDatasetFactory
    
    #subscription_id = '976ee174-3882-4721-b90a-b5fef6b72f24'
    #resource_group = 'aml-quickstarts-250208'
    #workspace_name = 'quick-starts-ws-250208'

    #workspace = Workspace.from_config()

    #dataset = Dataset.get_by_name(workspace, name='heart_rate_failure_prediction') 
    

    #ds = TabularDatasetFactory.from_delimited_files("https://mlstrg250208.blob.core.windows.net/azureml-blobstore-9c6e7a8b-e82f-4444-b018-b97d722da9f0/UI/2024-01-21_222438_UTC/heart.csv")  
    
    df = pd.read_csv("./heart.csv")
    x, y = clean_data(df)


    # TODO: Split data into train and test sets.

    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.33)
       

  

    model = LogisticRegression(C=args.C, max_iter=args.max_iter).fit(x_train, y_train)

    accuracy = model.score(x_test, y_test)
    run.log("Accuracy", np.float(accuracy))

if __name__ == '__main__':
    main()

