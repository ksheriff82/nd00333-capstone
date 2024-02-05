*NOTE:* This file is a template that you can use to create the README for your project. The *TODO* comments below will highlight the information you should be sure to include.

# Your Project Title Here

This capstone project is for creating the model using azure ml and hyper drive. This project uses the heard disease dataset from kaggle and uses azure studio for creating the model and use the best model for deployment and uses the model to predict the heart disease possibilities for given data.


## Dataset
https://raw.githubusercontent.com/ksheriff82/nd00333-capstone/master/starter_file/heart.csv

### Overview
This data is downloaded from kaggle. the data set has various parameters like Age, Gender, ChestPainType,RestingBP,Cholesterol,FastingBS,RestingECG,MaxHR,ExerciseAngina,Oldpeak,ST_Slope and predicts if those combinations could cause heart disease ot not.

### Task
This dataset can be used to build the model and train it, so that it can be improved to predict the possibility of heart disease when the patient data is provided for the above said metrics.

### Access
The data is loaded in github and stored as dataset in the Azure studio. The azureml script and hyperdrive would access the dataset that is uploaded to the studio.

## Automated ML
automl_settings = {
    "experiment_timeout_minutes" : 25,
    "max_concurrent_iterations": 4,
    "n_cross_validations": 5,
    "primary_metric": 'accuracy',
}

The above settings are used so that expirement dont run more than 25 mins, and ensure we have conncurrent run of 4 to save time also updated the cross validation to 5. This is classification experiment and accuracy is the primary metric that will be used for meaasuring the correctness of the model![runwidget_azureml](https://github.com/ksheriff82/nd00333-capstone/assets/43680905/142e40fe-438f-4d96-a5c5-bacec784267f)



### Results
We got the following metrics from the best model for Azure ML
Run Metrics
Accuracy
0.88130
AUC macro
0.93450
AUC micro
0.93673
AUC weighted
0.93450
Average precision score macro
0.93414
Average precision score micro
0.93561
Average precision score weighted
0.93515
Balanced accuracy
0.87715
F1 score macro
0.87901
F1 score micro
0.88130
F1 score weighted
0.88094
Log loss
0.40832
Matthews correlation
0.75993
Norm macro recall
0.75430
Precision score macro
0.88282
Precision score micro
0.88130
Precision score weighted
0.88246
Recall score macro
0.87715
Recall score micro
0.88130
Recall score weighted
0.88130
Weighted accuracy
0.88522


![azureml_bestmodel_metrics](https://github.com/ksheriff82/nd00333-capstone/assets/43680905/616142d2-42fc-425b-9dbb-6ad49376197a)

This metrics can be improved by providing more test data. Setup pipeline to run the model job as we get more data, that would help in improving the model for better predications.

Also, the data provided can be cleaned as well to remove any outliers.

## Hyperparameter Tuning
As this is a classification experiment, we  choose the following  parameters for hyper drive run, with early termination policy to prevent it running from longer duration. And also configured the iterations to 10 for better model.

We used scikit libraries and also utilized the train.py script which is used for cleaning the submitted dataset.

param_sampling = RandomParameterSampling(
    { 
        '--C': choice(0.1, 1.0,10),
        '--max_iter': choice(['5', '10', '20'])
    }
)

early_termination_policy = BanditPolicy(evaluation_interval=1, slack_factor=0.1, delay_evaluation=0)

![hyperdriver_run](https://github.com/ksheriff82/nd00333-capstone/assets/43680905/4839b2dc-41bb-40e0-8145-29373e847150)

### Results
![hyperdrive_metrics](https://github.com/ksheriff82/nd00333-capstone/assets/43680905/ed61b495-abbf-40b9-bdd5-d81a963704c4)

We got Accuracy and Regularization Strength from the results. The model can be improved by using several other sampling methods are more data.

## Model Deployment
The azureml model is chosen for deployment. the model is registered as capstone model.

![Screenshot 2024-02-04 210623](https://github.com/ksheriff82/nd00333-capstone/assets/43680905/c573727e-c272-4c81-97e5-87b03dcd912f)

The service is deployed as real time webservice with the name myservice. Below are the screenshot for the endpoint and swagger endpoint.
![Screenshot 2024-02-04 210708](https://github.com/ksheriff82/nd00333-capstone/assets/43680905/9e773336-900a-42c0-bc7f-103b901dce14)
![Screenshot 2024-02-04 210819](https://github.com/ksheriff82/nd00333-capstone/assets/43680905/d121542b-2f3c-4c9e-8961-83abde868de6)

![Screenshot 2024-02-04 210841](https://github.com/ksheriff82/nd00333-capstone/assets/43680905/a82adc24-425a-4561-80e0-f680b8c7a1b9)



## Screen Recording
https://www.youtube.com/watch?v=8fM_8FqDgmA
## Standout Suggestions
*TODO (Optional):* This is where you can provide information about any standout suggestions that you have attempted.
