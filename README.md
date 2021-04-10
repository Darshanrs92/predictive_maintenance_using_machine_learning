1.Introduction
The increasing availability of data is helping the industries to effectively schedule the maintenance activities. Machine learning approaches are becoming effective in this area facilitated by the growing capabilities of hardware and cloud based solutions. Effective management of maintenance activities helps in reducing the downtime cost and maximizing equipment lifetime. Machine Learning based PDM can be divided into two categories, a. Supervised approach b. Unsupervised approach.
a. Supervised approach :In this case, information about occurrence of failure event will be present in the dataset
b. Unsupervised approach :In this case, information about occurrence of failure event will not present in the dataset.
In this work, since the occurrence of failure event is present in the dataset, We used a supervised learning approach. any supervised machine learning approach should contain the dataset in the form, S = {x_i,y_i}, where x_i is a p dimensional vector and each dimension represents a feature, y_i is a failure related information which can take a continuous value or categorical value, In this work, y_i takes a categorical values.
2. Data
Source: https://www.kaggle.com/nphantawee/pump-sensor-data
This dataset is collected from the various sensors mounted on the water pump, water pump is used in conjunction with the electrical motor. Water pump lifts the water from bottom to top utilizing the mechanical energy of electrical motor
This dataset contains a timestamp column, for every timestamp, we have data from 51 sensors (sensor names are not known here)
This dataset also contains a column called 'Unnamed: 0' which holds the index values for each row
This dataset contains a 'machine_status' column which gives the failure related information for every timestamp
This dataset contains 220320 rows and 55 columns. In this dataset, machine_status column contains three categorical values, 1. NORMAL, 2. RECOVERING 3. BROKEN

2.1 Business Problem
Using the data available from various sensors, we need to predict the future health status of the machine, so that the maintenance activities can be effectively performed.
2.1.1 Modification of dataset to solve business problem
In this dataset, there are three classes in the machine_status column 1. NORMAL, 2. RECOVERING 3. BROKEN. We are predicting whether the machine will become faulty or not in the future time interval using this machine_status column as a target variable. Dataset is modified as follows:
We are considering the data points with machine status=RECOVERING, as machine status = 'BROKEN' since the recovering state is not the stable operating state of the machine
In the modified dataset, There will be two classes 1.Normal(encoded as 0), 2. Broken(encoded as 1). So, this will become a binary classification problem
The above modification is done as a part of the data preprocessing step. Further modification of the machine_status column to perform predictive analysis, that is to predict whether the machine will become faulty or not in the next 'm' timestamps is done after the best value of 'm' is obtained using hyperparameter tuning for 'm'

2.2 Machine Learning problem
At any given timestamp, predict whether the machine will become faulty or not in the next 'm' timestamps.
2.2.1 Multiple classifier approach to solve the Machine Learning problem
Let 'N' be the total maintenance cycles (each maintenance cycle consists of 
N-1 NORMAL points and 1 BROKEN point), and let 'n' be the total data points available in the dataset. Each maintenance cycle ends with a failure. If during the run, the fault under consideration takes place, that observation is considered as BROKEN(F) and NORMAL (NF) otherwise. N samples are available for class F and n-N samples are available for class NF.
Y(i) = y_i= { F if iteration i is faulty, 
 NF if iteration i is not faulty} - - (1)
In this work, a simple classification problem presented in (1) is modified as follows in context to multiple classifier approach.
Multiple classifier approach:
Instead of only labelling the last iteration of a maintenance cycle as F (as it is available in dataset), we label as F the last 'm' timestamps, from a PDM perspective. (after doing this modification, we can train a classifier and from this classifier, we can predict whether the machine will become faulty or not in next 'm' timestamps from the current timestamp). This helps in reducing the imbalance in the data as well. In this work, 'p' different values of the horizon 'm' are considered to formulate 'p' different classifiers which becomes the multiple classifier approach and the best value of 'm' is obtained by hyperparameter tuning for 'm' by considering different values of 'm'. In this work different values of 'm' considered are 60,120,180,240. 60 constitutes to 1 hour, 120 constitutes to 2 hours, 180 constitutes to 3 hours and 240 constitutes to 4 hours. Since KNN is used as a ML model to solve this problem, we also want to tune for the best value of 'K' by considering different values of 'K', so here we have done the hyperparameter tuning for both 'm' and 'K'. After obtaining the best value of 'm' and 'K', the target variable (machine_status) is modified as stated above and the classifier is trained using the best 'K'. This becomes the best classifier. Maintenance event is triggered when the best classifier (classifier with maximum F1-score) outputs the label F classification.
3. Performance metric(s)
Confusion matrix
F1-score (macro average)

4. Train, CV and Test Datasets
We have Split the dataset randomly into three parts train, cross validation and test with 49%,21%, 30% of data respectively

5. Exploratory Data analysis
5.1 Univariate analysis
Violin plots and PDF are plotted for the 10 most correlated features with the target variable
Percentile values are printed for 10 most correlated features with the target variable

5.2 Bivariate analysis:
Scatter plot is plotted between the 5 pairs of features, 5 pairs of features are selected based on the lowest correlation values among the features

6. Feature Engineering
Applied a simple moving average filter on each feature and added as the new features
Added five polynomial features
Added one average feature
In addition to the above feature engineering, All the columns are normalized using min-max normalization and the missing values are filled with the median values of that column with same class label

7. Machine Learning Model
Various ML models are tried for the above problem and found that KNN works best, so KNN is used to train and test the model, all the models training and results can be seen in the below provided github link
Model gave the best performance with 'm' = 180 timestamps (3 hours as per dataset), and k=5, so these values are used to train the model
In this work, for a given input, we are predicting whether machine will become faulty or not in next 3 hours from the current timestamp

8. Results
F1- score obtained : 0.9986

python code : please refer final_knn_model.ipynb for complete code with KNN
9. References:
https://www.appliedaicourse.com/
https://www.kaggle.com/nphantawee/pump-sensor-data
