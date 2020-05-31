# Binary Classification of Fraudulent Payments using Stochastic Gradient Descent (SGD)




## Table Of Contents


- [Introduction](#introduction)
- [Setup Instructions](#setup-instructions)
  * [Log in to the AWS console and create a notebook instance](#log-in-to-the-aws-console-and-create-a-notebook-instance)
  * [Use git to clone the repository into the notebook instance](#use-git-to-clone-the-repository-into-the-notebook-instance)
- [Machine Learning Pipeline](#machine-learning-pipeline)
  * [Step 1 - Loading and exploring the data](#step-1---loading-and-exploring-the-data)
    + [Part A - Calculating the percentage of fraudulent data](#part-a---calculating-the-percentage-of-fraudulent-data)
  * [Step 2 - Splitting data into training and test sets](#step-2---splitting-data-into-training-and-test-sets)
  * [Step 3 - Binary classification](#step-3---binary-classification)
    + [Part A - Define a LinearLearner model](#part-a---define-a-linearlearner-model)
    + [Part B - Train the model](#part-b---train-the-model)
    + [Part C - Evaluate model performance](#part-c---evaluate-model-performance)
  * [Step 4 - Making improvements to the model](#step-4---making-improvements-to-the-model)
    + [Part A - Tune for higher recall](#part-a---tune-for-higher-recall)
    + [Part B - Manage class imbalance](#part-b---manage-class-imbalance)
    + [Part C - Tune for higher precision](#part-c---tune-for-higher-precision)
  * [Important - Deleting the endpoint](#important---deleting-the-endpoint)




## Introduction



IThis project focuses on the problem of building a binary classification model to identify and flag fraudulent credit card transactions based on provided, *historical* data. The nature of fraudulent transactions is that there are significantly more valid transactions than fraudulent ones so it requires a slightly different approach than a traditional binary classification problem. In this scenario, the SageMaker [LinearLearner](https://docs.aws.amazon.com/sagemaker/latest/dg/linear-learner.html) model was used since it is well-suited for binary classification tasks that may require managing a class imbalance in the training set. A lot of this project will focus on making model improvements, specifically, techniques will be used to:


1. **Tuning a model's hyperparameters** and aiming for a specific metric, such as higher recall or precision
2. **Managing class imbalance**, which is when there are many more training examples in one class than another (in this case, many more valid transactions than fraudulent)


SageMaker's LinearLearner model uses a distributed implementation of [Stochastic Gradient Descent (SGD)](https://en.wikipedia.org/wiki/Stochastic_gradient_descent) to train and allows for easy optimization to meet different objectives. In this project, the goal was to solve the fraud detection problem with a variety of different approaches, one that favored higher precision, one that favored higher recall, and another that was more balanced. In a [2016 study](https://nilsonreport.com/upload/content_promo/The_Nilson_Report_10-17-2016.pdf), it was estimated that credit card fraud was responsible for over 20 billion dollars in loss, worldwide. Accurately detecting cases of fraud is an ongoing area of research.


<img src=img/fraud_detection.png width=50% />


The dataset used in this project was payment fraud (Dal Pozzolo et al. 2015) from [Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud/data). The dimensionality of the features has already been reduced across the 250k+ transactions in this dataset, and each transaction is labeled as fraudulent or valid. Everything for this project was done on Amazon Web Services (AWS) and their SageMaker platform as the goal of this project was to further familiarize myself with the AWS ecosystem.




## Setup Instructions


The notebook in this repository is intended to be executed using Amazon's SageMaker platform and the following is a brief set of instructions on setting up a managed notebook instance using SageMaker.


### Log in to the AWS console and create a notebook instance


Log in to the AWS console and go to the SageMaker dashboard. Click on 'Create notebook instance'. The notebook name can be anything and using ml.t2.medium is a good idea as it is covered under the free tier. For the role, creating a new role works fine. Using the default options is also okay. Important to note that you need the notebook instance to have access to S3 resources, which it does by default. In particular, any S3 bucket or object with 'sagemaker' in the name is available to the notebook.




### Use git to clone the repository into the notebook instance


Once the instance has been started and is accessible, click on 'Open Jupyter' to get to the Jupyter notebook main page. To start, clone this repository into the notebook instance.


Click on the 'new' dropdown menu and select 'terminal'. By default, the working directory of the terminal instance is the home directory, however, the Jupyter notebook hub's root directory is under 'SageMaker'. Enter the appropriate directory and clone the repository as follows.


```
cd SageMaker
git clone https://github.com/Supearnesh/ml-census-pca.git
exit
```


After you have finished, close the terminal window.




## Machine Learning Pipeline


This was the general outline followed for this SageMaker project:


1. Loading and exploring the data
        a. Calculating the percentage of fraudulent data
2. Splitting data into training and test sets
3. Binary classification
        a. Define a LinearLearner model
        b. Train the model
        c. Evaluate model performance
4. Making improvements to the model
        a. Tune for higher recall
        b. Manage class imbalance
        c. Tune for higher precision
8. Important: Deleting the endpoint




### Step 1 - Loading and exploring the data


The first step is to load and unzip the data in the `creditcardfraud.zip` file. This directory will hold one csv file containing the labeled transaction data, `creditcard.csv`.


It is important to look at the distribution of data since this will govern the best approach to develop a fraud detection model. It is crucial to learn how many data points are present in the training set, the number and type of features, and also the distribution of data over the classes (valid or fraudulent).


```python
# only have to run once
!wget https://s3.amazonaws.com/video.udacity-data.com/topher/2019/January/5c534768_creditcardfraud/creditcardfraud.zip
!unzip creditcardfraud
```

    --2020-05-30 21:30:23--  https://s3.amazonaws.com/video.udacity-data.com/topher/2019/January/5c534768_creditcardfraud/creditcardfraud.zip
    Resolving s3.amazonaws.com (s3.amazonaws.com)... 52.217.0.214
    Connecting to s3.amazonaws.com (s3.amazonaws.com)|52.217.0.214|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 69155632 (66M) [application/zip]
    Saving to: ‘creditcardfraud.zip’
    
    creditcardfraud.zip 100%[===================>]  65.95M  73.9MB/s    in 0.9s    
    
    2020-05-30 21:30:24 (73.9 MB/s) - ‘creditcardfraud.zip’ saved [69155632/69155632]
    
    Archive:  creditcardfraud.zip
      inflating: creditcard.csv          


```python
# read in the csv file
local_data = 'creditcard.csv'

# print out some data
transaction_df = pd.read_csv(local_data)
print('Data shape (rows, cols): ', transaction_df.shape)
print()
transaction_df.head()
```

    Data shape (rows, cols):  (284807, 31)


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }
    .dataframe tbody tr th {
        vertical-align: top;
    }
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Time</th>
      <th>V1</th>
      <th>V2</th>
      <th>V3</th>
      <th>V4</th>
      <th>V5</th>
      <th>V6</th>
      <th>V7</th>
      <th>V8</th>
      <th>V9</th>
      <th>...</th>
      <th>V21</th>
      <th>V22</th>
      <th>V23</th>
      <th>V24</th>
      <th>V25</th>
      <th>V26</th>
      <th>V27</th>
      <th>V28</th>
      <th>Amount</th>
      <th>Class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>-1.359807</td>
      <td>-0.072781</td>
      <td>2.536347</td>
      <td>1.378155</td>
      <td>-0.338321</td>
      <td>0.462388</td>
      <td>0.239599</td>
      <td>0.098698</td>
      <td>0.363787</td>
      <td>...</td>
      <td>-0.018307</td>
      <td>0.277838</td>
      <td>-0.110474</td>
      <td>0.066928</td>
      <td>0.128539</td>
      <td>-0.189115</td>
      <td>0.133558</td>
      <td>-0.021053</td>
      <td>149.62</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0</td>
      <td>1.191857</td>
      <td>0.266151</td>
      <td>0.166480</td>
      <td>0.448154</td>
      <td>0.060018</td>
      <td>-0.082361</td>
      <td>-0.078803</td>
      <td>0.085102</td>
      <td>-0.255425</td>
      <td>...</td>
      <td>-0.225775</td>
      <td>-0.638672</td>
      <td>0.101288</td>
      <td>-0.339846</td>
      <td>0.167170</td>
      <td>0.125895</td>
      <td>-0.008983</td>
      <td>0.014724</td>
      <td>2.69</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.0</td>
      <td>-1.358354</td>
      <td>-1.340163</td>
      <td>1.773209</td>
      <td>0.379780</td>
      <td>-0.503198</td>
      <td>1.800499</td>
      <td>0.791461</td>
      <td>0.247676</td>
      <td>-1.514654</td>
      <td>...</td>
      <td>0.247998</td>
      <td>0.771679</td>
      <td>0.909412</td>
      <td>-0.689281</td>
      <td>-0.327642</td>
      <td>-0.139097</td>
      <td>-0.055353</td>
      <td>-0.059752</td>
      <td>378.66</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.0</td>
      <td>-0.966272</td>
      <td>-0.185226</td>
      <td>1.792993</td>
      <td>-0.863291</td>
      <td>-0.010309</td>
      <td>1.247203</td>
      <td>0.237609</td>
      <td>0.377436</td>
      <td>-1.387024</td>
      <td>...</td>
      <td>-0.108300</td>
      <td>0.005274</td>
      <td>-0.190321</td>
      <td>-1.175575</td>
      <td>0.647376</td>
      <td>-0.221929</td>
      <td>0.062723</td>
      <td>0.061458</td>
      <td>123.50</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2.0</td>
      <td>-1.158233</td>
      <td>0.877737</td>
      <td>1.548718</td>
      <td>0.403034</td>
      <td>-0.407193</td>
      <td>0.095921</td>
      <td>0.592941</td>
      <td>-0.270533</td>
      <td>0.817739</td>
      <td>...</td>
      <td>-0.009431</td>
      <td>0.798278</td>
      <td>-0.137458</td>
      <td>0.141267</td>
      <td>-0.206010</td>
      <td>0.502292</td>
      <td>0.219422</td>
      <td>0.215153</td>
      <td>69.99</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 31 columns</p>
</div>




#### Part A - Calculating the percentage of fraudulent data


It is important to learn the distribution of this transaction data over the classes, valid and fraudulent, before determining how to approach building a binary classification model to solve this problem. The  `fraudulent_percentage` function below counts the number of data points in each class and calculates the *percentage* of the data points that are fraudulent.


```python
# Calculate the fraction of data points that are fraudulent
def fraudulent_percentage(transaction_df):
    '''Calculate the fraction of all data points that have a 'Class' label of 1; fraudulent.
       :param transaction_df: Dataframe of all transaction data points; has a column 'Class'
       :return: A fractional percentage of fraudulent data points/all points
    '''
    
    # counts for all classes
    counts = transaction_df['Class'].value_counts()
    
    # get fraudulent and valid cnts
    fraud_cnts = counts[1]
    valid_cnts = counts[0]
    
    # calculate percentage of fraudulent data
    fraud_percentage = fraud_cnts/(fraud_cnts+valid_cnts)
    
    return fraud_percentage
```


Below, the code is being tested by calling the function and printing the result.


```python
# call the function to calculate the fraud percentage
fraud_percentage = fraudulent_percentage(transaction_df)

print('Fraudulent percentage = ', fraud_percentage)
print('Total # of fraudulent pts: ', fraud_percentage*transaction_df.shape[0])
print('Out of (total) pts: ', transaction_df.shape[0])
```

    Fraudulent percentage =  0.001727485630620034
    Total # of fraudulent pts:  492.0
    Out of (total) pts:  284807




### Step 2 - Splitting data into training and test sets


In order to evaluate the performance of the fraud classifier that is built using some training data, it will need to be tested on data that it did not see during the training process. So, the initial dataset will need to be split into distinct training and test sets.


The `train_test_split` function below does the following:


* randomly shuffles the transaction data
* splits the data into two sets according to the `train_frac` parameter
* gets train/test features and labels
* returns the tuples: (train_features, train_labels), (test_features, test_labels)


```python
# split into train/test
def train_test_split(transaction_df, train_frac= 0.7, seed=1):
    '''Shuffle the data and randomly split into train and test sets;
       separate the class labels (the column in transaction_df) from the features.
       :param df: Dataframe of all credit card transaction data
       :param train_frac: The decimal fraction of data that should be training data
       :param seed: Random seed for shuffling and reproducibility, default = 1
       :return: Two tuples (in order): (train_features, train_labels), (test_features, test_labels)
       '''
    
    # convert the df into a matrix for ease of splitting
    df_matrix = transaction_df.as_matrix()
    
    # shuffle the data
    np.random.seed(seed)
    np.random.shuffle(df_matrix)
    
    # split the data
    train_size = int(df_matrix.shape[0] * train_frac)
    
    # features are all but last column
    train_features = df_matrix[:train_size, :-1]
    
    # class labels *are* last column
    train_labels = df_matrix[:train_size, -1]
    
    # test data
    test_features = df_matrix[train_size:, :-1]
    test_labels = df_matrix[train_size:, -1]
    
    return (train_features, train_labels), (test_features, test_labels)
    
    # shuffle and split the data
    train_features = None
    train_labels = None
    test_features = None
    test_labels = None
    
    return (train_features, train_labels), (test_features, test_labels)
```


The `train_test_split` function is being tested below by creating training and test sets, as well as ensuring that the data is correctly split according to the `train_frac` ratio and that the class labels are defined within (0, 1).


```python
# get train/test data
(train_features, train_labels), (test_features, test_labels) = train_test_split(transaction_df, train_frac=0.7)
```

    /home/ec2-user/anaconda3/envs/amazonei_mxnet_p36/lib/python3.6/site-packages/ipykernel/__main__.py:12: FutureWarning: Method .as_matrix will be removed in a future version. Use .values instead.


```python
# manual test

# for a split of 0.7:0.3 there should be ~2.33x as many training as test pts
print('Training data pts: ', len(train_features))
print('Test data pts: ', len(test_features))
print()

# take a look at first item and see that it aligns with first row of data
print('First item: \n', train_features[0])
print('Label: ', train_labels[0])
print()

# test split
assert len(train_features) > 2.333*len(test_features), \
        'Unexpected number of train/test points for a train_frac=0.7'
# test labels
assert np.all(train_labels)== 0 or np.all(train_labels)== 1, \
        'Train labels should be 0s or 1s.'
assert np.all(test_labels)== 0 or np.all(test_labels)== 1, \
        'Test labels should be 0s or 1s.'
print('Tests passed!')
```

    Training data pts:  199364
    Test data pts:  85443
    
    First item: 
     [ 1.19907000e+05 -6.11711999e-01 -7.69705324e-01 -1.49759145e-01
     -2.24876503e-01  2.02857736e+00 -2.01988711e+00  2.92491387e-01
     -5.23020325e-01  3.58468461e-01  7.00499612e-02 -8.54022784e-01
      5.47347360e-01  6.16448382e-01 -1.01785018e-01 -6.08491804e-01
     -2.88559430e-01 -6.06199260e-01 -9.00745518e-01 -2.01311157e-01
     -1.96039343e-01 -7.52077614e-02  4.55360454e-02  3.80739375e-01
      2.34403159e-02 -2.22068576e+00 -2.01145578e-01  6.65013699e-02
      2.21179560e-01  1.79000000e+00]
    Label:  0.0
    
    Tests passed!




### Step 3 - Binary classification


Since there are defined labels present in the training set, a **supervised learning** approach is best-suited and a binary classifier can be trained to sort data into one of two transaction classes: fraudulent or valid. Afterwards, the model can be evaluated to see how well it generalizes against some test data. Parameters can be passed into the LinearLearner model to define the model's tendency to favor higher precision or higher recall.


<img src='img/linear_separator.png' width=50% />


A LinearLearner has two main applications:


* for regression tasks, in which a line is fit to some data points, and the goal is to produce a predicted output value given some data point (e.g. predicting housing prices given square footage)
* for binary classification, in which a line is separating two classes of data, and the goal is to effectively label data as either 1 for points above the line or 0 for points on or below the line (e.g. predicting if an email is spam or not)


In this scenario, the LinearLearner model will be trained to determine if the credit card transaction is valid or fraudulent.




#### Part A - Define a LinearLearner model


In order to begin with defining a LinearLearner model, the first step is to create a LinearLearner Estimator. All estimators require some constructor arguments to be passed in. The  [LinearLearner documentation](https://sagemaker.readthedocs.io/en/stable/linear_learner.html) covers how to instantiate a LinearLearner estimator. There are a lot of arguments that can be passed in but they are not all required. It is best to start with a simple model, utilizing default values where applicable. Later, additional hyperparameters can be applied to help solve for specific use cases.


It is recommended to use these instances that are available in the free tier of usage: `'ml.c4.xlarge'` for training and `'ml.t2.medium'` for deployment.


```python
# import LinearLearner
from sagemaker import LinearLearner

# specify an output path
prefix = 'creditcard'
output_path = 's3://{}/{}'.format(bucket, prefix)

# instantiate LinearLearner
linear = LinearLearner(role = role,
                       train_instance_count = 1,
                       train_instance_type = 'ml.c4.xlarge',
                       predictor_type = 'binary_classifier',
                       output_path = output_path,
                       sagemaker_session = sagemaker_session,
                       epochs = 15)
```


Training features and labels for SageMaker built-in models need to be converted into NumPy arrays of float values. Then the [record_set function](https://sagemaker.readthedocs.io/en/stable/linear_learner.html#sagemaker.LinearLearner.record_set) can be used to format the data as a RecordSet and kick off training.


```python
# convert features/labels to numpy
train_x_np = train_features.astype('float32')
train_y_np = train_labels.astype('float32')

# create RecordSet of training data
formatted_train_data = linear.record_set(train_x_np, labels = train_y_np)
```




#### Part B - Train the model


After the estimator has been instantiated, the formatted training data can be passed for training with a call to the `.fit()` function.


```python
%%time 
# train the estimator on formatted training data
linear.fit(formatted_train_data)
```


The trained model can now be deployed to create a predictor. This can be used to make predictions on the test data and evaluate the model.


```python
%%time 
# deploy and create a predictor
linear_predictor = linear.deploy(initial_instance_count = 1,
                                 instance_type = 'ml.t2.medium')
```

    ---------------!CPU times: user 250 ms, sys: 17 ms, total: 267 ms
    Wall time: 7min 32s



#### Part C - Evaluate model performance


Once the model is deployed, it can be tested to see how it performs when applied to test data.


According to the deployed [predictor documentation](https://sagemaker.readthedocs.io/en/stable/linear_learner.html#sagemaker.LinearLearnerPredictor), this predictor expects an `ndarray` of input features and returns a list of Records.


> "The prediction is stored in the "predicted_label" key of the `Record.label` field."


The model can be tested on just one test point to see the resulting list.


```python
# test one prediction
test_x_np = test_features.astype('float32')
result = linear_predictor.predict(test_x_np[0])

print(result)
```

    [label {
      key: "predicted_label"
      value {
        float32_tensor {
          values: 0.0
        }
      }
    }
    label {
      key: "score"
      value {
        float32_tensor {
          values: 0.001805478474125266
        }
      }
    }
    ]


The `predicted_label` determines that the transaction is valid, with a `score` value that is very close to `0`, or the valid class.


```python
# code to evaluate the endpoint on test data
# returns a variety of model metrics
def evaluate(predictor, test_features, test_labels, verbose=True):
    """
    Evaluate a model on a test set given the prediction endpoint.  
    Return binary classification metrics.
    :param predictor: A prediction endpoint
    :param test_features: Test features
    :param test_labels: Class labels for test data
    :param verbose: If True, prints a table of all performance metrics
    :return: A dictionary of performance metrics.
    """
    
    # We have a lot of test data, so we'll split it into batches of 100
    # split the test data set into batches and evaluate using prediction endpoint    
    prediction_batches = [predictor.predict(batch) for batch in np.array_split(test_features, 100)]
    
    # LinearLearner produces a `predicted_label` for each data point in a batch
    # get the 'predicted_label' for every point in a batch
    test_preds = np.concatenate([np.array([x.label['predicted_label'].float32_tensor.values[0] for x in batch]) 
                                 for batch in prediction_batches])
    
    # calculate true positives, false positives, true negatives, false negatives
    tp = np.logical_and(test_labels, test_preds).sum()
    fp = np.logical_and(1-test_labels, test_preds).sum()
    tn = np.logical_and(1-test_labels, 1-test_preds).sum()
    fn = np.logical_and(test_labels, 1-test_preds).sum()
    
    # calculate binary classification metrics
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    accuracy = (tp + tn) / (tp + fp + tn + fn)
    
    # printing a table of metrics
    if verbose:
        print(pd.crosstab(test_labels, test_preds, rownames=['actual (row)'], colnames=['prediction (col)']))
        print("\n{:<11} {:.3f}".format('Recall:', recall))
        print("{:<11} {:.3f}".format('Precision:', precision))
        print("{:<11} {:.3f}".format('Accuracy:', accuracy))
        print()
        
    return {'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn, 
            'Precision': precision, 'Recall': recall, 'Accuracy': accuracy}
```


Next, the `evaluate` function defined above will be used to take in a deployed predictor, some test features and labels, and return a dictionary of metrics to calculate TP/FP/TN/FN as well as recall, precision, and accuracy.


```python
print('Metrics for simple, LinearLearner.\n')

# get metrics for linear predictor
metrics = evaluate(linear_predictor, 
                   test_features.astype('float32'), 
                   test_labels, 
                   verbose=True) # verbose means we'll print out the metrics
```

    Metrics for simple, LinearLearner.
    
    prediction (col)    0.0  1.0
    actual (row)                
    0.0               85269   33
    1.0                  32  109
    
    Recall:     0.773
    Precision:  0.768
    Accuracy:   0.999




### Step 4 - Making improvements to the model


The default LinearLearner model had a high accuracy, but still classified fraudulent and valid data points incorrectly. Specifically classifying more than 30 points as false negatives (incorrectly labeled, fraudulent transactions), and a little over 30 points as false positives (incorrectly labeled, valid transactions). Optimizing according to a specific metric is called **model tuning**, and SageMaker provides a number of ways to automatically tune a model.


**1. Model optimization**
* If this model was being designed for use in a bank application, it may be a project requirement that users do *not* want any valid transactions to be categorized as fraudulent. That is, the model should produce as few **false positives** (0s classified as 1s) as possible.
* On the other hand, if the bank manager asks for an application that will catch almost *all* cases of fraud, even if it means a higher number of false positives, then the model should produce as few **false negatives** as possible.
* To train according to specific product demands and goals, it is not advisable to only optimize for accuracy. Instead, the goal should be to optimize for a metric that can help decrease the number of false positives or negatives.


<img src='img/precision_recall.png' width=40% />


**2. Imbalanced training data**
* At the start of this project, it became evident that only 0.17% of the training data was labeled as fraudulent. So, even if a model labeled **all** of the data as valid, it would still have a high accuracy.
* This may result in overfitting towards valid data, which accounts for some **false negatives**; cases in which fraudulent data (1) is incorrectly characterized as valid (0).


It would make sense to address these issues in order; first, tuning the model and optimizing for a specific metric during training, and second, accounting for class imbalance in the training set.




#### Part A - Tune for higher recall


**Scenario:**
* A bank has asked for a model that detects cases of fraud with an accuracy of about 85%.


In this case, the model should produce as many true positives and as few false negatives, as possible. This corresponds to a model with a high **recall**: true positives / (true positives + false negatives).


To aim for a specific metric, LinearLearner offers the hyperparameter `binary_classifier_model_selection_criteria`, which is the model evaluation criteria for the training dataset. A reference to this parameter is in [LinearLearner's documentation](https://sagemaker.readthedocs.io/en/stable/linear_learner.html#sagemaker.LinearLearner). It will also be necessary to further specify the exact value to aim for based on these [hyperparameters](https://docs.aws.amazon.com/sagemaker/latest/dg/ll_hyperparameters.html).


Assuming that performance on a training set will be within about 5% of the performance on a test set, in order to produce a model with recall of about 85% it would make sense to aim a bit higher at 90%.


```python
# instantiate and train a LinearLearner

# tune the model for higher recall
linear_recall = LinearLearner(role = role,
                              train_instance_count = 1, 
                              train_instance_type = 'ml.c4.xlarge',
                              predictor_type = 'binary_classifier',
                              output_path = output_path,
                              sagemaker_session = sagemaker_session,
                              epochs = 15,
                              binary_classifier_model_selection_criteria = 'precision_at_target_recall', # target recall
                              target_recall = 0.9) # 90% recall

# train the estimator on formatted training data
linear_recall.fit(formatted_train_data)
```


Next, the tuned predictor can be deployed and evaluated. It was hypothesized earlier that a tuned model, optimized for a higher recall, would produce fewer false negatives (fraudulent transactions incorrectly labeled as valid). It is important to validate if the number of false negatives decreased after tuning the model.


```python
%%time 
# deploy and create a predictor
recall_predictor = linear_recall.deploy(initial_instance_count=1,
                                        instance_type='ml.t2.medium')
```

    -----------------!CPU times: user 283 ms, sys: 9.08 ms, total: 292 ms
    Wall time: 8min 32s


```python
print('Metrics for tuned (recall), LinearLearner.\n')

# get metrics for tuned predictor
metrics = evaluate(recall_predictor, 
                   test_features.astype('float32'), 
                   test_labels, 
                   verbose=True)
```

    Metrics for tuned (recall), LinearLearner.
    
    prediction (col)    0.0   1.0
    actual (row)                 
    0.0               81913  3389
    1.0                  10   131
    
    Recall:     0.929
    Precision:  0.037
    Accuracy:   0.960




#### Part B - Manage class imbalance


The current model is tuned to produce higher recall, which aims to reduce the number of false negatives. Earlier, it was discussed how class imbalance may create bias in the model towards predicting that all transactions are valid, resulting in higher false negatives and true negatives. It stands to reason that this model could be further improved if this imbalance was taken into consideration.


To account for class imbalance during training of a binary classifier, LinearLearner offers the hyperparameter, `positive_example_weight_mult`, which is the weight assigned to positive (1, fraudulent) examples when training a binary classifier. The weight of negative examples (0, valid) is fixed at 1. A reference to this parameter is in [LinearLearner's documentation](https://sagemaker.readthedocs.io/en/stable/linear_learner.html#sagemaker.LinearLearner). In **addition** to tuning a model for higher recall ( `linear_recall` may be used as a starting point), a parameter that helps account for class imbalance should be *added*. From the [hyperparameter documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/ll_hyperparameters.html) on `positive_example_weight_mult`, it reads:


> "If the algorithm should choose a weight so that errors in classifying negative vs. positive examples have equal impact on training loss, specify `balanced`."


A specific float value may also be used, in which case positive examples should be weighed more heavily than negative examples, since there are fewer of them.


```python
# instantiate a LinearLearner

# include params for tuning for higher recall
# *and* account for class imbalance in training data
linear_balanced = LinearLearner(role = role,
                                train_instance_count = 1,
                                train_instance_type = 'ml.c4.xlarge',
                                predictor_type='binary_classifier',
                                output_path = output_path,
                                sagemaker_session = sagemaker_session,
                                epochs = 15,
                                binary_classifier_model_selection_criteria = 'precision_at_target_recall', # target recall
                                target_recall = 0.9, # 90% recall
                                positive_example_weight_mult = 'balanced') # added to account for class imbalance in training set

# train the estimator on formatted training data
linear_balanced.fit(formatted_train_data)
```


Next, the tuned predictor can be deployed and evaluated. It was hypothesized earlier that a tuned model, optimized for a higher recall, with a parameter added to account for an imbalanced training set, would produce fewer false positives (valid transactions incorrectly labeled as fraudulent). It is important to validate if the number of false positives decreased after tuning the model.


```python
%%time 
# deploy and create a predictor
balanced_predictor = linear_balanced.deploy(initial_instance_count = 1,
                                            instance_type='ml.t2.medium')
```

    -----------------!CPU times: user 280 ms, sys: 12.3 ms, total: 292 ms
    Wall time: 8min 32s


```python
print('Metrics for balanced, LinearLearner.\n')

# get metrics for balanced predictor
metrics = evaluate(balanced_predictor, 
                   test_features.astype('float32'), 
                   test_labels, 
                   verbose=True)
```

    Metrics for balanced, LinearLearner.
    
    prediction (col)    0.0   1.0
    actual (row)                 
    0.0               84277  1025
    1.0                  12   129
    
    Recall:     0.915
    Precision:  0.112
    Accuracy:   0.988




#### Part C - Tune for higher precision


**Scenario:**
* A bank has asked for a model that optimizes for a good user experience; users should only ever have up to about 15% of their valid transactions flagged as fraudulent.


In this case, the model should produce as many true positives and as few false positives, as possible. This corresponds to a model with a high **precision**: true positives / (true positives + false positives).


To aim for a specific metric, LinearLearner offers the hyperparameter `binary_classifier_model_selection_criteria`, which is the model evaluation criteria for the training dataset. A reference to this parameter is in [LinearLearner's documentation](https://sagemaker.readthedocs.io/en/stable/linear_learner.html#sagemaker.LinearLearner). It will also be necessary to further specify the exact value to aim for based on these [hyperparameters](https://docs.aws.amazon.com/sagemaker/latest/dg/ll_hyperparameters.html).


Assuming that performance on a training set will be within about 5% of the performance on a test set, in order to produce a model with precision of about 85% it would make sense to aim a bit higher at 90%.


```python
%%time
# instantiate and train a LinearLearner

# include params for tuning for higher precision
# *and* account for class imbalance in training data
linear_precision = LinearLearner(role = role,
                                train_instance_count = 1,
                                train_instance_type = 'ml.c4.xlarge',
                                predictor_type='binary_classifier',
                                output_path = output_path,
                                sagemaker_session = sagemaker_session,
                                epochs = 15,
                                binary_classifier_model_selection_criteria = 'recall_at_target_precision', # target precision
                                target_precision = 0.9, # 90% precision
                                positive_example_weight_mult = 'balanced') # added to account for class imbalance in training set

# train the estimator on formatted training data
linear_precision.fit(formatted_train_data)
```


Next, the tuned predictor can be deployed and evaluated. It was hypothesized earlier that a tuned model, optimized for a higher precision, would produce fewer false positives (valid transactions incorrectly labeled as fraudulent). It is important to validate if the number of false positives decreased after tuning the model.


```python
%%time 
# deploy and evaluate a predictor
precision_predictor = linear_precision.deploy(initial_instance_count=1,
                                              instance_type='ml.t2.medium')
```

    -----------------!CPU times: user 283 ms, sys: 8.36 ms, total: 291 ms
    Wall time: 8min 32s


```python
print('Metrics for tuned (precision), LinearLearner.\n')

# get metrics for balanced predictor
metrics = evaluate(precision_predictor, 
                   test_features.astype('float32'), 
                   test_labels, 
                   verbose=True)
```

    Metrics for tuned (precision), LinearLearner.

    prediction (col)    0.0  1.0
    actual (row)                
    0.0               85276   26
    1.0                  31  110

    Recall:     0.780
    Precision:  0.809
    Accuracy:   0.999




### Important - Deleting the endpoint


Always remember to shut down the model endpoint if it is no longer being used. AWS charges for the duration that an endpoint is left running, so if it is left on then there could be an unexpectedly large AWS bill.


```python
predictor.delete_endpoint()
```
