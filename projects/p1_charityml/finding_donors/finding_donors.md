
# Data Scientist Nanodegree
## Supervised Learning
## Project: Finding Donors for *CharityML*

## Getting Started

In this project, you will employ several supervised algorithms of your choice to accurately model individuals' income using data collected from the 1994 U.S. Census. You will then choose the best candidate algorithm from preliminary results and further optimize this algorithm to best model the data. Your goal with this implementation is to construct a model that accurately predicts whether an individual makes more than $50,000. This sort of task can arise in a non-profit setting, where organizations survive on donations.  Understanding an individual's income can help a non-profit better understand how large of a donation to request, or whether or not they should reach out to begin with.  While it can be difficult to determine an individual's general income bracket directly from public sources, we can (as we will see) infer this value from other publically available features. 

The dataset for this project originates from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Census+Income). The datset was donated by Ron Kohavi and Barry Becker, after being published in the article _"Scaling Up the Accuracy of Naive-Bayes Classifiers: A Decision-Tree Hybrid"_. You can find the article by Ron Kohavi [online](https://www.aaai.org/Papers/KDD/1996/KDD96-033.pdf). The data we investigate here consists of small changes to the original dataset, such as removing the `'fnlwgt'` feature and records with missing or ill-formatted entries.

----
## Exploring the Data


```python
# Import libraries necessary for this project
import numpy as np
import pandas as pd
from time import time
from IPython.display import display # Allows the use of display() for DataFrames

# Import supplementary visualization code visuals.py
import visuals as vs

# Pretty display for notebooks
%matplotlib inline

# Load the Census dataset
data = pd.read_csv("census.csv")



# Success - Display the first record
display(data.head(n=1))

```


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
      <th>age</th>
      <th>workclass</th>
      <th>education_level</th>
      <th>education-num</th>
      <th>marital-status</th>
      <th>occupation</th>
      <th>relationship</th>
      <th>race</th>
      <th>sex</th>
      <th>capital-gain</th>
      <th>capital-loss</th>
      <th>hours-per-week</th>
      <th>native-country</th>
      <th>income</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>39</td>
      <td>State-gov</td>
      <td>Bachelors</td>
      <td>13.0</td>
      <td>Never-married</td>
      <td>Adm-clerical</td>
      <td>Not-in-family</td>
      <td>White</td>
      <td>Male</td>
      <td>2174.0</td>
      <td>0.0</td>
      <td>40.0</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
  </tbody>
</table>
</div>


### Implementation: Data Exploration


```python
# TODO: Total number of records
n_records = data.shape[0]


# TODO: Number of records where individual's income is more than $50,000
n_greater_50k = data[data.income == '>50K'].shape[0]

# TODO: Number of records where individual's income is at most $50,000
n_at_most_50k = data[data.income == '<=50K'].shape[0]

# TODO: Percentage of individuals whose income is more than $50,000
greater_percent = round(n_greater_50k/n_records*100, 3)

# Print the results
print("Total number of records: {}".format(n_records))
print("Individuals making more than $50,000: {}".format(n_greater_50k))
print("Individuals making at most $50,000: {}".format(n_at_most_50k))
print("Percentage of individuals making more than $50,000: {}%".format(greater_percent))
```

    Total number of records: 45222
    Individuals making more than $50,000: 11208
    Individuals making at most $50,000: 34014
    Percentage of individuals making more than $50,000: 24.784%


** Featureset Exploration **

* **age**: continuous. 
* **workclass**: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked. 
* **education**: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool. 
* **education-num**: continuous. 
* **marital-status**: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse. 
* **occupation**: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces. 
* **relationship**: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried. 
* **race**: Black, White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other. 
* **sex**: Female, Male. 
* **capital-gain**: continuous. 
* **capital-loss**: continuous. 
* **hours-per-week**: continuous. 
* **native-country**: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.

----
## Preparing the Data
Before data can be used as input for machine learning algorithms, it often must be cleaned, formatted, and restructured — this is typically known as **preprocessing**. Fortunately, for this dataset, there are no invalid or missing entries we must deal with, however, there are some qualities about certain features that must be adjusted. This preprocessing can help tremendously with the outcome and predictive power of nearly all learning algorithms.

### Transforming Skewed Continuous Features
A dataset may sometimes contain at least one feature whose values tend to lie near a single number, but will also have a non-trivial number of vastly larger or smaller values than that single number.  Algorithms can be sensitive to such distributions of values and can underperform if the range is not properly normalized. With the census dataset two features fit this description: '`capital-gain'` and `'capital-loss'`. 


```python
# Split the data into features and target label
income_raw = data['income']
features_raw = data.drop('income', axis = 1)

# Visualize skewed continuous features of original data
vs.distribution(data)
```


![png](output_9_0.png)


For highly-skewed feature distributions such as `'capital-gain'` and `'capital-loss'`, it is common practice to apply a <a href="https://en.wikipedia.org/wiki/Data_transformation_(statistics)">logarithmic transformation</a> on the data so that the very large and very small values do not negatively affect the performance of a learning algorithm. Using a logarithmic transformation significantly reduces the range of values caused by outliers. Care must be taken when applying this transformation however: The logarithm of `0` is undefined, so we must translate the values by a small amount above `0` to apply the the logarithm successfully.


```python
# Log-transform the skewed features
skewed = ['capital-gain', 'capital-loss']
features_log_transformed = pd.DataFrame(data = features_raw)
features_log_transformed[skewed] = features_raw[skewed].apply(lambda x: np.log(x + 1))

# Visualize the new log distributions
vs.distribution(features_log_transformed, transformed = True)
```


![png](output_11_0.png)


### Normalizing Numerical Features
In addition to performing transformations on features that are highly skewed, it is often good practice to perform some type of scaling on numerical features. Applying a scaling to the data does not change the shape of each feature's distribution (such as `'capital-gain'` or `'capital-loss'` above); however, normalization ensures that each feature is treated equally when applying supervised learners. Note that once scaling is applied, observing the data in its raw form will no longer have the same original meaning, as exampled below.


```python
# Import sklearn.preprocessing.StandardScaler
from sklearn.preprocessing import MinMaxScaler

# Initialize a scaler, then apply it to the features
scaler = MinMaxScaler() # default=(0, 1)
numerical = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']

features_log_minmax_transform = pd.DataFrame(data = features_log_transformed)
features_log_minmax_transform[numerical] = scaler.fit_transform(features_log_transformed[numerical])

# Show an example of a record with scaling applied
display(features_log_minmax_transform.head(n = 5))
```


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
      <th>age</th>
      <th>workclass</th>
      <th>education_level</th>
      <th>education-num</th>
      <th>marital-status</th>
      <th>occupation</th>
      <th>relationship</th>
      <th>race</th>
      <th>sex</th>
      <th>capital-gain</th>
      <th>capital-loss</th>
      <th>hours-per-week</th>
      <th>native-country</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.301370</td>
      <td>State-gov</td>
      <td>Bachelors</td>
      <td>0.800000</td>
      <td>Never-married</td>
      <td>Adm-clerical</td>
      <td>Not-in-family</td>
      <td>White</td>
      <td>Male</td>
      <td>0.667492</td>
      <td>0.0</td>
      <td>0.397959</td>
      <td>United-States</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.452055</td>
      <td>Self-emp-not-inc</td>
      <td>Bachelors</td>
      <td>0.800000</td>
      <td>Married-civ-spouse</td>
      <td>Exec-managerial</td>
      <td>Husband</td>
      <td>White</td>
      <td>Male</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.122449</td>
      <td>United-States</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.287671</td>
      <td>Private</td>
      <td>HS-grad</td>
      <td>0.533333</td>
      <td>Divorced</td>
      <td>Handlers-cleaners</td>
      <td>Not-in-family</td>
      <td>White</td>
      <td>Male</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.397959</td>
      <td>United-States</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.493151</td>
      <td>Private</td>
      <td>11th</td>
      <td>0.400000</td>
      <td>Married-civ-spouse</td>
      <td>Handlers-cleaners</td>
      <td>Husband</td>
      <td>Black</td>
      <td>Male</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.397959</td>
      <td>United-States</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.150685</td>
      <td>Private</td>
      <td>Bachelors</td>
      <td>0.800000</td>
      <td>Married-civ-spouse</td>
      <td>Prof-specialty</td>
      <td>Wife</td>
      <td>Black</td>
      <td>Female</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.397959</td>
      <td>Cuba</td>
    </tr>
  </tbody>
</table>
</div>


### Implementation: Data Preprocessing

From the table in **Exploring the Data** above, we can see there are several features for each record that are non-numeric. Typically, learning algorithms expect input to be numeric, which requires that non-numeric features (called *categorical variables*) be converted. One popular way to convert categorical variables is by using the **one-hot encoding** scheme. One-hot encoding creates a _"dummy"_ variable for each possible category of each non-numeric feature. For example, assume `someFeature` has three possible entries: `A`, `B`, or `C`. We then encode this feature into `someFeature_A`, `someFeature_B` and `someFeature_C`.

|   | someFeature |                    | someFeature_A | someFeature_B | someFeature_C |
| :-: | :-: |                            | :-: | :-: | :-: |
| 0 |  B  |  | 0 | 1 | 0 |
| 1 |  C  | ----> one-hot encode ----> | 0 | 0 | 1 |
| 2 |  A  |  | 1 | 0 | 0 |

Additionally, as with the non-numeric features, we need to convert the non-numeric target label, `'income'` to numerical values for the learning algorithm to work. Since there are only two possible categories for this label ("<=50K" and ">50K"), we can avoid using one-hot encoding and simply encode these two categories as `0` and `1`, respectively.


```python
# TODO: One-hot encode the 'features_log_minmax_transform' data using pandas.get_dummies()
features_final = pd.get_dummies(features_log_minmax_transform)

# TODO: Encode the 'income_raw' data to numerical values
income = income_raw.map({'<=50K':0, '>50K':1})

# Print the number of features after one-hot encoding
encoded = list(features_final.columns)
print("{} total features after one-hot encoding.".format(len(encoded)))

# Uncomment the following line to see the encoded feature names
# print(encoded)
```

    103 total features after one-hot encoding.


### Shuffle and Split Data
Now all _categorical variables_ have been converted into numerical features, and all numerical features have been normalized. As always, we will now split the data (both features and their labels) into training and test sets. 80% of the data will be used for training and 20% for testing.


```python
# Import train_test_split
from sklearn.model_selection import train_test_split

# Split the 'features' and 'income' data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features_final, 
                                                    income, 
                                                    test_size = 0.2, 
                                                    random_state = 0)

# Show the results of the split
print("Training set has {} samples.".format(X_train.shape[0]))
print("Testing set has {} samples.".format(X_test.shape[0]))
```

    Training set has 36177 samples.
    Testing set has 9045 samples.


----
## Evaluating Model Performance
In this section, we will investigate four different algorithms, and determine which is best at modeling the data. Three of these algorithms will be supervised learners of your choice, and the fourth algorithm is known as a *naive predictor*.

### Metrics and the Naive Predictor
*CharityML*, equipped with their research, knows individuals that make more than \$50,000 are most likely to donate to their charity. Because of this, *CharityML* is particularly interested in predicting who makes more than \$50,000 accurately. It would seem that using **accuracy** as a metric for evaluating a particular model's performace would be appropriate. Additionally, identifying someone that *does not* make more than \$50,000 as someone who does would be detrimental to *CharityML*, since they are looking to find individuals willing to donate. Therefore, a model's ability to precisely predict those that make more than \$50,000 is *more important* than the model's ability to **recall** those individuals. We can use **F-beta score** as a metric that considers both precision and recall:

$$ F_{\beta} = (1 + \beta^2) \cdot \frac{precision \cdot recall}{\left( \beta^2 \cdot precision \right) + recall} $$

In particular, when $\beta = 0.5$, more emphasis is placed on precision. This is called the **F$_{0.5}$ score** (or F-score for simplicity).

Looking at the distribution of classes (those who make at most \$50,000, and those who make more), it's clear most individuals do not make more than \$50,000. This can greatly affect **accuracy**, since we could simply say *"this person does not make more than \$50,000"* and generally be right, without ever looking at the data! Making such a statement would be called **naive**, since we have not considered any information to substantiate the claim. It is always important to consider the *naive prediction* for your data, to help establish a benchmark for whether a model is performing well. That been said, using that prediction would be pointless: If we predicted all people made less than \$50,000, *CharityML* would identify no one as donors. 


#### Note: Recap of accuracy, precision, recall

** Accuracy ** measures how often the classifier makes the correct prediction. It’s the ratio of the number of correct predictions to the total number of predictions (the number of test data points).

** Precision ** tells us what proportion of messages we classified as spam, actually were spam.
It is a ratio of true positives(words classified as spam, and which are actually spam) to all positives(all words classified as spam, irrespective of whether that was the correct classificatio), in other words it is the ratio of

`[True Positives/(True Positives + False Positives)]`

** Recall(sensitivity)** tells us what proportion of messages that actually were spam were classified by us as spam.
It is a ratio of true positives(words classified as spam, and which are actually spam) to all the words that were actually spam, in other words it is the ratio of

`[True Positives/(True Positives + False Negatives)]`

For classification problems that are skewed in their classification distributions like in our case, for example if we had a 100 text messages and only 2 were spam and the rest 98 weren't, accuracy by itself is not a very good metric. We could classify 90 messages as not spam(including the 2 that were spam but we classify them as not spam, hence they would be false negatives) and 10 as spam(all 10 false positives) and still get a reasonably good accuracy score. For such cases, precision and recall come in very handy. These two metrics can be combined to get the F1 score, which is weighted average(harmonic mean) of the precision and recall scores. This score can range from 0 to 1, with 1 being the best possible F1 score(we take the harmonic mean as we are dealing with ratios).

### Question 1 - Naive Predictor Performance
* If we chose a model that always predicted an individual made more than $50,000, what would  that model's accuracy and F-score be on this dataset? You must use the code cell below and assign your results to `'accuracy'` and `'fscore'` to be used later.

** Please note ** that the the purpose of generating a naive predictor is simply to show what a base model without any intelligence would look like. In the real world, ideally your base model would be either the results of a previous model or could be based on a research paper upon which you are looking to improve. When there is no benchmark model set, getting a result better than random choice is a place you could start from.


```python
'''
TP = np.sum(income) # Counting the ones as this is the naive case. Note that 'income' is the 'income_raw' data 
encoded to numerical values done in the data preprocessing step.
FP = income.count() - TP # Specific to the naive case

TN = 0 # No predicted negatives in the naive case
FN = 0 # No predicted negatives in the naive case
'''
# TODO: Calculate accuracy, precision and recall
accuracy = np.sum(income)/income.count()
recall = np.sum(income)/np.sum(income)
precision = np.sum(income)/income.count()

# TODO: Calculate F-score using the formula above for beta = 0.5 and correct values for precision and recall.
beta = 0.5
fscore =(1+beta**2)*precision*recall/(((beta**2)*precision) + recall)

# Print the results 
print("Naive Predictor: [Accuracy score: {:.4f}, F-score: {:.4f}]".format(accuracy, fscore))
```

    Naive Predictor: [Accuracy score: 0.2478, F-score: 0.2917]


###  Supervised Learning Models
**The following are some of the supervised learning models that are currently available in** [`scikit-learn`](http://scikit-learn.org/stable/supervised_learning.html) **that you may choose from:**
- Gaussian Naive Bayes (GaussianNB)
- Decision Trees
- Ensemble Methods (Bagging, AdaBoost, Random Forest, Gradient Boosting)
- K-Nearest Neighbors (KNeighbors)
- Stochastic Gradient Descent Classifier (SGDC)
- Support Vector Machines (SVM)
- Logistic Regression

### Question 2 - Model Application
List three of the supervised learning models above that are appropriate for this problem that you will test on the census data. For each model chosen

- Describe one real-world application in industry where the model can be applied. 
- What are the strengths of the model; when does it perform well?
- What are the weaknesses of the model; when does it perform poorly?
- What makes this model a good candidate for the problem, given what you know about the data?

** HINT: **

Structure your answer in the same format as above^, with 4 parts for each of the three models you pick. Please include references with your answer.

**Answer: **

1) Decision Tree
- a system recommending apps to its users based on user data can use a Decision Tree model. (the example given in the course)
- handles both continuous and categorical features well, is a really fast algorithm, is a white box model, easily interpretable (tree can be visualized)
- Can overfit the data (mitigated by varying hyperparameter space), can be unstable (small changes in training set can produce completely different tree)
- we have both continuous and categorical features, we want the algorithm to be fast since we are going to repeat it over the hyperparameter space, which makes this model a good candidate.

source: https://scikit-learn.org/stable/modules/tree.html

2) AdaBoost (Ensemble Method)
- For example, AdaBoost can be used for spam email classification.
- Boosting algorithms can greatly reduce the bias of any learning algorithm that consistently produces predictions which are better than random guesses, while keeping a low variance, or even improving on the variance of the weak learners.
- Boosting algorithms work best when using simple algorithms as weak learners; when using more complex algorithms as weak learners, bagging/averaging algorithms are usually superior. Boosting algorithms can also have a tendancy to overfit, and the computation time can be quite long with more complex weak learners since it is a sequential method.
- Using AdaBoost with Decision Stumps is appropriate for this dataset as it handles both continuous and categorical features well, and is well-adapted for binary classification. 

sources: https://people.cs.pitt.edu/~milos/courses/cs2750/Readings/boosting.pdf
https://scikit-learn.org/stable/modules/ensemble.html#fs1995

3) SVC
- Hand-writing recognition, for example a digit classifier, is an example that can use a SVM algorithm.
- SVMs work well in datasets with large numbers of features. They are also highly customizable, as they can use several different Kernel functions.
- SVMS also don't have an efficient way of providing probability estimates. Overfitting can also be a problem with datasets with more features than samples.
- As this dataset has a large number of features, it seems appropriate to use a SVM. The high versatility means this model can easily fit most datasets, given appropriate hyperparameters.

sources: http://www.pybloggers.com/2016/02/using-support-vector-machines-for-digit-recognition/
https://scikit-learn.org/stable/modules/svm.html#svm-classification

### Implementation - Creating a Training and Predicting Pipeline
To properly evaluate the performance of each model you've chosen, it's important that you create a training and predicting pipeline that allows you to quickly and effectively train models using various sizes of training data and perform predictions on the testing data. Your implementation here will be used in the following section.
In the code block below, you will need to implement the following:
 - Import `fbeta_score` and `accuracy_score` from [`sklearn.metrics`](http://scikit-learn.org/stable/modules/classes.html#sklearn-metrics-metrics).
 - Fit the learner to the sampled training data and record the training time.
 - Perform predictions on the test data `X_test`, and also on the first 300 training points `X_train[:300]`.
   - Record the total prediction time.
 - Calculate the accuracy score for both the training subset and testing set.
 - Calculate the F-score for both the training subset and testing set.
   - Make sure that you set the `beta` parameter!


```python
# TODO: Import two metrics from sklearn - fbeta_score and accuracy_score
from sklearn.metrics import accuracy_score, fbeta_score

def train_predict(learner, sample_size, X_train, y_train, X_test, y_test): 
    '''
    inputs:
       - learner: the learning algorithm to be trained and predicted on
       - sample_size: the size of samples (number) to be drawn from training set
       - X_train: features training set
       - y_train: income training set
       - X_test: features testing set
       - y_test: income testing set
    '''
    
    results = {}
    
    # TODO: Fit the learner to the training data using slicing with 'sample_size' using .fit(training_features[:], training_labels[:])
    start = time() # Get start time
    learner.fit(X_train[:sample_size], y_train[:sample_size])
    end = time() # Get end time
    
    # TODO: Calculate the training time
    results['train_time'] = end-start
        
    # TODO: Get the predictions on the test set(X_test),
    #       then get predictions on the first 300 training samples(X_train) using .predict()
    start = time() # Get start time
    predictions_test = learner.predict(X_test)
    predictions_train = learner.predict(X_train[:300])
    end = time() # Get end time
    
    # TODO: Calculate the total prediction time
    results['pred_time'] = end-start
            
    # TODO: Compute accuracy on the first 300 training samples which is y_train[:300]
    results['acc_train'] = accuracy_score(y_train[:300], predictions_train)
        
    # TODO: Compute accuracy on test set using accuracy_score()
    results['acc_test'] = accuracy_score(y_test, predictions_test)
    
    # TODO: Compute F-score on the the first 300 training samples using fbeta_score()
    results['f_train'] = fbeta_score(y_train[:300], predictions_train, beta=0.5)
        
    # TODO: Compute F-score on the test set which is y_test
    results['f_test'] = fbeta_score(y_test, predictions_test, beta=0.5)
       
    # Success
    print("{} trained on {} samples.".format(learner.__class__.__name__, sample_size))
        
    # Return the results
    return results
```

### Implementation: Initial Model Evaluation
In the code cell, you will need to implement the following:
- Import the three supervised learning models you've discussed in the previous section.
- Initialize the three models and store them in `'clf_A'`, `'clf_B'`, and `'clf_C'`.
  - Use a `'random_state'` for each model you use, if provided.
  - **Note:** Use the default settings for each model — you will tune one specific model in a later section.
- Calculate the number of records equal to 1%, 10%, and 100% of the training data.
  - Store those values in `'samples_1'`, `'samples_10'`, and `'samples_100'` respectively.

**Note:** Depending on which algorithms you chose, the following implementation may take some time to run!


```python
# TODO: Import the three supervised learning models from sklearn
from sklearn.svm import SVC #way too long, good f1
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier #overfitting first two (randomforest fast, bagging long), good last two, GBC best but long, AdaBoost shortest
from sklearn.naive_bayes import GaussianNB, MultinomialNB #too low f1 score
from sklearn.tree import DecisionTreeClassifier #overfitting, fast
from sklearn.linear_model import LogisticRegression, SGDClassifier #not bad, sgdc inconsistent f1 score
from sklearn.neighbors import KNeighborsClassifier #too long, f1 score average

# TODO: Initialize the three models
clf_A = DecisionTreeClassifier(random_state=1)
clf_B = AdaBoostClassifier(random_state=1)
clf_C = SVC(random_state=1)

# TODO: Calculate the number of samples for 1%, 10%, and 100% of the training data
# HINT: samples_100 is the entire training set i.e. len(y_train)
# HINT: samples_10 is 10% of samples_100 (ensure to set the count of the values to be `int` and not `float`)
# HINT: samples_1 is 1% of samples_100 (ensure to set the count of the values to be `int` and not `float`)
samples_100 = len(y_train)
samples_10 = int(len(y_train)/10)
samples_1 = int(len(y_train)/100)

# Collect results on the learners
results = {}
for clf in [clf_A, clf_B, clf_C]:
    clf_name = clf.__class__.__name__
    results[clf_name] = {}
    for i, samples in enumerate([samples_1, samples_10, samples_100]):
        results[clf_name][i] = \
        train_predict(clf, samples, X_train, y_train, X_test, y_test)

# Run metrics visualization for the three supervised learning models chosen
vs.evaluate(results, accuracy, fscore)
```

    DecisionTreeClassifier trained on 361 samples.
    DecisionTreeClassifier trained on 3617 samples.


    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:29: DeprecationWarning: numpy.core.umath_tests is an internal NumPy module and should not be imported. It will be removed in a future NumPy release.
      from numpy.core.umath_tests import inner1d


    DecisionTreeClassifier trained on 36177 samples.
    AdaBoostClassifier trained on 361 samples.
    AdaBoostClassifier trained on 3617 samples.
    AdaBoostClassifier trained on 36177 samples.


    /anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1135: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 due to no predicted samples.
      'precision', 'predicted', average, warn_for)


    SVC trained on 361 samples.
    SVC trained on 3617 samples.
    SVC trained on 36177 samples.



![png](output_28_5.png)


----
## Improving Results
In this final section, you will choose from the three supervised learning models the *best* model to use on the student data. You will then perform a grid search optimization for the model over the entire training set (`X_train` and `y_train`) by tuning at least one parameter to improve upon the untuned model's F-score. 

### Question 3 - Choosing the Best Model

* Based on the evaluation you performed earlier, in one to two paragraphs, explain to *CharityML* which of the three models you believe to be most appropriate for the task of identifying individuals that make more than \$50,000. 

**Answer: **

The first model, the simple Decision Tree, seems to be overfitting the data, as it gets near perfect accuracy and F1 scores on the training sets, but scores much lower on the testing tests. 

The third model (SVC), is a bit better and doesn't overfit as much, and is highly versatile, so it could potentially be improved further, however the training computation time (100 seconds) makes it prohibitively time-consuming.

The second model (AdaBoost with Decision Stump) scores the highest on the testing set (F1 score), doesn't overfit as much, and is also highly versatile, so we can hope to improve the score even further with some adjustments. The only problem is the training computation time which is still relatively high (around 1 second), but still bearable.

I therefore believe AdaBoost to be the best of the three models for this dataset.

### Question 4 - Describing the Model in Layman's Terms

* In one to two paragraphs, explain to *CharityML*, in layman's terms, how the final model chosen is supposed to work. Be sure that you are describing the major qualities of the model, such as how the model is trained and how the model makes a prediction. Avoid using advanced mathematical jargon, such as describing equations.

** HINT: **

When explaining your model, if using external resources please include all citations.

**Answer: ** 

AdaBoost consists of fitting a sequence of weak-learners (for example a simple one-level decision tree, called a decision stump, will consistently produce predictions which are slightly better than random guessing, and can be considered a weak learner) to modified versions of the data, and then combining those weak learners into a single strong learner by taking the majority vote of the weak learners for each sample (for example, if 6 weak learners categorize a given sample as '1', and 5 weak learners categorize it as '0', the strong learner will categorize it as '1').

The algorithm applies a weight for each sample, and for the first weak learner, each weight w_1, w_2,..., w_N is 1/N, so each sample has the same importance, and the first weak learner fits the unmodified dataset. 

For each subsequent step, a bigger weight is applied to samples which the previous weak learner incorrectly classified, so that the next weak learner has to focus more on the samples which are harder to categorize, hopefully reducing the bias. The samples correctly predicted also have their weights reduced (since the sum of weights has to be constant at each step).

source: https://scikit-learn.org/stable/modules/ensemble.html#fs1995

### Implementation: Model Tuning
Fine tune the chosen model. Use grid search (`GridSearchCV`) with at least one important parameter tuned with at least 3 different values. You will need to use the entire training set for this. In the code cell below, you will need to implement the following:
- Import [`sklearn.grid_search.GridSearchCV`](http://scikit-learn.org/0.17/modules/generated/sklearn.grid_search.GridSearchCV.html) and [`sklearn.metrics.make_scorer`](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.make_scorer.html).
- Initialize the classifier you've chosen and store it in `clf`.
 - Set a `random_state` if one is available to the same state you set before.
- Create a dictionary of parameters you wish to tune for the chosen model.
 - Example: `parameters = {'parameter' : [list of values]}`.
 - **Note:** Avoid tuning the `max_features` parameter of your learner if that parameter is available!
- Use `make_scorer` to create an `fbeta_score` scoring object (with $\beta = 0.5$).
- Perform grid search on the classifier `clf` using the `'scorer'`, and store it in `grid_obj`.
- Fit the grid search object to the training data (`X_train`, `y_train`), and store it in `grid_fit`.

**Note:** Depending on the algorithm chosen and the parameter list, the following implementation may take some time to run!


```python
# TODO: Import 'GridSearchCV', 'make_scorer', and any other necessary libraries
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer

# TODO: Initialize the classifier
clf = AdaBoostClassifier(random_state=1)

# TODO: Create the parameters list you wish to tune, using a dictionary if needed.
# HINT: parameters = {'parameter_1': [value1, value2], 'parameter_2': [value1, value2]}
#parameters = {'base_estimator':['DecisionTreeClassifier(max_depth=1)', 'DecisionTreeClassifier(max_depth=2)'],'n_estimators':[10, 20, 50, 100, 200], 'learning_rate':[0.1,0.3,1.0,3.0], 'algorithm':['SAMME','SAMME.R']}
parameters = {'n_estimators':[10, 20, 50, 100, 200], 'learning_rate':[0.1,0.3,1.0,3.0], 'algorithm':['SAMME','SAMME.R']}


# TODO: Make an fbeta_score scoring object using make_scorer()
scorer = make_scorer(fbeta_score, beta=0.5)

# TODO: Perform grid search on the classifier using 'scorer' as the scoring method using GridSearchCV()
grid_obj = GridSearchCV(clf, parameters, scoring=scorer)

# TODO: Fit the grid search object to the training data and find the optimal parameters using fit()
grid_fit = grid_obj.fit(X_train, y_train)

# Get the estimator
best_clf = grid_fit.best_estimator_

# Make predictions using the unoptimized and model
predictions = (clf.fit(X_train, y_train)).predict(X_test)
best_predictions = best_clf.predict(X_test)

# Report the before-and-afterscores
print("Unoptimized model\n------")
print("Accuracy score on testing data: {:.4f}".format(accuracy_score(y_test, predictions)))
print("F-score on testing data: {:.4f}".format(fbeta_score(y_test, predictions, beta = 0.5)))
print("\nOptimized Model\n------")
print("Final accuracy score on the testing data: {:.4f}".format(accuracy_score(y_test, best_predictions)))
print("Final F-score on the testing data: {:.4f}".format(fbeta_score(y_test, best_predictions, beta = 0.5)))
```

    /anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1135: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 due to no predicted samples.
      'precision', 'predicted', average, warn_for)
    /anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1135: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 due to no predicted samples.
      'precision', 'predicted', average, warn_for)
    /anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1135: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 due to no predicted samples.
      'precision', 'predicted', average, warn_for)
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:166: RuntimeWarning: invalid value encountered in true_divide
      sample_weight /= sample_weight_sum
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:605: RuntimeWarning: invalid value encountered in greater
      return self.classes_.take(pred > 0, axis=0)
    /anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1135: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 due to no predicted samples.
      'precision', 'predicted', average, warn_for)
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:166: RuntimeWarning: invalid value encountered in true_divide
      sample_weight /= sample_weight_sum
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:605: RuntimeWarning: invalid value encountered in greater
      return self.classes_.take(pred > 0, axis=0)
    /anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1135: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 due to no predicted samples.
      'precision', 'predicted', average, warn_for)
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:166: RuntimeWarning: invalid value encountered in true_divide
      sample_weight /= sample_weight_sum
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:605: RuntimeWarning: invalid value encountered in greater
      return self.classes_.take(pred > 0, axis=0)
    /anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1135: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 due to no predicted samples.
      'precision', 'predicted', average, warn_for)
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:166: RuntimeWarning: invalid value encountered in true_divide
      sample_weight /= sample_weight_sum
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:605: RuntimeWarning: invalid value encountered in greater
      return self.classes_.take(pred > 0, axis=0)
    /anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1135: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 due to no predicted samples.
      'precision', 'predicted', average, warn_for)
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:166: RuntimeWarning: invalid value encountered in true_divide
      sample_weight /= sample_weight_sum
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:605: RuntimeWarning: invalid value encountered in greater
      return self.classes_.take(pred > 0, axis=0)
    /anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1135: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 due to no predicted samples.
      'precision', 'predicted', average, warn_for)
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:166: RuntimeWarning: invalid value encountered in true_divide
      sample_weight /= sample_weight_sum
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:605: RuntimeWarning: invalid value encountered in greater
      return self.classes_.take(pred > 0, axis=0)
    /anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1135: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 due to no predicted samples.
      'precision', 'predicted', average, warn_for)
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:166: RuntimeWarning: invalid value encountered in true_divide
      sample_weight /= sample_weight_sum
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:605: RuntimeWarning: invalid value encountered in greater
      return self.classes_.take(pred > 0, axis=0)
    /anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1135: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 due to no predicted samples.
      'precision', 'predicted', average, warn_for)
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:166: RuntimeWarning: invalid value encountered in true_divide
      sample_weight /= sample_weight_sum
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:605: RuntimeWarning: invalid value encountered in greater
      return self.classes_.take(pred > 0, axis=0)
    /anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1135: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 due to no predicted samples.
      'precision', 'predicted', average, warn_for)
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:166: RuntimeWarning: invalid value encountered in true_divide
      sample_weight /= sample_weight_sum
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:605: RuntimeWarning: invalid value encountered in greater
      return self.classes_.take(pred > 0, axis=0)
    /anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1135: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 due to no predicted samples.
      'precision', 'predicted', average, warn_for)
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:166: RuntimeWarning: invalid value encountered in true_divide
      sample_weight /= sample_weight_sum
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:605: RuntimeWarning: invalid value encountered in greater
      return self.classes_.take(pred > 0, axis=0)
    /anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1135: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 due to no predicted samples.
      'precision', 'predicted', average, warn_for)
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:166: RuntimeWarning: invalid value encountered in true_divide
      sample_weight /= sample_weight_sum
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:605: RuntimeWarning: invalid value encountered in greater
      return self.classes_.take(pred > 0, axis=0)
    /anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1135: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 due to no predicted samples.
      'precision', 'predicted', average, warn_for)
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:166: RuntimeWarning: invalid value encountered in true_divide
      sample_weight /= sample_weight_sum
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:605: RuntimeWarning: invalid value encountered in greater
      return self.classes_.take(pred > 0, axis=0)
    /anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1135: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 due to no predicted samples.
      'precision', 'predicted', average, warn_for)
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:166: RuntimeWarning: invalid value encountered in true_divide
      sample_weight /= sample_weight_sum
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:605: RuntimeWarning: invalid value encountered in greater
      return self.classes_.take(pred > 0, axis=0)
    /anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1135: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 due to no predicted samples.
      'precision', 'predicted', average, warn_for)
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:166: RuntimeWarning: invalid value encountered in true_divide
      sample_weight /= sample_weight_sum
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:605: RuntimeWarning: invalid value encountered in greater
      return self.classes_.take(pred > 0, axis=0)
    /anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1135: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 due to no predicted samples.
      'precision', 'predicted', average, warn_for)
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:166: RuntimeWarning: invalid value encountered in true_divide
      sample_weight /= sample_weight_sum
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:580: RuntimeWarning: invalid value encountered in greater
      ((sample_weight > 0) |
    /anaconda3/lib/python3.7/site-packages/sklearn/ensemble/weight_boosting.py:605: RuntimeWarning: invalid value encountered in greater
      return self.classes_.take(pred > 0, axis=0)
    /anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1135: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 due to no predicted samples.
      'precision', 'predicted', average, warn_for)


    Unoptimized model
    ------
    Accuracy score on testing data: 0.8576
    F-score on testing data: 0.7246
    
    Optimized Model
    ------
    Final accuracy score on the testing data: 0.8651
    Final F-score on the testing data: 0.7396


### Question 5 - Final Model Evaluation

* What is your optimized model's accuracy and F-score on the testing data? 
* Are these scores better or worse than the unoptimized model? 
* How do the results from your optimized model compare to the naive predictor benchmarks you found earlier in **Question 1**?_  

**Note:** Fill in the table below with your results, and then provide discussion in the **Answer** box.

#### Results:

|     Metric     | Unoptimized Model | Optimized Model |
| :------------: | :---------------: | :-------------: | 
| Accuracy Score |         0.8576    |   0.8651        |
| F-score        |         0.7246    |   0.7396        |


**Answer: **

While the optimized model improved in both metrics (accuracy and F-score) over the unoptimized model, the optimization is rather minimal, with an increase of 0.0075 (0.75%) in the accuracy and an increase of 0.015 (1.5%) in the F-score.

Compared to the naive predictor, both models are much better, as the naive predictor has an accuracy score of 0.2478 and an F-score of 0.2917.

----
## Feature Importance

An important task when performing supervised learning on a dataset like the census data we study here is determining which features provide the most predictive power. By focusing on the relationship between only a few crucial features and the target label we simplify our understanding of the phenomenon, which is most always a useful thing to do. In the case of this project, that means we wish to identify a small number of features that most strongly predict whether an individual makes at most or more than \$50,000.

Choose a scikit-learn classifier (e.g., adaboost, random forests) that has a `feature_importance_` attribute, which is a function that ranks the importance of features according to the chosen classifier.  In the next python cell fit this classifier to training set and use this attribute to determine the top 5 most important features for the census dataset.

### Question 6 - Feature Relevance Observation
When **Exploring the Data**, it was shown there are thirteen available features for each individual on record in the census data. Of these thirteen records, which five features do you believe to be most important for prediction, and in what order would you rank them and why?

**Answer:**

I suppose that these five features are the most important for the prediction, in order from most important to least:

- capital gain: it seems obvious that capital gain is correlated with income, and this is in my opinion the best predictor.
- hours per week: part-time employment pays proportionally less than full-time, so there should be a direct correlation between hours spent of the job and income
- education level: those who achieve higher education are far more likely to land better-paying jobs
- age: income tends to correlate with job experience, which in turn correlates with age (although past retirement age this stops being true)
- sex: there is a well-documented gender pay gap in the United States, so I expect sex to be correlated with income, although it will probably be less predictive than the 4 features above.

### Implementation - Extracting Feature Importance
Choose a `scikit-learn` supervised learning algorithm that has a `feature_importance_` attribute availble for it. This attribute is a function that ranks the importance of each feature when making predictions based on the chosen algorithm.

In the code cell below, you will need to implement the following:
 - Import a supervised learning model from sklearn if it is different from the three used earlier.
 - Train the supervised model on the entire training set.
 - Extract the feature importances using `'.feature_importances_'`.


```python
# TODO: Import a supervised learning model that has 'feature_importances_'
from sklearn.ensemble import AdaBoostClassifier

# TODO: Train the supervised model on the training set using .fit(X_train, y_train)
model = AdaBoostClassifier(random_state=1).fit(X_train,y_train)

# TODO: Extract the feature importances using .feature_importances_ 
importances = model.feature_importances_

# Plot
vs.feature_plot(importances, X_train, y_train)
```


![png](output_43_0.png)


### Question 7 - Extracting Feature Importance

Observe the visualization created above which displays the five most relevant features for predicting if an individual makes at most or above \$50,000.  
* How do these five features compare to the five features you discussed in **Question 6**?
* If you were close to the same answer, how does this visualization confirm your thoughts? 
* If you were not close, why do you think these features are more relevant?

**Answer:**

After implementing the feature importance with AdaBoost, I see that I missed the most important feature:  capital loss, which is apparently even more important than capital gain. I can see why this is the case, as people will tend to report only large capital losses, and large capital losses require having had a large capital, which correlates strongly with being in the high income bracket. Indeed 51% of people who report capital losses earn more than 50K, against only 25% of the total sampled population.

Other than that, I only missed the order of the other 4 features, I notably underestimated the importance of age, but overall this visualisation has confirmed my thoughts.

### Feature Selection
How does a model perform if we only use a subset of all the available features in the data? With less features required to train, the expectation is that training and prediction time is much lower — at the cost of performance metrics. From the visualization above, we see that the top five most important features contribute more than half of the importance of **all** features present in the data. This hints that we can attempt to *reduce the feature space* and simplify the information required for the model to learn. The code cell below will use the same optimized model you found earlier, and train it on the same training set *with only the top five important features*. 


```python
# Import functionality for cloning a model
from sklearn.base import clone

# Reduce the feature space
X_train_reduced = X_train[X_train.columns.values[(np.argsort(importances)[::-1])[:5]]]
X_test_reduced = X_test[X_test.columns.values[(np.argsort(importances)[::-1])[:5]]]

# Train on the "best" model found from grid search earlier
clf = (clone(best_clf)).fit(X_train_reduced, y_train)

# Make new predictions
reduced_predictions = clf.predict(X_test_reduced)

# Report scores from the final model using both versions of data
print("Final Model trained on full data\n------")
print("Accuracy on testing data: {:.4f}".format(accuracy_score(y_test, best_predictions)))
print("F-score on testing data: {:.4f}".format(fbeta_score(y_test, best_predictions, beta = 0.5)))
print("\nFinal Model trained on reduced data\n------")
print("Accuracy on testing data: {:.4f}".format(accuracy_score(y_test, reduced_predictions)))
print("F-score on testing data: {:.4f}".format(fbeta_score(y_test, reduced_predictions, beta = 0.5)))
```

    Final Model trained on full data
    ------
    Accuracy on testing data: 0.8651
    F-score on testing data: 0.7396
    
    Final Model trained on reduced data
    ------
    Accuracy on testing data: 0.8385
    F-score on testing data: 0.6920


### Question 8 - Effects of Feature Selection

* How does the final model's F-score and accuracy score on the reduced data using only five features compare to those same scores when all features are used?
* If training time was a factor, would you consider using the reduced data as your training set?

**Answer:**

The final F-score is 0.6920 with the reduced training set, as opposed to 0.7396 with the full training set. We see that the reduced training set does not even attain the score of the unoptimized model (0.7246), the drop in score is quite significant. Still the F-score remains much higher than what we had with the naive estimator.

If training time was a factor, I would try to use another, more cost-effective model before considering dropping columns from the training set, as the differences in training time can be enormous, at a relatively low cost for the F-score. That being said, both options (changing model and using a reduced training set) could be valid depending on the dataset in consideration.
