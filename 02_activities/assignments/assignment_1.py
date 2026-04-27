#!/usr/bin/env python
# coding: utf-8

# # Assignment 1

# You only need to write one line of code for each question. When answering questions that ask you to identify or interpret something, the length of your response doesn’t matter. For example, if the answer is just ‘yes,’ ‘no,’ or a number, you can just give that answer without adding anything else.
# 
# We will go through comparable code and concepts in the live learning session. If you run into trouble, start by using the help `help()` function in Python, to get information about the datasets and function in question. The internet is also a great resource when coding (though note that **no outside searches are required by the assignment!**). If you do incorporate code from the internet, please cite the source within your code (providing a URL is sufficient).
# 
# Please bring questions that you cannot work out on your own to office hours, work periods or share with your peers on Slack. We will work with you through the issue.

# ### Classification using KNN
# 
# Let's set up our workspace and use the **Wine dataset** from `scikit-learn`. This dataset contains 178 wine samples with 13 chemical features, used to classify wines into different classes based on their origin.
# 
# The **response variable** is `class`, which indicates the type of wine. We'll use all of the chemical features to predict this response variable.

# In[15]:


# Import standard libraries
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import recall_score, precision_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score


# In[16]:


from sklearn.datasets import load_wine

# Load the Wine dataset
wine_data = load_wine()

# Convert to DataFrame
wine_df = pd.DataFrame(wine_data.data, columns=wine_data.feature_names)

# Bind the 'class' (wine target) to the DataFrame
wine_df['class'] = wine_data.target

# Display the DataFrame
wine_df


# #### **Question 1:** 
# #### Data inspection
# 
# Before fitting any model, it is essential to understand our data. **Use Python code** to answer the following questions about the **Wine dataset**:
# 
# _(i)_ How many observations (rows) does the dataset contain?

# In[17]:


wine_df.shape[0]



# _(ii)_ How many variables (columns) does the dataset contain?

# In[ ]:


wine_df.shape[1]


# _(iii)_ What is the 'variable type' of the response variable `class` (e.g., 'integer', 'category', etc.)? What are the 'levels' (unique values) of the variable?

# In[ ]:


wine_df['class'].dtype, wine_df['class'].unique()


# 
# _(iv)_ How many predictor variables do we have (Hint: all variables other than `class`)? 

# In[ ]:


wine_df.shape[1] - 1 


# You can use `print()` and `describe()` to help answer these questions.

# #### **Question 2:** 
# #### Standardization and data-splitting
# 
# Next, we must preform 'pre-processing' or 'data munging', to prepare our data for classification/prediction. For KNN, there are three essential steps. A first essential step is to 'standardize' the predictor variables. We can achieve this using the scaler method, provided as follows:

# In[ ]:


# Select predictors (excluding the last column)
predictors = wine_df.iloc[:, :-1]

# Standardize the predictors
scaler = StandardScaler()
predictors_standardized = pd.DataFrame(scaler.fit_transform(predictors), columns=predictors.columns)

# Display the head of the standardized predictors
print(predictors_standardized.head())


# (i) Why is it important to standardize the predictor variables?

# It is because KNN is dependant on distance between data points and variables on larger scales can dominate those distances if we don't standardize. 

# (ii) Why did we elect not to standard our response variable `Class`?

# We chose Class because class is a categorical as opposed to continous, so scaling has no meaning. 

# (iii) A second essential step is to set a random seed. Do so below (Hint: use the random.seed function). Why is setting a seed important? Is the particular seed value important? Why or why not?

# random.seed(123) Setting a random seed ensures reproducbility by fixing the sequence of pseudo-random numbers used in processes such as train-testing splitting. Without a fixed seed, each run may produce different data parittions and thus, different model outcomes. The type of seed value itself is not critical, as any value can yield a consistence sequence, what matters is that the same seed is used to ensure results can be replicated and compared reliably.

# (iv) A third essential step is to split our standardized data into separate training and testing sets. We will split into 75% training and 25% testing. The provided code randomly partitions our data, and creates linked training sets for the predictors and response variables. 
# 
# Extend the code to create a non-overlapping test set for the predictors and response variables.

# In[ ]:


# set a seed for reproducibility
np.random.seed(123)
random.seed(123)

# split the data into a training and testing set. hint: use train_test_split !

X_train, X_test, y_train, y_test = train_test_split (predictors_standardized, wine_df['class'], test_size=0.25, random_state=123, stratify=wine_df['class'])


# #### **Question 3:**
# #### Model initialization and cross-validation
# We are finally set to fit the KNN model. 
# 
# 
# Perform a grid search to tune the `n_neighbors` hyperparameter using 10-fold cross-validation. Follow these steps:
# 
# 1. Initialize the KNN classifier using `KNeighborsClassifier()`.
# 2. Define a parameter grid for `n_neighbors` ranging from 1 to 50.
# 3. Implement a grid search using `GridSearchCV` with 10-fold cross-validation to find the optimal number of neighbors.
# 4. After fitting the model on the training data, identify and return the best value for `n_neighbors` based on the grid search results.

# In[ ]:


knn = KNeighborsClassifier()

param_grid = {'n_neighbors' : range(1, 51)}

grid = GridSearchCV(knn, param_grid, cv=10)
grid.fit(X_train, y_train)

best_k = grid.best_params_['n_neighbors']
best_k


# #### **Question 4:**
# #### Model evaluation
# 
# Using the best value for `n_neighbors`, fit a KNN model on the training data and evaluate its performance on the test set using `accuracy_score`.

# In[ ]:


knn_final = KNeighborsClassifier(n_neighbors=best_k)

knn_final.fit(X_train, y_train) 

y_pred = knn_final.predict(X_test)

accuracy_score(y_test, y_pred)


# # Criteria
# 
# 
# | **Criteria**                                           | **Complete**                                      | **Incomplete**                                    |
# |--------------------------------------------------------|---------------------------------------------------|--------------------------------------------------|
# | **Data Inspection**                                    | Data is inspected for number of variables, observations and data types. | Data inspection is missing or incomplete.         |
# | **Data Scaling**                                       | Data scaling or normalization is applied where necessary (e.g., using `StandardScaler`). | Data scaling or normalization is missing or incorrectly applied. |
# | **Model Initialization**                               | The KNN model is correctly initialized and a random seed is set for reproducibility.            | The KNN model is not initialized, is incorrect, or lacks a random seed for reproducibility. |
# | **Parameter Grid for `n_neighbors`**                   | The parameter grid for `n_neighbors` is correctly defined. | The parameter grid is missing or incorrectly defined. |
# | **Cross-Validation Setup**                             | Cross-validation is set up correctly with 10 folds. | Cross-validation is missing or incorrectly set up. |
# | **Best Hyperparameter (`n_neighbors`) Selection**       | The best value for `n_neighbors` is identified using the grid search results. | The best `n_neighbors` is not selected or incorrect. |
# | **Model Evaluation on Test Data**                      | The model is evaluated on the test data using accuracy. | The model evaluation is missing or uses the wrong metric. |
# 

# ## Submission Information
# 
# 🚨 **Please review our [Assignment Submission Guide](https://github.com/UofT-DSI/onboarding/blob/main/onboarding_documents/submissions.md)** 🚨 for detailed instructions on how to format, branch, and submit your work. Following these guidelines is crucial for your submissions to be evaluated correctly.
# 
# ### Note:
# 
# If you like, you may collaborate with others in the cohort. If you choose to do so, please indicate with whom you have worked with in your pull request by tagging their GitHub username. Separate submissions are required.
# 
# ### Submission Parameters:
# * Submission Due Date: `11:59 PM - 04/19/2026`
# * The branch name for your repo should be: `assignment-1`
# * What to submit for this assignment:
#     * This Jupyter Notebook (assignment_1.ipynb) should be populated and should be the only change in your pull request.
# * What the pull request link should look like for this assignment: `https://github.com/<your_github_username>/LCR/pull/<pr_id>`
#     * Open a private window in your browser. Copy and paste the link to your pull request into the address bar. Make sure you can see your pull request properly. This helps the technical facilitator and learning support staff review your submission easily.
# 
# Checklist:
# - [ ] Created a branch with the correct naming convention.
# - [ ] Ensured that the repository is public.
# - [ ] Reviewed the PR description guidelines and adhered to them.
# - [ ] Verify that the link is accessible in a private browser window.
# 
# If you encounter any difficulties or have questions, please don't hesitate to reach out to our team via our Slack at `#dc3-help`. Our Technical Facilitators and Learning Support staff are here to help you navigate any challenges.
# 
