
# coding: utf-8

# # Machine Learning Engineer Nanodegree
# ## Introduction and Foundations
# ## Project: Titanic Survival Exploration
# 
# In 1912, the ship RMS Titanic struck an iceberg on its maiden voyage and sank, resulting in the deaths of most of its passengers and crew. In this introductory project, we will explore a subset of the RMS Titanic passenger manifest to determine which features best predict whether someone survived or did not survive. To complete this project, you will need to implement several conditional predictions and answer the questions below. Your project submission will be evaluated based on the completion of the code and your responses to the questions.
# 

# # Getting Started
# To begin working with the RMS Titanic passenger data, we will first need to `import` the functionality we need, and load our data into a `pandas` DataFrame.  
# Run the code cell below to load our data and display the first few entries (passengers) for examination using the `.head()` function.
# 

# In[3]:

# Import libraries necessary for this project
import numpy as np
import pandas as pd
from IPython.display import display # Allows the use of display() for DataFrames

# Import supplementary visualizations code visuals.py
import visuals as vs

# Pretty display for notebooks
get_ipython().magic(u'matplotlib inline')

# Load the dataset
in_file = 'titanic_data.csv'
full_data = pd.read_csv(in_file)

# Print the first few entries of the RMS Titanic data
display(full_data.head())


# From a sample of the RMS Titanic data, we can see the various features present for each passenger on the ship:
# - **Survived**: Outcome of survival (0 = No; 1 = Yes)
# - **Pclass**: Socio-economic class (1 = Upper class; 2 = Middle class; 3 = Lower class)
# - **Name**: Name of passenger
# - **Sex**: Sex of the passenger
# - **Age**: Age of the passenger (Some entries contain `NaN`)
# - **SibSp**: Number of siblings and spouses of the passenger aboard
# - **Parch**: Number of parents and children of the passenger aboard
# - **Ticket**: Ticket number of the passenger
# - **Fare**: Fare paid by the passenger
# - **Cabin** Cabin number of the passenger (Some entries contain `NaN`)
# - **Embarked**: Port of embarkation of the passenger (C = Cherbourg; Q = Queenstown; S = Southampton)
# 
# Since we're interested in the outcome of survival for each passenger or crew member, we can remove the **Survived** feature from this dataset and store it as its own separate variable `outcomes`. We will use these outcomes as our prediction targets.  
# Run the code cell below to remove **Survived** as a feature of the dataset and store it in `outcomes`.

# In[4]:

# Store the 'Survived' feature in a new variable and remove it from the dataset
outcomes = full_data['Survived']
data = full_data.drop('Survived', axis = 1)

# Show the new dataset with 'Survived' removed
display(data.head())


# The very same sample of the RMS Titanic data now shows the **Survived** feature removed from the DataFrame. Note that `data` (the passenger data) and `outcomes` (the outcomes of survival) are now *paired*. That means for any passenger `data.loc[i]`, they have the survival outcome `outcomes[i]`.
# 
# To measure the performance of our predictions, we need a metric to score our predictions against the true outcomes of survival. Since we are interested in how *accurate* our predictions are, we will calculate the proportion of passengers where our prediction of their survival is correct. Run the code cell below to create our `accuracy_score` function and test a prediction on the first five passengers.  

# In[5]:

def accuracy_score(truth, pred):
    """ Returns accuracy score for input truth and predictions. """
    
    # Ensure that the number of predictions matches number of outcomes
    if len(truth) == len(pred): 
        
        # Calculate and return the accuracy as a percent
        return "Predictions have an accuracy of {:.2f}%.".format((truth == pred).mean()*100)
    
    else:
        return "Number of predictions does not match number of outcomes!"
    
# Test the 'accuracy_score' function
predictions = pd.Series(np.ones(5, dtype = int))
print accuracy_score(outcomes[:5], predictions)


# # Making Predictions
# 
# If we were asked to make a prediction about any passenger aboard the RMS Titanic whom we knew nothing about, then the best prediction we could make would be that they did not survive. This is because we can assume that a majority of the passengers (more than 50%) did not survive the ship sinking.  
# The `predictions_0` function below will always predict that a passenger did not survive.

# In[6]:

def predictions_0(data):
    """ Model with no features. Always predicts a passenger did not survive. """

    predictions = []
    for _, passenger in data.iterrows():
        
        # Predict the survival of 'passenger'
        predictions.append(0)
    
    # Return our predictions
    return pd.Series(predictions)

# Make the predictions
predictions = predictions_0(data)


# In[7]:

print accuracy_score(outcomes, predictions)


# Let's take a look at whether the feature **Sex** has any indication of survival rates among passengers using the `survival_stats` function. This function is defined in the `visuals.py` Python script included with this project. The first two parameters passed to the function are the RMS Titanic data and passenger survival outcomes, respectively. The third parameter indicates which feature we want to plot survival statistics across.  
# Run the code cell below to plot the survival outcomes of passengers based on their sex.

# In[8]:

vs.survival_stats(data, outcomes, 'Sex')


# In[9]:

def predictions_1(data):
    """ Model with one feature: 
            - Predict a passenger survived if they are female. """
    
    predictions = []
    for _, passenger in data.iterrows():
        
        if passenger['Sex']=='female':
            predictions.append(1)
        else:
            predictions.append(0)
    
    # Return our predictions
    return pd.Series(predictions)

# Make the predictions
predictions = predictions_1(data)


# In[10]:

print accuracy_score(outcomes, predictions)


# 
# Using just the **Sex** feature for each passenger, we are able to increase the accuracy of our predictions by a significant margin. Now, let's consider using an additional feature to see if we can further improve our predictions. For example, consider all of the male passengers aboard the RMS Titanic: Can we find a subset of those passengers that had a higher rate of survival? Let's start by looking at the **Age** of each male, by again using the `survival_stats` function. This time, we'll use a fourth parameter to filter out the data so that only passengers with the **Sex** 'male' will be included.  
# Run the code cell below to plot the survival outcomes of male passengers based on their age.

# In[11]:

vs.survival_stats(data, outcomes, 'Age', ["Sex == 'male'", "Pclass == 1"])


# In[12]:

def predictions_2(data):
    """ Model with two features: 
            - Predict a passenger survived if they are female.
            - Predict a passenger survived if they are male and younger than 10. """
    
    predictions = []
    for _, passenger in data.iterrows():
        
        if passenger['Sex'] == 'female':
            predictions.append(1)
        else:
            if passenger['Age']<10:
                predictions.append(1)
            else:
                predictions.append(0)
        # Remove the 'pass' statement below 
        # and write your prediction conditions here
    
    # Return our predictions
    return pd.Series(predictions)

# Make the predictions
predictions = predictions_2(data)


# In[13]:

print accuracy_score(outcomes, predictions)


# Adding the feature **Age** as a condition in conjunction with **Sex** improves the accuracy by a small margin more than with simply using the feature **Sex** alone. Now it's your turn: Find a series of features and conditions to split the data on to obtain an outcome prediction accuracy of at least 80%. This may require multiple features and multiple levels of conditional statements to succeed. You can use the same feature multiple times with different conditions.   

# In[14]:

vs.survival_stats(data, outcomes, 'Age', ["Sex == 'male'", "Pclass == 3"])


# In[28]:

def predictions_3(data):
    """ Model with multiple features. Makes a prediction with an accuracy of at least 80%. """
    
    predictions = []
    for _, passenger in data.iterrows():
        if passenger['Sex'] == 'female':
            if  passenger['Pclass'] == 3 and passenger['Age']>40 and passenger['Age']<50:
                predictions.append(0)
            else:
                predictions.append(1)
        else:
            if passenger['Age']<=10:
                predictions.append(1)
            elif passenger['Age'] < 37 and passenger['Pclass'] == 1 and passenger['Age'] > 33:
                predictions.append(1)
            else:
                predictions.append(0)
    # Return our predictions
    return pd.Series(predictions)

# Make the predictions
predictions = predictions_3(data)


# In[29]:

print accuracy_score(outcomes, predictions)


# We used Pclass feature along with sex and age to get the prediction. PClass suggests that the higher class had the priority when the passengers were strugling to survive. By visualizing the combination of sex and age and Pclass, I deduced some conditions and implemented. The combination of features were more informative than the individual features. The first split is based on sex. For females, the next splits were done using their Pclass and Age and for men, further splits were done Age and then combination of Age and PClass.<br>
# 
# Predictions have an accuracy of 80.81%.

# # Conclusion
# 
# After several iterations of exploring and conditioning on the data, you have built a useful algorithm for predicting the survival of each passenger aboard the RMS Titanic. The technique applied in this project is a manual implementation of a simple machine learning model, the *decision tree*. A decision tree splits a set of data into smaller and smaller groups (called *nodes*), by one feature at a time. Each time a subset of the data is split, our predictions become more accurate if each of the resulting subgroups are more homogeneous (contain similar labels) than before. 
# 
# A decision tree is just one of many models that come from *supervised learning*. In supervised learning, we attempt to use features of the data to predict or model things with objective outcome labels. That is to say, each of our data points has a known outcome value, such as a categorical, discrete label like `'Survived'`, or a numerical, continuous value like predicting the price of a house.

# Supervised alogorithms can be used to suggest similar bought products by the customers when they shop online. For example: If I bought dairy products and there's a high likelyhood of me buying a new brand of same dairy product. So, based on historical data, target variable is bought and features could be age or sex.

# In[ ]:

THE END


# In[ ]:



