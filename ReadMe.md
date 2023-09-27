# Dataset
brief explanation of each feature:

* CustomerID: A unique identifier for each customer.
* Name: The name of the customer (text).
* Age: The age of the customer (integer).
* Gender: The gender of the customer (text, e.g. Male or Female).
* Location: The location or city where the customer is located (5 location  ['Los Angeles' 'New York' 'Miami' 'Chicago' 'Houston']).
* Subscription_Length_Months: The duration of the customer's subscription in months (integer).
* Monthly_Bill: The amount billed to the customer each month (numeric).
* Total_Usage_GB: The total data usage in gigabytes by the customer (numeric).
* Churn: A binary indicator of whether the customer has churned (0 for No, 1 for Yes).

<hr>

## Correlations

Summary of correlations between churn and other variables:

* Churn vs. Age: Positive but very weak correlation (0.001559)
* Churn vs. Subscription_Length_Months: Positive but very weak correlation (0.002328)
* Churn vs. Monthly_Bill: Close to zero correlation (-0.000211)
* Churn vs. Total_Usage_GB: Negative but very weak correlation (-0.002842)
 
Conclusion: There are no strong linear relationships between churn and the other variables in the dataset. Therefore, none of these features strongly determine or predict churn.

I.e Churn is not strongly correlated with any of the other variables, so it cannot be predicted with certainty based on those variables alone.

<hr>

# Feature Engineering 

* Total_Spend : Total amount spent by each customer during their subscription.
* Data_Value : Indicates how efficiently a customer uses data in relation to their bill.
* Age_Group : Categorizes customers into age brackets: 'Young,' 'Middle-aged,' or 'Senior.'
* Loyal_Customer : Identifies if a customer is loyal (1) or not (0) based on a long subscription.
* Monthly_Usage_per_Age : Average data usage for each age group.
* Usage_Difference : Difference between a customer's data usage and their age group's average usage, highlighting outliers or unique patterns.

<hr>


## Model Optimization:
* evaluate the performance of four different machine learning models on a classification task. The four models are:

    * Logistic Regression
    * Decision Tree
    * K-Nearest Neighbors
    * Random Forest

* Evaluate the performance of custom classifiers using 5-fold cross-validation.
* Tune the best-performing KNN model using GridSearchCV for hyperparameter tuning and save the optimized model.


### Result: 
* Precision (P): 50% of predicted "Fraud" cases were correct.
* Recall (R): 50% of actual "Fraud" cases were detected.
* F1-Score (F1): A balanced measure combining P and R.
* Accuracy: Correctly predicted 50% of cases.
* Class Distribution: About 50% "Not Fraud" and 50% "Fraud.

Not good Prediction

<hr>

## Improvement to be done :
* The dataset lacks in data/ feature that are corelated to the churn . 
* New feature needed
* Feature Engineering 

<hr>


# Deployment:

Made a streamlit app and hosted the app on stremlit cloud .