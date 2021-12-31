# Machine Learning  Project (Jupyter Notebook)
Machine Learning and Data Mining


# Datasets Used
1. Robot
2. Airbnb

# How to Execute
1. 'ml-project' folder is having all the codes.
2. 'pip install -r requiremets.txt'
3. cd 'python-scripts'
4. run 'streamlit run streamlit.py'
5. Streamlit will open itself in a new browser window or else you can run it using the ip address provided.

# About Airbnb 
The Airbnb dataset contains details of Airbnb listings for New York City for several years up to and 
including 2019. The dataset includes the location, various details about the host and property, along 
with price and some details about reviews.
I took two approaches to this dataset. I looked at predictions based on price using regression 
models, as price is a continuous variable. And looking at predictions based on popularity of listings, 
which were classification problems.

# Hypothesis
My hypotheses are grouped into price predictions and popularity predictions. Some of which could 
be easily visualised and others through modelling.
• Which hosts are the busiest and why?
• What can we learn from predictions? (ex: locations, prices, reviews, etc)
• What can we learn about different hosts and areas? (which areas are expensive/best)
•Is there any noticeable difference of traffic among different areas and what could be the reason for 
it?
•What type of listings are there in New York City? (Private room, Apartment etc)

# Data Preparation and Exploration
## Airbnb 
### Regression
For the regression models, the price was calculated as average price per room_type per 
neighbourhood. The following fields were dropped as they were not relevant to the problem or were 
duplicated in other fields: id, name, host_name, latitude, longitude, reviews_per_month, 
calculated_host_listings_count and availability_365. After visualising the features, it was clear that 
these features did not have much influence on price and were therefore removed. Further 
experimentation with host_id and number_of _reviews showed some relevance of these features 
with the price, most number_of_reviews lied within the price range of 0-2000 and which hosts had 
moderate/expensive listings
### Classification
There are no fields which contain evaluations of the listing, therefore the ‘number_of_reviews’ field 
was chosen as a possible target variable with the assumption that if a property was to receive many 
negative reviews, demand for that property would drop significantly and it would therefore not 
receive many more reviews. When inspecting the data in this field it was obvious that the data was 
very unbalanced.
The lack of balance in this dataset proved a challenge and had to be dealt with in a number of ways –
from data preparation to the choice of algorithm and validation to make sense of the data and 
results of the models.
Firstly, 20% of data points had no reviews at all. Also, there were a significant number of listings 
where the date of the last review was a number of years ago. This suggested that these properties 
were not active or were so unpopular that they were not relevant to the hypotheses. It was 
therefore decided to drop all listings of 0 reviews and any, where the date of the last review was 
prior to 2015. 
The values of number_of_reviews field ranged from 0 < 630. It was necessary to split the data into 
bins in order to deal with the lack of balance. Quartile bins, which are bins with equal number of 
datapoints were created. However, using pandas qcut reduced the accuracy of the models. Using 
pandas cut instead with boundaries of the bins limiting the number of data points similarly to 
quartiles but by reducing the imbalance rather than eliminating it provided better accuracy results.
This resulted in the following bins: 1-30, 31-50, 51-650. The categorical fields in the independent
variables were then converted to binary data using one-hot encoding for passing to the models.
The SMOTE function (Synthetic Minority Over-sampling Technique) was also used to address the 
imbalance of the dataset. This function oversamples the minority class by using a set number of 
nearest neighbours of in the minority class to create synthetic samples in that class. This resulted in 
a very balanced dataset, but the models did not train well and underfitted

## Algorithms
### Classification
#### Naïve Bayes
Naïve Bayes classifies variables according to Bayes theorem with the assumption that variables are 
independent. 
The Robot dataset was assumed to be independent, so it would make sense to apply this algorithm.
Algorithm parameters were left as default values.
Although in the Airbnb dataset it was not clear where the features were independent or not, the 
model was applied to the dataset.

#### Decision Tree
The Decision Tree algorithm was chosen as an algorithm which is generally good for most datasets. It 
is particularly useful for cases where it is useful or sometimes essential to understand why a 
particular decision was made by the model.
For both Robot and Airbnb, the ccp_alpha value was set to 0.005 to perform minimal cost 
complexity pruning. Random state was set to 5 to obtain a deterministic behaviour during fitting. All 
other parameters were left as default.

#### Perceptron
The Perceptron algorithm is applied to binary classification of data. Once both datasets had been 1-
hot encoded, this was an easy choice for a model. For Robot eta0 was set to 0.1 after experimenting 
with a few values including the default = 1. Similarly, max_iter was set to 100 following 
experimentations. Similar experimentations were conducted for Airbnb, however the algorithm 
proved not best suited to this dataset.

#### Multi-Layer Perceptron
MLP is an effective algorithm for non-linear separable problems and therefore suitable for both 
datasets. For the Robot several different parameter settings were tried until the optimum values were found 
for suitable class separation. The number of hidden layers was set as 6; number of epochs = 500; 
learning rate = 0.05.
For Airbnb the MLP need to be adjusted, as a first step the train /test needed to be normalised. This
improved the accuracy by 2%. Hidden layers were changed from 5 to 10 as well as max_iterations 
from 100 to 1000. The activation function used was logistic because it improved the performance 
when compared to the default = ‘relu’. It was therefore decided to use the same parameter settings 
as for Robot.

#### Support Vector Machine
SVM is a linear classifier which learns an (n – 1)-dimensional classifier for classification of data into 
two classes. SVM is more efficient for two class classification and both the Robot and Airbnb 
datasets had more than 2 classes, and running the model proved how inappropriate it was for our 
datasets as it performed very poorly, despite trying different parameters. SVM is also very efficient 
for datasets with a large number of features. Neither datasets had a large number of features, with
Robot dataset having 24/4 features and Airbnb having 16 features, another reason the model did 
not perform well.

#### XGBoost
XGBoost (Extreme Gradient Boost) is decision tree ensemble algorithm with gradient boosting 
framework. 
Gradient boosting is a supervised learning algorithm, which attempts to accurately predict a target 
variable by combining the estimates of a set of simpler, weaker models.
The algorithm approaches the process of sequential tree building using parallelized implementation 
and comes with built-in cross validation method at each iteration.

#### Keras Neural Network
The Keras Neural Network model was applied to the Robot dataset with the number of epochs 
reduced to 200 and the number of hidden layer was set as 35 which gave a better performance 
when compared to the MLP with 500 epochs and no 6 hidden layers(6 hidden layers gave the best 
performance for MLP).

### Regression 
Two regression models were applied to the Airbnb regression problem: OLS and Decision Tree 
Regressor.

#### OLS
Ordinary Least Squares Regression model estimate the parameters in a regression model by 
minimizing the sum of the squared residuals. 

#### Decision Tree Regressor
The regression decision tree works to fit a sine curve to learn local linear regressions which aim to fit 
the sine curve. For Airbnb the max_depth was the only parameter, which was modified from the 
default and set to 20 after some experimentation.

# Performance Metrics 
Several different performance metrics were used to evaluate the models. 

## ROC-AUC
Receiver Operating Characteristic curve plots the false positive rate against the true positive rate and 
shows how well the model is capable of distinguishing between the classes. ROC curves are used 
when there are roughly equal numbers of observations for each class. For our models ROC curves 
have been used to evaluate output of the models trained on the Robot dataset. 

### Precision Recall Curve
Precision-Recall curves plots the true positive rate against the positive predicted value and describes 
how good the model is at predicting the positive class. Precision recall curve is better suited for use 
when there is a moderate to large class imbalance, as there is in the Airbnb classification problem.

### Confusion Matrix
A confusion matrix is a table which describes the performance of a classification model by providing 
a summary of prediction results. Many modelling evaluation metrics use the confusion matrix as the 
basis for their metrics. Both Robot and Airbnb classifications were assessed with a confusion matrix.

### K-fold Cross Validation
K-fold cross validation is a method of checking the accuracy of a model’s ability to make predictions 
on unseen data. K-fold tends to be less biased – that is less optimistic in its evaluation of a model 
and allows all observations from the original dataset to appear in training and test sets. K-fold was 
applied to both Robot and Airbnb classification problems with 10 folds

## Accuracy
Accuracy is a simple performance measure of correct predictions out of total predictions. It is a quick 
and straightforward indication of the performance of a model. Accuracy has been applied to all 
Robot and Airbnb classification problems.

## Regression Performance

### R2
R-squared represents the proportion of the variance for a dependent variable which is explained by 
independent variables. R-squared explains to what extent the variance of one variable explains the 
variance of another variable.

### MSE
Mean Squared Error measures the average of the squares of errors between the estimated values 
and what is estimated. This is a risk function corresponding to the expected value of the squared 
error loss.

### MAE
Mean Absolute Error measures the average magnitude of the error in a set of predictions regardless 
of the error direction – in other words taking the absolute value.

# Results
## Airbnb 
### Classification
The Airbnb dataset for classification problem was very unbalanced. Some steps were taken to 
balance this dataset as discussed in Data Preparation and Exploration.
Results of Naïve Bayes model for Airbnb accuracy were ~ 70%. This can be explained by a few 
factors. The imbalanced dataset was a major factor, even though this was addressed to some extent 
by using bins for the target variable. The variation in values within the bins was quite high and 
therefore the results were not very meaningful.
With the Decision Tree model the results were similar to Naïve Bayes, which was a bit unexpected
with accuracy about 73%. It was assumed that the DT would be able to learn more about the 
relationships between the variables and therefore be able to generalise better. 
Results from the Perceptron model were as expected with accuracy for training data at 56% and test 
data 51%. 
The Multi-layer Perceptron performed better than Perceptron, with the conclusion that the extra 
hidden layers and the extra learned weights contributed to a better model. However, like the results 
of the other classification models for Airbnb MLP’s performance with accuracy of ~ 73% was 
comparable.

## Airbnb Classification Conclusion 
The models’ accuracy explains how popular listings are, implying which hosts would be the busiest -
by predicting the reviews_bins. However, the data distribution in reviews_bins was quite uneven, so 
the predictions don’t give meaningful information. The main conclusions from the Airbnb classification problem is that this dataset is too unbalanced and attempts to balance hide any 
relationships between variables, such as the variation of the values in bins was quite large and 
minimised any relationship.

### Regression
Looking at Airbnb data as a regression problem proved more productive, this was largely due to 
higher degree of accuracy of calculating an average price per room_type per neighbourhood and the 
relevancy of this derived figure to other variables. 
The results of the OLS model show a high value for R-squared and Adjusted R-squared with 0.938.
This shows that the variance of room_type, neighbourhood and neighbourhood_group explains the 
variance in the average price very well. Similarly, the F-statistic value of 3318 in comparison to the p-value of 0.00 shows this group of variables to be significant in predicting the average price of listings.
Results of the Decision Tree Regression performance were similar to OLS with R-squared value of 
0.9866 showing again that the variance of the independent variables explains the variance in the 
dependent variable. The MSE is 86.17 which seems quite high. However, taking into account R-squared value and MAE of 4.09, suggests MSE is reasonable. It is therefore reasonable to conclude 
that the model performs well.

#### Airbnb Regression Conclusion 
The models can accurately predict the price per neighbourhood explaining which neighbourhood 
would be more expensive. The greater the accuracy on test data, the closer the predictions would be
to the actual average price i.e. we can learn about the average prices of neighbourhoods by the 
models’ predictions
