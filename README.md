# sparkify_churn_events
Background
This project investigates customer churn for a fictional music streaming app (Sparkify). Using spark a large database of user behaviour is used to discern features that could be used to predict user churn. The investigation is conducted on a smaller subset of data with the intention to apply the model to the full dataset. This investigation is completed in 5 steps. Initially data is loaded and cleaned, then and initial investigation with visualization is completed. This is followed by an initial modelling phase is completed and a modelling technieuq is selected. This model is then refined by tuning the parameters. Finally feature importance is displayed.

Files
Sparkify.ipynb: the python notebook containing the investigation
mini_sparkify_event_data.zip: small subset of the dataset used to train and test the model: the json file had to be compressed to upload to github as the file size was too large, download and unzip to json in order to use the file

Packages
ETL packages: packages used to prepare and clean data. Also to start the spark session used throughout the investigation
import pandas as pd
import numpy as np
from pyspark.sql import SparkSession, Window
import datetime
import pyspark.sql.functions as F
import pyspark.sql.types as T

Visualization Packages: pacakges used to visualize and plot data
import matplotlib.pyplot as plt
import seaborn as sns

Modeling packages: used to prepare and model data 
from pyspark.ml.feature import VectorAssembler, Normalizer, StandardScaler
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, LinearSVC, GBTClassifier, OneVsRest
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.mllib.evaluation import MulticlassMetrics

Getting started: how to run the code
Open the Sparkify.ipynb in a Jupyter notebook and run each block sequentially

Approach and Results
In this analysis customer churn was investigated by first cleaning up the data, then selecting features that could be relevant and used to predict churn. The features chosen were:
    Thumbs up: are users who like songs more or less likely to cancel their account: users who like a lot of songs are probably less likely to cancel a membership as they are       enjoying the app more
    Thumbs down: are users who dislike songs more or less likely to cancel their account:: users who dislike a lot of songs are probably more likely to cancel a membership as       they are enjoying the app less
    Gender: does gender predict churn rate: are men or women more likely to cancel an account?
    Total songs: does the number of songs listened to predict churn rate: the more songs a user listens to, the more the use the app suggesting they enjoy it more and are less       likely to cancel their membership
    Upgrade: if a user upgrades from a free to a paid account, can this predict churn events: less likely to cancel their membership as they enjoyed the app enough to pay for       additional features
    Add friends: if a user has added friends to their account, can this predict churn events: if they add friends more likely to be using the app more and enjoying it more so       less likely to cancel membership
These features were then engineered into numerical values to prepare for modelling.
Initially 4 models were run to see which would work best with the dataset and features:
  1.Logistic Regression: commonly used as an initial model to try as it is easy to use and quick to implement. It is also more robust to overfitting
    Accuracy = 0.766
    F1 value = 0.7281
  2.Random Forest Classifier: model based on decision trees trained in parallel that works well with large data sets
    Accuracy = 0.809
    F1 value = 0.756
  3.Gradient Boosting Classifier: model based on sequentially trained decision trees that works well with unbalanced data
    Accuracy = 0.723
    F1 value = 0.729
  4.Linear Support Vector Machine: model that differentiates between two classifications by creating a hyperplane in the a a space that has dimensions equal to the number of         features. It does this as many times as is required to find the maximum distance between data points of both classifications. Works well with a large number of features.
      Accuracy = 0.809
      F1 value = 0.723
Model effectiveness was judged on F1 value rather than accuracy. This is because the data was unbalanced with very few churn events in the testing data so looking at a weighted measurement of accuracy gives a better idea of model performance than a normal accuracy calculation
Looking at the F1 value, random forest classifier had the highest F1 so this was chosen to further parameter tuning. We ran the tuned model on the same data with a larger testing dataset to see how it performed. However tuning the parameters did no improve the F1 value and decreased the accuracy scores. This suggests that increasing the testing data set decreased the accuracy, as it is not a weighted representaton of model effectiveness, but that the model was already at its peak performance with the standard parameters. 

The results of the random forest model are not terrible and do well to predict a sizeable amount of churn behaviour, however it could be improved. Further investigative research into feature choice could improve the results of the model. Also a better understanding of how features are correlated with one another could also improve results.

The final analysis was investigating which features showed the greatest importance. The feature with the greatest importance was the add_friend feature (0.2292). This was followed by song total (0.2131), then thumbs down (0.196), and then thumbs up (0.1896), which all had similar levels of importance. Finally the least important features were upgrade (0.0844) and gender (0.0443).
