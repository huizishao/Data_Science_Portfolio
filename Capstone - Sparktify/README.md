# Capstone Project - Sparkify Music Service Analytics

### Motivation

For many mustic streaming services, every time when users interact with the service, while they are playing songs, logging out and etc, there generate data. All these data contain the key insight for predicting the churn (Cancel the service) of the users and keeping the business thrive. Because of the size of the data, it is a challenging and common problem that we regularly encounter in any customer-facing business. Additionally, the ability to efficiently manipulate large datasets with Spark is one of the highest-demand skills in the field of data.

### Project Objective

Using the user information logs, we attempt to use Spark MLlib to build machine learning models to predict the churning possibilities of a user and understand the features that contribute to the churning behaviors of the users.

### Data Descriptions

The full dataset is 12GB. Here we only analyze the small subset of it. The data contains user's demographic info, users activities, timestamps and etc. 

### File Descriptions

- Sprakify.ipynb: main file of the project, contains the process of exploring the data and build the machine learning model using pyspark
- mini_sparkify_event_data.json: mini subset of full dataset for analysis

### Results

With the features I engineered as in the project, Gradient Boosting Tree Model can give F1 score of 0.88. The active days and adding to playlist are tow of the most important factors for the churn of the users. We could offer discount based on those options.


### Blog

For further discussion you could find more details and visuals at my medium blog post available [here](https://medium.com/@jessie.sssy/understanding-customer-churning-abd6525d61c5)

For my GitHub repository, you can click [here](https://github.com/huizishao/Udacity_DataScientist_Nanodegree)

### Required Packages

* Pandas
* pyspark
* matplot
* numpy