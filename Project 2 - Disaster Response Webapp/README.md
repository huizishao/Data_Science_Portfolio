# Disaster Response Pipeline Project

### Project Description:

In this project, I analyzed the disaster data from [Figure Eight](https://appen.com/) and built a model for an API that classifies disaster messages. The data set containning real that were sent during disaster events.I created a machine learning pipeline to categorize these events so that the messages to an appropriate disaster relief agency. It also include a web app where an emergency worker can input a new message and get classification results in several categories. The web app will also display visualizations of the data.

### File Descriptions:
The project contains the following files,

* ETL Pipeline Preparation.ipynb: Notebook experiment for the ETL pipelines
* ML Pipeline Preparation.ipynb: Notebook experiment for the machine learning pipelines
* data/process_data.py: The ETL pipeline used to process data in preparation for model building.
* models/train_classifier.py: The Machine Learning pipeline used to fit, tune, evaluate, and export the model to a Python pickle
* app/templates/~.html: HTML pages for the web app
* run.py: Start the Python server for the web app and prepare visualizations.

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/messages.csv data/categories.csv DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py DisasterResponse.db randomforest.pkl`

2. Run the following command in the app's directory to run your web app.
        `python run.py`

3. Go to http://127.0.0.1:5000/