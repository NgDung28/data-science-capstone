# Dog Breed Classifier Web App

## Udacity Data Scientist Nanodegree Capstone Project

## Table of Contents
 * [Project Motivation](#project-motivation)
 * [File Descriptions](#file-descriptions)
 * [Instructions](#instructions)
 * [Licensing, Authors, Acknowledgements, etc.](#licensing-authors-acknowledgements-etc)
 
### Project Motivation
In this project, I have applied my data engineering skills to use machine learning and artificial intellegence to classify images of dogs according to their breed. 
I have created a pipeline to process real-world, user-supplied images. Given an image of a dog, your algorithm will identify an estimate of the canine’s breed. If supplied an image of a human, the code will identify the resembling dog breed.
Also, this project includes a web app where users can upload their image directly to the website and get prediction result. 

### File Descriptions
app    

| - templates    
| |- master.html # main page of web app    
| |- go.html # prediction result page of web app    
|- run.py # Flask file that runs app    

models    

| - haarcascade   
| |- haarcascade_frontalface_alt.xml # opencv model to detect faces      
|- saved_models
| |- weights.best.resnet.hdf5 # resnet50 model to detect dog breed
|- dog_names.p #  list of dog names
|- classifier.py # load model and predict functions

images - folder of uploaded image files

dog_app.ipynb #  detail pipeline to process, train and test models

README.md    

### Instructions:
1. Run dog_app.ipynb to run pipeline that trains classifier and saves

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

4. Go to http://0.0.0.0:3001/ to open the homepage

### Summary of the results
The main findings of the code can be found at the Medium Blog post available [here](https://medium.com/@ngdung28996/can-you-discriminate-between-different-dog-breeds-49c8f5ef1abd)

### Licensing, Authors, Acknowledgements, etc.
    - Udacity for training data and starter code for the web app. 
    # GET_PASSES_THIS_REPO_UDACITY_PLEASE
