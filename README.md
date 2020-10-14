# Medical Appointment No-Shows: Project Overview
* Created a model that estimates whether an individual will be a no-show to a medical appointment (accuracy score ~ 80%) to help medical professionals and their administrative staff determine which appointments should be targeted in retention efforts. 
* Used a dataset provided by Aquarela Advanced Analytics consisting of over 110k medical appointments made in Brazil
* Engingeered features from the tables provided, including the number of days between the appointment being scheduled and taking place. 
* Optimized Logistic, SGD, KNN, and Random Forest Regressors using GridsearchCV to reach the best model. 

## Code and Resources Used

**Python Version:** 3.8
**Packages:** pandas, numpy, sklearn, matplotlib, seaborn 
**Project Structure:** https://www.youtube.com/watch?v=MpF9HENQjDo&list=PL2zq7klxX5ASFejJj80ob9ZAnBHdz5O1t
**Confusion Matrix Visualisation:** https://medium.com/@dtuk81/confusion-matrix-visualization-fc31e3f30fea
**Information About 'Bolso Familia':** https://en.wikipedia.org/wiki/Bolsa_Fam%C3%ADlia

## Dataset

The dataset used for this project can be found at https://www.kaggle.com/joniarroba/noshowappointments. It initially consisted of the following variables:


* 01 - PatientId
  * Identification of a patient
* 02 - AppointmentID
  * Identification of each appointment
* 03 - Gender
  * Male or Female
* 04 - DataMarcacaoConsulta
  * The day of the actual appointment
* 05 - DataAgendamento
  * The day someone called or registered the appointment
* 06 - Age
  * Age of the patient
* 07 - Neighbourhood
  * Where the appointment takes place
* 08 - Scholarship
  * True of False - see 'Bolso Familia' in the 'Resources' section for more information
* 09 - Hipertension
  * True or False
* 10 - Diabetes
  * True or False
* Alcoholism
  * True or False
* Handcap
  * True or False
* SMS_received
  * True if 1 or more messages sent to the patient, false otherwise
* No-show
  * True or False

## Data Cleaning

The dataset was largely complete, and had no NaN values - this meant the majority of adjustments were made to extract new variables for EDA and to ensure the variables could be used in our models.

*	Converted columns to datetime objects where appropriate
*	Ensured UTC was converted to BRT
*	Parsed apart dates and times for further analysis 
*	Found the difference between the scheduling date of an appointment and the date it took place  
*	Converted the target variable from strings to ints 
*	Dropped data where the patient's age was less than 0
*	Dropped data where the appointment was scheduled after it took place 

## EDA

I looked at the distributions of the data and the value counts for the various variables. Below are a few highlights from the Jupyter Notebook. In particular, the heatmap shows a strong correlation between no-shows and two variables: the time between scheduling/ appointment date, and whether the patient received a text about their appointment. 

![alt text](https://github.com/anastasiabizyayeva/Medical_Appointment_No_Shows/blob/master/images/heatmap.JPG "Heatmap of Variable Correlation")
![alt text](https://github.com/anastasiabizyayeva/Medical_Appointment_No_Shows/blob/master/images/gender.JPG "Dataset Gender Breakdown")
![alt text](https://github.com/anastasiabizyayeva/Medical_Appointment_No_Shows/blob/master/images/neighbourhoods.JPG "Dataset Neighbourhood Breakdown")

## Model Building

I started by transforming the categorical variables into dummy variables and concatenating them to the remaining numeric variables. I then split the data into train and tests sets with a test size of 20%.   

I tried four different models and evaluated them using their accuracy score. I chose this variable because it's a straitforward and intuitive figure for judging the success of one's model, and works especially well for binary classification problems.

The four models I explored are:
*	**Logistic Regression** – Baseline for the model
*	**Stochastic Gradient Descent** – Effective when the number of samples is large (in this case, 110k)
* **K-Nearest Neighbours** - Also effective for a large dataset, and interesting to experiment with because of the number of parameters that can be tuned
*	**Random Forest** – With the sparsity of the data, thought the sub-sampling would be effective and would control for overfitting

## Model performance

The strongest model was the scaled and hyperparameter-tuned SGD classifier. All model accuracy scores below (in descending order):

*	**Stochastic Gradient Descent:** accuracy score = 80.01%
*	**Logistic Regression:** accuracy score = 79.66%
* **K-Nearest Neighbours:** accuracy score = 78.96%
*	**Random Forest:** accuracy score = 77.05%

| Confusion Matrix for SGD Classifier| 
| ------------- |
| ![alt text](https://github.com/anastasiabizyayeva/Medical_Appointment_No_Shows/blob/master/images/GSSGD%20heatmap.png "SGD Confusion Matrix")   | 
