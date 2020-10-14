# Medical Appointment No-Shows: Project Overview
* Created a model that estimates whether an individual will be a no-show to a medical appointment (accuracy score ~ 80%) to help medical professionals and their administrative staff determine which appointments should be targeted in retention efforts. 
* Used a dataset provided by Aquarela Advanced Analytics consisting of over 110k medical appointments made in Brazil
* Engingeered features from the tables provided, including the number of days between the appointment being scheduled and taking place. 
* Optimized Logistic, SGD, KNN, and Random Forest Regressors using GridsearchCV to reach the best model. 

## Code and Resources Used

**Python Version:** 3.9

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

## Model Building

## Model Performance

## Productionization
