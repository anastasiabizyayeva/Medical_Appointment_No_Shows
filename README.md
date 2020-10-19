# Medical Appointment No-Shows: Project Overview

* Created a model that estimates whether an individual will be a no-show to a medical appointment (ROC AUC = 0.933) to help medical professionals and their administrative staff determine which appointments should be targeted in retention efforts. 
* Used a dataset provided by Aquarela Advanced Analytics consisting of over 110k medical appointments made in Brazil
* Engingeered features from the tables provided, including the scheduled day, appointment day, scheduled time, time between scheduling and appointments, day of the week of the appointment, number of appointments per patient, and number of no-shows. 
* Optimized Logistic, KNN, and Random Forest Classifiers using GridsearchCV to reach the best model. Final model performance was:
 * ROC AUC: 0.933
 * Recall: 0.740
 * Accuracy Score: 0.877

## Code and Resources Used

**Python Version:** 3.8

**Packages:** pandas, numpy, sklearn, matplotlib, seaborn, counter, imblearn

**Project Structure:** https://www.youtube.com/watch?v=MpF9HENQjDo&list=PL2zq7klxX5ASFejJj80ob9ZAnBHdz5O1t

**SMOTE for Imbalanced Classification with Python:** https://machinelearningmastery.com/smote-oversampling-for-imbalanced-classification/

**Threshold-Moving for Imbalanced Classification:** https://machinelearningmastery.com/threshold-moving-for-imbalanced-classification/#:~:text=The%20decision%20for%20converting%20a,in%20the%20range%20between%200

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
* Got the day of the week of appointments 
* Got the number of appointments scheduled by each patient 
* Got the number of no-shows by patient
*	Dropped data where the patient's age was less than 0
*	Dropped data where the appointment was scheduled after it took place 

## EDA

I looked at the distributions of the data and the value counts for the various variables. Below are a few highlights from the Jupyter Notebook. In particular, the heatmap shows a strong correlation between no-shows and two variables: the time between scheduling/ appointment date, and whether the patient received a text about their appointment. 

|![alt text](https://github.com/anastasiabizyayeva/Medical_Appointment_No_Shows/blob/master/images/corr-heatmap.png "Heatmap of Variable Correlation")|![alt text](https://github.com/anastasiabizyayeva/Medical_Appointment_No_Shows/blob/master/images/Sched-to-app-time.png "Scheduling Outliers")|![alt text](https://github.com/anastasiabizyayeva/Medical_Appointment_No_Shows/blob/master/images/Neighbourhoods.png "Dataset Neighbourhood Breakdown")|
| ------------- |:-------------:| -----:|

## Model Building

During the EDA portion of the exploration, I discovered a class imbalance in our dataset, which needed to be fixed before models could be trained. I first split the data into training and test sets, then balanced the classes with a combination of SMOTE and undersampling so that the ratio of positive to negative cases was 1:2 rather than 1:4. Results below:

|![alt text](https://github.com/anastasiabizyayeva/Medical_Appointment_No_Shows/blob/master/images/imbalanced_class.png "Imbalanced Classes")| ![alt text](https://github.com/anastasiabizyayeva/Medical_Appointment_No_Shows/blob/master/images/balanced_class.png "Balanced Classes")| 
| ------------- |:-------------:|

Once this was complete I split the training data into training and validation sets so I could test a few different models. I evaluated the models based on ROC as well as recall. Given that our ultimate goal is to reduce no-shows, I wanted to focus on optimizing recall more so than precision. After all, if we classify a few people who are going to show up to their appointments as no-showers (the false positives), they might receive an extra text or call, but ultimately still be unaffected. That said, if we have a high false negative rate, we'll be missing opportunities to intervene in appointments that would otherwise be missed.

The models that I explored are:
*	**Logistic Regression** – Baseline for the model
* **K-Nearest Neighbours** - Effective when the number of samples is large (in this case, 110k), and interesting to experiment with because of the number of parameters that can be tuned
*	**Random Forest** – With the sparsity of the data, thought the sub-sampling would be effective and would control for overfitting

The ROC and Recall AUC curves are found below:

| ROC AUC Curve| Recall AUC Curve|
| ------------- |:-------------:|
|![alt text](https://github.com/anastasiabizyayeva/Medical_Appointment_No_Shows/blob/master/images/ROC_AUC.png "ROC AUC")| ![alt text](https://github.com/anastasiabizyayeva/Medical_Appointment_No_Shows/blob/master/images/recall.png "Recall AUC")|

From these curves, it was clear that the Random Forest Classifier was the best model for our exploration. I then manually tuned four hyperparameters with the following results:

|n_estimators (best = 64)| max_depth (best = 8)|min_samples_split (best = 0.1)|min_samples_leaf (best = 67)|
| ------------- |:-------------:| -----:|-----:|
|![alt text](https://github.com/anastasiabizyayeva/Medical_Appointment_No_Shows/blob/master/images/n_estimators.png "n_estimators")|![alt text](https://github.com/anastasiabizyayeva/Medical_Appointment_No_Shows/blob/master/images/max_depth.png "max_depth")|![alt text](https://github.com/anastasiabizyayeva/Medical_Appointment_No_Shows/blob/master/images/min_sample_splits.png "min_samples_split")|$![alt text](https://github.com/anastasiabizyayeva/Medical_Appointment_No_Shows/blob/master/images/min_samples_leaf.png "min_samples_leaf")|

This was followed by a GridSearch, which ultimately provided the best hyperparameters for our RF model as a result of being able to consider all permutations of the hyperparameter lists I passed, though the computation took about 3 hours. 

## Model Performance

The strongest model was the GridSearch hyperparameter-tuned RF Classifier. Once this was established, the training and validation data was recombined and the model was retrained on this new dataset. Its performance was evaluated against the holdout test set. 

The RF Classifier's performance is as follows:

 * ROC AUC: 0.933
 * Recall: 0.740
 * Accuracy Score: 0.877

However, what is most exciting for our purposes is that the confusion matrix produces only 355 false negatives, which is only 9% of our positive class. This is great because any interventions will target the majority of the 'no-show' category, though the trade-off is that we have 2,406 false positive cases. 

| Confusion Matrix for Random Forest Classifier| 
| ------------- |
| ![alt text](https://github.com/anastasiabizyayeva/Medical_Appointment_No_Shows/blob/master/images/final_model_heatmap.png "RF Confusion Matrix")   | 
