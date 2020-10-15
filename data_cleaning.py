# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 14:23:32 2020
@author: Anastasia
"""
import pandas as pd 
from pytz import timezone
import calendar

# Read in raw data file 

file = 'raw_data.csv'
df = pd.read_csv(file)

# Explore summary stats

print(df.describe())
print(df.isnull().sum())
print(df.columns)
print(df.dtypes)

# Convert scheduled day to datetime object

dt_sched = pd.to_datetime(df['ScheduledDay'], utc = True, infer_datetime_format=True)

# Convert scheduled time to Brasilia Standard Time (GMT -3) (BRT)

sched_brt = dt_sched.apply(lambda x: x.astimezone(timezone('Brazil/East')))

# Parse the time from the date in 'scheduled day'

sched_t = sched_brt.apply(lambda x: x.time())

# Add sched_brt, and sched_t columns to df & convert 'appointmentday' to a datetime object

df['ScheduledDay'] = sched_brt.apply(lambda x: x.date())
df['ScheduledTime'] = sched_t
df['AppointmentDay'] = pd.to_datetime(df['AppointmentDay']).apply(lambda x: x.date())

# Convert 'no-show' column from string to int 

df['No-show'] = df['No-show'].apply(lambda x: 1 if x == 'No' else 0)

# Filter out rows where age is less than 0

df = df[df['Age'] > 0]

# Handicap should be a yes or a no (1 or 0) so changing all values higher than 1 to 1

df['Handcap'] = df['Handcap'].apply(lambda x: 0 if x == 0 else 1)

# Subtract appointment day from scheduled day to see how many days have elapsed 

df['Sched_to_App_Time'] = (df['AppointmentDay'] - df['ScheduledDay']).apply(lambda x: x.days)

# Filter out appointments that are scheduled after they've already taken place 

df = df[df['Sched_to_App_Time'] >= 0]

# See what day of the week the appointment is taking place

df['App_DoW'] = df['ScheduledDay'].apply(lambda x: calendar.day_name[x.weekday()])

# See if there are any patients with multiple appointments

df['Num_Apps'] = df.groupby('PatientId')['ScheduledTime'].transform('count')

# See if there are any patients with multiple no-shows 

df['Num_NS'] = df.groupby('PatientId')['No-show'].transform('sum')

# Count the number of conditions a patient has (Hypertension, diabetes, alcoholism, handicapped)

# Export cleaned data 

df.to_csv('cleaned_data.csv')