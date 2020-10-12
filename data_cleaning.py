# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 14:23:32 2020

@author: Anastasia
"""
import pandas as pd 
from pytz import timezone

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

print(df.dtypes)

# Subtract appointment day from scheduled day to see how many days have elapsed 

df['Sched_to_App_Time'] = (df['AppointmentDay'] - df['ScheduledDay']).apply(lambda x: x.days)

df.to_csv('cleaned_data.csv')