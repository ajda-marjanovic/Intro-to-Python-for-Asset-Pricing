# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 15:15:14 2023

@author: ajda
"""

# import packages
import numpy as np
import pandas as pd


#%% Lists, DataFrames, Pandas

# make two lists of ages and names
age = [22,23,26,24]
name = ['John', 'Susan', 'Anne', 'Tim']

# combine them in a dataframe of students
df = pd.DataFrame(age, columns = ['Age'], index=name)

df.index

df.columns

# visualize the row related to Anne
df.loc['Anne']

# if we want more than 1 element, we use another pair of square brackets
df.loc[['Anne', 'Susan']]

# if we want to locate by index (get the first row of data)
# NB: Python starts indexing at 0
df.iloc[0]

# select rows 1 and 2
df.iloc[1:3]

# let's do some computations on the dataframe
df.sum()

df.mean()

df2 = df.apply(lambda x: x+2)

#%% Functions

# compute students' age once they complete the master's program (in two years)

# first using apply function
def add_two_years(x):
    return x+2

df.apply(add_two_years)

# then defining our own function, which is more flexible (has an additional agrument
# - how many years we want to add)
def add_years(x, y):
    return x + y

add_years(df['Age'], 2)

#%% Numeric variables, NumPy, summary statistics

#create a column of grades as draws from a uniform r.v.
np.random.seed(4)

# four draws between 25 and 30
random_grades = np.random.uniform(25, 30, 4)

# round to one decimal place using list comprehension
rounded_grades = [round(x,1) for x in random_grades]

# add as a new column named 'Grade'
df['Grade'] = rounded_grades

# add another column with strings
df['Program'] = pd.DataFrame(['Economics', 'Economics', 'Finance', 'Finance'],
                             index = ['Tim', 'John', 'Susan', 'Anne'])

# create a new dataframe with new student information
new_student = pd.DataFrame({'Age':26, 'Grade': 28.4, 'Program': 'Economics'}, index=['Martha'])

# concatenate the two dataframes
df_new = pd.concat([df, new_student])

# some summary statistics of the two columns
df_new[['Age', 'Grade']].mean()
df_new[['Age', 'Grade']].std()
df_new[['Age', 'Grade']].max()
df_new[['Age', 'Grade']].min()

# check data type of the columns
df_new.dtypes

# let's create a new 12x3 matrix with draws from a standard normal distribution

rand_numbers = np.random.standard_normal((12,3))

rand_numbers.round(4)

df = pd.DataFrame(rand_numbers)

# rename columns
df.columns = [['A', 'B', 'C']]

df['C'].loc[4]

# add a time dimension
dates = pd.date_range('2022-01-01', periods = 12, freq = 'MS')

# rename index
df.index = dates

# visualize first 3 and last 3 rows of the dataframe
df.head(3)
df.tail(3)

df.sum() # sum across columns
df.sum(axis=1) # sum across rows

df.mean()
df.mean(axis=1)

df.cumsum()

# summary statistics
df.describe()

# some computations using numpy
np.square(df)
np.sqrt(np.abs(df))
np.sqrt(np.abs(df[['A', 'C']])).mean()
