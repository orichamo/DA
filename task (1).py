# Coded by Ori Chamo
import numpy as np
import pandas as pd
import scipy as sp

SCALE = 1000

# Computation aids
def exceeding_from_c0_wrt_median(number): # A function for calculating the score for each country
    return ((number - c0) - median)
def space_separated_int_to_int(space_separated_int):
    return int(space_separated_int.replace(' ', '')) # space_separated_int is a string

# Read the .csv file into a Pandas dataframe
df = pd.read_csv('data_task.csv')

# Extract the NO2 column
df_NO2_annual_mean_column = df['NO2'][2:].astype(float)
M = df_NO2_annual_mean_column.median() # Find median
c0 = 10 # Set c0 before calculating score
median = M # Set the median before calculating score
S1 = df_NO2_annual_mean_column.map(exceeding_from_c0_wrt_median)

# Extract the PM2.5 column (code duplication...)
df_PM2_annual_mean_column = df['PM2.5'][2:].astype(float) 
M_PM2 = df_PM2_annual_mean_column.median()
c0 = 2.5
median = M_PM2
S2 = df_PM2_annual_mean_column.map(exceeding_from_c0_wrt_median)

# Extract more columns
# Note: 'Unnamed: 7' is the column of premature deaths attributed to NO2 with C0 = 10
#       'Unnamed: 4' is the column of premature deaths attributed to PM2.5 with C0 = 2.5
#       'Unnamed: 1' is the column of countries population
df_NO2_deaths_column = df['Unnamed: 7'][2:].map(space_separated_int_to_int) # Convert the strings to integers
df_PM2_deaths_column = df['Unnamed: 4'][2:].map(space_separated_int_to_int)
df_country_population = df['Unnamed: 1'][2:].map(space_separated_int_to_int)

# Normalize deaths column by population size
S3 = df_NO2_deaths_column.divide(SCALE*df_country_population) 
S4 = df_PM2_deaths_column.divide(SCALE*df_country_population)

# Calculate correlations between S1,S2
annual_mean_correlation_pearson = S2.corr(S1, method='pearson')
annual_mean_correlation_spearman = S2.corr(S1, method='spearman')

# Calculate correlations between S3,S4
premature_deaths_correlation_pearson = S4.corr(S3, method='pearson')
premature_deaths_correlation_spearman = S4.corr(S3, method='spearman')

# Calculate correlation between annual mean and premature deaths in every country
premature_death_wrt_annual_mean_correlation_NO2 = S3.corr(S1, method='pearson')
premature_death_wrt_annual_mean_correlation_PM2 = S4.corr(S2, method='pearson')

# Print correlations
print("The Pearson correlation between the NO2 and PM2.5 annual means is:", annual_mean_correlation_pearson)
print("The Spearman correlation between the NO2 and PM2.5 annual means is:", annual_mean_correlation_spearman)
print("The Pearson correlation between the NO2 and PM2.5 number of premature deaths is:", premature_deaths_correlation_pearson)
print("The Spearman correlation between the NO2 and PM2.5 number of premature deaths is:", premature_deaths_correlation_spearman)
print("The Pearson correlation between the NO2 annual means and premature deaths is:", premature_death_wrt_annual_mean_correlation_NO2)
print("The Pearson correlation between the PM2.5 annual means and premature deaths is:", premature_death_wrt_annual_mean_correlation_PM2)

    