#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 22:23:24 2023

@author: ptp.18
"""
# %% Import modules
# data analysis modules
from statsmodels.iolib.summary2 import summary_col
import statsmodels.formula.api as smf
import pandas as pd
# import statsmodels as sm
import numpy as np

# data visualisation modules
import seaborn as sns
import matplotlib.pyplot as plt

# %% House - Simple linear regression model: log(price) = b0 + b1 Distance + u

# Load Melbourne housing price data
file_path = '/Users/ptp.18/Desktop/Python/Melbourne_housing_FULL.csv'
melb_df = pd.read_csv(file_path)
print(melb_df.info())

# compute log(price)
melb_df['logPrice'] = np.log(melb_df['Price'])

# we want to model only house (i.e. exclude units and Townhouses)
house_df = melb_df[melb_df.Type == 'h']
print(f"Average house price = AU$ {house_df.Price.mean():.0f}")
print(f"Average house distance = {house_df.Distance.mean():.1f} km2")

# Visualisation using seaborn lmplot
gunit = sns.lmplot(data=house_df, x='Distance', y='logPrice', fit_reg=False)
gunit.set(title='Melbourne House Market - Distance')

# %% Exclud a few outliers with distance >=20
housex_df = house_df[house_df.Distance < 20]
print(
    f"Average house price (exc. Distance>=20) = AU$ {housex_df.Price.mean():.0f}")
print("Average building area (exc. Distance >=20)" +
      f" = {housex_df.Distance.mean():.1f} km2")

# Visualise the data without the outliers
gunitx = sns.lmplot(data=housex_df, x='Distance', y='logPrice', fit_reg=False)
gunitx.set(title='Melbourne House Market (Exc. Distance >=20)')

# Now the data look more reasonable after we dropped the outliers, we proceed with estimating the hedonic model

# %% Use the formula approach of statsmodel to estimate

# import the formula module of statsmodels

# specify the formula (i.e. the model)
housexreg = smf.ols('logPrice ~ Distance', data=housex_df)

# now, fit (ie. estimate) the model
housexresult = housexreg.fit()

print(housexresult.summary())
print(f"b1_hat = {housexresult.params['Distance']}")
print(f"SE(b1_hat) = {housexresult.bse['Distance']}")


# %% Plotting predicted (yhat) vs actual (y)

# Drop missing observations from whole sample
df_plot = housex_df.dropna(subset=['logPrice', 'Distance'])

# Plot predicted values
fig, ax = plt.subplots()
ax.scatter(df_plot['Distance'], df_plot['logPrice'],
           alpha=0.5, label='observed')
ax.scatter(df_plot['Distance'], housexresult.predict(),
           alpha=0.5, label='predicted')
ax.set_xlabel('Distance (km2)')
ax.set_ylabel('Log(Price)')
ax.set_title(
    'Predicted vs actual values for simple OLS regression - House & Distance')
ax.legend()

# %% Comparing estimation with and without outliers

# specify the formula (i.e. the model) with outliers
housereg = smf.ols('logPrice ~ Distance', data=house_df)

# now, fit (ie. estimate) the model
houseresult = housereg.fit()

# print(houseresult.summary())
# print(f"b1_hat = {unitresult.params['Distance']}")
# print(f"SE(b1_hat) = {unitresult.bse['Distance']}")


info_dict = {'R-squared': lambda x: "{:.2f}".format(x.rsquared),
             'No. observations': lambda x: "{0:d}".format(int(x.nobs))}

results_compare = summary_col(results=[housexresult, houseresult],
                              float_format='%0.4f',
                              stars=True,
                              model_names=['Without outliers',
                                           'With outliers'],
                              info_dict=info_dict,
                              regressor_order=['Distance',
                                               'const'])

results_compare.add_title('Table - OLS Regressions - House & Distance')

print(results_compare)


# %% Simple linear regression model: log(price) = b0 + b1 Rooms + u

house_df = melb_df[melb_df.Type == 'h']
print(f"Average house price = AU$ {house_df.Price.mean():.0f}")
print(f"Average house Rooms = {house_df.Rooms.mean():.1f} km2")

# Visualisation using seaborn lmplot
gunit = sns.lmplot(data=house_df, x='Rooms', y='logPrice', fit_reg=False)
gunit.set(title='Melbourne House Market - Rooms')

# %% Exclude a few outliers with Rooms >=6
housex_df = house_df[house_df.Rooms < 6]
print(
    f"Average house price (exc. Rooms>=6) = AU$ {housex_df.Price.mean():.0f}")
print("Average building area (exc. Rooms >=6)" +
      f" = {housex_df.Rooms.mean():.1f}")

# Visualise the data without the outliers
gunitx = sns.lmplot(data=housex_df, x='Rooms', y='logPrice', fit_reg=False)
gunitx.set(title='Melbourne House Market (Exc. Rooms >=6)')

# Now the data look more reasonable after we dropped the outliers, we proceed
# with estimating the hedonic model

# %% Use the formula approach of statsmodel to estimate

# import the formula module of statsmodels

# specify the formula (i.e. the model)
housexreg = smf.ols('logPrice ~ Rooms', data=housex_df)

# now, fit (ie. estimate) the model
housexresult = housexreg.fit()

print(housexresult.summary())
print(f"b1_hat = {housexresult.params['Rooms']}")
print(f"SE(b1_hat) = {housexresult.bse['Rooms']}")


# %% Plotting predicted (yhat) vs actual (y)

# Drop missing observations from whole sample
df_plot = housex_df.dropna(subset=['logPrice', 'Rooms'])

# Plot predicted values
fig, ax = plt.subplots()
ax.scatter(df_plot['Rooms'], df_plot['logPrice'], alpha=0.5, label='observed')
ax.scatter(df_plot['Rooms'], housexresult.predict(),
           alpha=0.5, label='predicted')
ax.set_xlabel('Rooms')
ax.set_ylabel('Log(Price)')
ax.set_title('Predicted vs actual values for simple OLS regression')
ax.legend()

# %% Comparing estimation with and without outliers

# specify the formula (i.e. the model) with outliers
housereg = smf.ols('logPrice ~ Rooms', data=house_df)

# now, fit (ie. estimate) the model
houseresult = housereg.fit()

# print(unitresult.summary())
# print(f"b1_hat = {unitresult.params['BuildingArea']}")
# print(f"SE(b1_hat) = {unitresult.bse['BuildingArea']}")


info_dict = {'R-squared': lambda x: "{:.2f}".format(x.rsquared),
             'No. observations': lambda x: "{0:d}".format(int(x.nobs))}

results_compare = summary_col(results=[housexresult, houseresult],
                              float_format='%0.4f',
                              stars=True,
                              model_names=['Without outliers',
                                           'With outliers'],
                              info_dict=info_dict,
                              regressor_order=['Rooms',
                                               'const'])

results_compare.add_title('Table - OLS Regressions - House & Rooms')

print(results_compare)


# %% Simple linear regression model: log(price) = b0 + b1 Bedroom2 + u

house_df = melb_df[melb_df.Type == 'h']
print(f"Average house price = AU$ {house_df.Price.mean():.0f}")
print(f"Average house Bedroom2 = {house_df.Bedroom2.mean():.1f} km2")

# Visualisation using seaborn lmplot
gunit = sns.lmplot(data=house_df, x='Bedroom2', y='logPrice', fit_reg=False)
gunit.set(title='Melbourne House Market - Bedroom2')

# %% Exclude a few outliers with Bedroom2 >=6
housex_df = house_df[house_df.Bedroom2 < 6]
print(
    f"Average house price (exc. Bedroom2>=6) = AU$ {housex_df.Price.mean():.0f}")
print("Average building area (exc. Bedroom2 >=6)" +
      f" = {housex_df.Bedroom2.mean():.1f}")

# Visualise the data without the outliers
gunitx = sns.lmplot(data=housex_df, x='Bedroom2', y='logPrice', fit_reg=False)
gunitx.set(title='Melbourne House Market (Exc. Bedroom2 >=6)')

# Now the data look more reasonable after we dropped the outliers, we proceed
# with estimating the hedonic model

# %% Use the formula approach of statsmodel to estimate

# import the formula module of statsmodels

# specify the formula (i.e. the model)
housexreg = smf.ols('logPrice ~ Bedroom2', data=housex_df)

# now, fit (ie. estimate) the model
housexresult = housexreg.fit()

print(housexresult.summary())
print(f"b1_hat = {housexresult.params['Bedroom2']}")
print(f"SE(b1_hat) = {housexresult.bse['Bedroom2']}")


# %% Plotting predicted (yhat) vs actual (y)

# Drop missing observations from whole sample
df_plot = housex_df.dropna(subset=['logPrice', 'Bedroom2'])

# Plot predicted values
fig, ax = plt.subplots()
ax.scatter(df_plot['Bedroom2'], df_plot['logPrice'],
           alpha=0.5, label='observed')
ax.scatter(df_plot['Bedroom2'], housexresult.predict(),
           alpha=0.5, label='predicted')
ax.set_xlabel('Bedroom2')
ax.set_ylabel('Log(Price)')
ax.set_title('Predicted vs actual values for simple OLS regression')
ax.legend()

# %% Comparing estimation with and without outliers

# specify the formula (i.e. the model) with outliers
housereg = smf.ols('logPrice ~ Bedroom2', data=house_df)

# now, fit (ie. estimate) the model
houseresult = housereg.fit()

# print(unitresult.summary())
# print(f"b1_hat = {unitresult.params['BuildingArea']}")
# print(f"SE(b1_hat) = {unitresult.bse['BuildingArea']}")


info_dict = {'R-squared': lambda x: "{:.2f}".format(x.rsquared),
             'No. observations': lambda x: "{0:d}".format(int(x.nobs))}

results_compare = summary_col(results=[housexresult, houseresult],
                              float_format='%0.4f',
                              stars=True,
                              model_names=['Without outliers',
                                           'With outliers'],
                              info_dict=info_dict,
                              regressor_order=['Bedroom2',
                                               'const'])

results_compare.add_title('Table - OLS Regressions - House & Bedroom2')

print(results_compare)


# %% Simple linear regression model: log(price) = b0 + b1 Bathroom + u

house_df = melb_df[melb_df.Type == 'h']
print(f"Average house price = AU$ {house_df.Price.mean():.0f}")
print(f"Average house Bathroom = {house_df.Bathroom.mean():.1f} km2")

# Visualisation using seaborn lmplot
gunit = sns.lmplot(data=house_df, x='Bathroom', y='logPrice', fit_reg=False)
gunit.set(title='Melbourne House Market - Bathroom')

# %% Exclude a few outliers with Bathroom >=6
housex_df = house_df[house_df.Bathroom < 6]
print(
    f"Average house price (exc. Bathroom>=6) = AU$ {housex_df.Price.mean():.0f}")
print("Average building area (exc. Bathroom >=6)" +
      f" = {housex_df.Bathroom.mean():.1f}")

# Visualise the data without the outliers
gunitx = sns.lmplot(data=housex_df, x='Bathroom', y='logPrice', fit_reg=False)
gunitx.set(title='Melbourne House Market (Exc. Bathroom >=6)')

# Now the data look more reasonable after we dropped the outliers, we proceed
# with estimating the hedonic model

# %% Use the formula approach of statsmodel to estimate

# import the formula module of statsmodels

# specify the formula (i.e. the model)
housexreg = smf.ols('logPrice ~ Bathroom', data=housex_df)

# now, fit (ie. estimate) the model
housexresult = housexreg.fit()

print(housexresult.summary())
print(f"b1_hat = {housexresult.params['Bathroom']}")
print(f"SE(b1_hat) = {housexresult.bse['Bathroom']}")


# %% Plotting predicted (yhat) vs actual (y)

# Drop missing observations from whole sample
df_plot = housex_df.dropna(subset=['logPrice', 'Bathroom'])

# Plot predicted values
fig, ax = plt.subplots()
ax.scatter(df_plot['Bathroom'], df_plot['logPrice'],
           alpha=0.5, label='observed')
ax.scatter(df_plot['Bathroom'], housexresult.predict(),
           alpha=0.5, label='predicted')
ax.set_xlabel('Bathroom')
ax.set_ylabel('Log(Price)')
ax.set_title('Predicted vs actual values for simple OLS regression')
ax.legend()

# %% Comparing estimation with and without outliers

# specify the formula (i.e. the model) with outliers
housereg = smf.ols('logPrice ~ Bathroom', data=house_df)

# now, fit (ie. estimate) the model
houseresult = housereg.fit()

# print(unitresult.summary())
# print(f"b1_hat = {unitresult.params['BuildingArea']}")
# print(f"SE(b1_hat) = {unitresult.bse['BuildingArea']}")


info_dict = {'R-squared': lambda x: "{:.2f}".format(x.rsquared),
             'No. observations': lambda x: "{0:d}".format(int(x.nobs))}

results_compare = summary_col(results=[housexresult, houseresult],
                              float_format='%0.4f',
                              stars=True,
                              model_names=['Without outliers',
                                           'With outliers'],
                              info_dict=info_dict,
                              regressor_order=['Bathroom',
                                               'const'])

results_compare.add_title('Table - OLS Regressions - House & Bathroom')

print(results_compare)


# %% Simple linear regression model: log(price) = b0 + b1 Car + u

house_df = melb_df[melb_df.Type == 'h']
print(f"Average house price = AU$ {house_df.Price.mean():.0f}")
print(f"Average house Car = {house_df.Car.mean():.1f} km2")

# Visualisation using seaborn lmplot
gunit = sns.lmplot(data=house_df, x='Car', y='logPrice', fit_reg=False)
gunit.set(title='Melbourne House Market - Car')

# %% Exclude a few outliers with Car >=6
housex_df = house_df[house_df.Car < 6]
print(f"Average house price (exc. Car>=6) = AU$ {housex_df.Price.mean():.0f}")
print("Average building area (exc. Car >=6)" +
      f" = {housex_df.Car.mean():.1f}")

# Visualise the data without the outliers
gunitx = sns.lmplot(data=housex_df, x='Car', y='logPrice', fit_reg=False)
gunitx.set(title='Melbourne House Market (Exc. Car >=6)')

# Now the data look more reasonable after we dropped the outliers, we proceed
# with estimating the hedonic model

# %% Use the formula approach of statsmodel to estimate

# import the formula module of statsmodels

# specify the formula (i.e. the model)
housexreg = smf.ols('logPrice ~ Car', data=housex_df)

# now, fit (ie. estimate) the model
housexresult = housexreg.fit()

print(housexresult.summary())
print(f"b1_hat = {housexresult.params['Car']}")
print(f"SE(b1_hat) = {housexresult.bse['Car']}")


# %% Plotting predicted (yhat) vs actual (y)

# Drop missing observations from whole sample
df_plot = housex_df.dropna(subset=['logPrice', 'Car'])

# Plot predicted values
fig, ax = plt.subplots()
ax.scatter(df_plot['Car'], df_plot['logPrice'], alpha=0.5, label='observed')
ax.scatter(df_plot['Car'], housexresult.predict(),
           alpha=0.5, label='predicted')
ax.set_xlabel('Car')
ax.set_ylabel('Log(Price)')
ax.set_title('Predicted vs actual values for simple OLS regression')
ax.legend()

# %% Comparing estimation with and without outliers

# specify the formula (i.e. the model) with outliers
housereg = smf.ols('logPrice ~ Car', data=house_df)

# now, fit (ie. estimate) the model
houseresult = housereg.fit()

# print(unitresult.summary())
# print(f"b1_hat = {unitresult.params['BuildingArea']}")
# print(f"SE(b1_hat) = {unitresult.bse['BuildingArea']}")


info_dict = {'R-squared': lambda x: "{:.2f}".format(x.rsquared),
             'No. observations': lambda x: "{0:d}".format(int(x.nobs))}

results_compare = summary_col(results=[housexresult, houseresult],
                              float_format='%0.4f',
                              stars=True,
                              model_names=['Without outliers',
                                           'With outliers'],
                              info_dict=info_dict,
                              regressor_order=['Car',
                                               'const'])

results_compare.add_title('Table - OLS Regressions - House & Car')

print(results_compare)


# %% Simple linear regression model: log(price) = b0 + b1 Landsize + u

house_df = melb_df[melb_df.Type == 'h']
print(f"Average house price = AU$ {house_df.Price.mean():.0f}")
print(f"Average house Landsize = {house_df.Landsize.mean():.1f} km2")

# Visualisation using seaborn lmplot
gunit = sns.lmplot(data=house_df, x='Landsize', y='logPrice', fit_reg=False)
gunit.set(title='Melbourne House Market - Landsize')

# %% Exclude a few outliers with Landsize >=5000
housex_df = house_df[house_df.Landsize < 5000]
print(
    f"Average house price (exc. Landsize>=5000) = AU$ {housex_df.Price.mean():.0f}")
print("Average building area (exc. Landsize >=5000)" +
      f" = {housex_df.Landsize.mean():.1f}")

# Visualise the data without the outliers
gunitx = sns.lmplot(data=housex_df, x='Landsize', y='logPrice', fit_reg=False)
gunitx.set(title='Melbourne House Market (Exc. Landsize >=5000)')

# Now the data look more reasonable after we dropped the outliers, we proceed
# with estimating the hedonic model

# %% Use the formula approach of statsmodel to estimate

# import the formula module of statsmodels

# specify the formula (i.e. the model)
housexreg = smf.ols('logPrice ~ Landsize', data=housex_df)

# now, fit (ie. estimate) the model
housexresult = housexreg.fit()

print(housexresult.summary())
print(f"b1_hat = {housexresult.params['Landsize']}")
print(f"SE(b1_hat) = {housexresult.bse['Landsize']}")


# %% Plotting predicted (yhat) vs actual (y)

# Drop missing observations from whole sample
df_plot = housex_df.dropna(subset=['logPrice', 'Landsize'])

# Plot predicted values
fig, ax = plt.subplots()
ax.scatter(df_plot['Landsize'], df_plot['logPrice'],
           alpha=0.5, label='observed')
ax.scatter(df_plot['Landsize'], housexresult.predict(),
           alpha=0.5, label='predicted')
ax.set_xlabel('Landsize')
ax.set_ylabel('Log(Price)')
ax.set_title('Predicted vs actual values for simple OLS regression')
ax.legend()

# %% Comparing estimation with and without outliers

# specify the formula (i.e. the model) with outliers
housereg = smf.ols('logPrice ~ Landsize', data=house_df)

# now, fit (ie. estimate) the model
houseresult = housereg.fit()

# print(unitresult.summary())
# print(f"b1_hat = {unitresult.params['BuildingArea']}")
# print(f"SE(b1_hat) = {unitresult.bse['BuildingArea']}")


info_dict = {'R-squared': lambda x: "{:.2f}".format(x.rsquared),
             'No. observations': lambda x: "{0:d}".format(int(x.nobs))}

results_compare = summary_col(results=[housexresult, houseresult],
                              float_format='%0.4f',
                              stars=True,
                              model_names=['Without outliers',
                                           'With outliers'],
                              info_dict=info_dict,
                              regressor_order=['Landsize',
                                               'const'])

results_compare.add_title('Table - OLS Regressions - House & Landsize')

print(results_compare)


# %% Simple linear regression model: log(price) = b0 + b1 BuildingArea + u

house_df = melb_df[melb_df.Type == 'h']
print(f"Average house price = AU$ {house_df.Price.mean():.0f}")
print(f"Average house BuildingArea = {house_df.BuildingArea.mean():.1f} km2")

# Visualisation using seaborn lmplot
gunit = sns.lmplot(data=house_df, x='BuildingArea',
                   y='logPrice', fit_reg=False)
gunit.set(title='Melbourne House Market - BuildingArea')

# %% Exclude a few outliers with BuildingArea >=400
housex_df = house_df[house_df.BuildingArea < 400]
print(
    f"Average house price (exc. BuildingArea>=400) = AU$ {housex_df.Price.mean():.0f}")
print("Average building area (exc. BuildingArea >=400)" +
      f" = {housex_df.BuildingArea.mean():.1f}")

# Visualise the data without the outliers
gunitx = sns.lmplot(data=housex_df, x='BuildingArea',
                    y='logPrice', fit_reg=False)
gunitx.set(title='Melbourne House Market (Exc. BuildingArea >=400)')

# Now the data look more reasonable after we dropped the outliers, we proceed
# with estimating the hedonic model

# %% Use the formula approach of statsmodel to estimate

# import the formula module of statsmodels

# specify the formula (i.e. the model)
housexreg = smf.ols('logPrice ~ BuildingArea', data=housex_df)

# now, fit (ie. estimate) the model
housexresult = housexreg.fit()

print(housexresult.summary())
print(f"b1_hat = {housexresult.params['BuildingArea']}")
print(f"SE(b1_hat) = {housexresult.bse['BuildingArea']}")


# %% Plotting predicted (yhat) vs actual (y)

# Drop missing observations from whole sample
df_plot = housex_df.dropna(subset=['logPrice', 'BuildingArea'])

# Plot predicted values
fig, ax = plt.subplots()
ax.scatter(df_plot['BuildingArea'], df_plot['logPrice'],
           alpha=0.5, label='observed')
ax.scatter(df_plot['BuildingArea'], housexresult.predict(),
           alpha=0.5, label='predicted')
ax.set_xlabel('BuildingArea')
ax.set_ylabel('Log(Price)')
ax.set_title('Predicted vs actual values for simple OLS regression')
ax.legend()

# %% Comparing estimation with and without outliers

# specify the formula (i.e. the model) with outliers
housereg = smf.ols('logPrice ~ BuildingArea', data=house_df)

# now, fit (ie. estimate) the model
houseresult = housereg.fit()

# print(unitresult.summary())
# print(f"b1_hat = {unitresult.params['BuildingArea']}")
# print(f"SE(b1_hat) = {unitresult.bse['BuildingArea']}")


info_dict = {'R-squared': lambda x: "{:.2f}".format(x.rsquared),
             'No. observations': lambda x: "{0:d}".format(int(x.nobs))}

results_compare = summary_col(results=[housexresult, houseresult],
                              float_format='%0.4f',
                              stars=True,
                              model_names=['Without outliers',
                                           'With outliers'],
                              info_dict=info_dict,
                              regressor_order=['BuildingArea',
                                               'const'])

results_compare.add_title('Table - OLS Regressions - House & BuildingArea')

print(results_compare)


# %% Simple linear regression model: log(price) = b0 + b1 YearBuilt + u

house_df = melb_df[melb_df.Type == 'h']
print(f"Average house price = AU$ {house_df.Price.mean():.0f}")
print(f"Average house YearBuilt = {house_df.YearBuilt.mean():.1f} km2")

# Visualisation using seaborn lmplot
gunit = sns.lmplot(data=house_df, x='YearBuilt', y='logPrice', fit_reg=False)
gunit.set(title='Melbourne House Market - YearBuilt')

# %% Exclude a few outliers with YearBuilt <=1900
housex_df = house_df[house_df.YearBuilt > 1900]
print(
    f"Average house price (exc. YearBuilt>=1900) = AU$ {housex_df.Price.mean():.0f}")
print("Average building area (exc. YearBuilt >=1900)" +
      f" = {housex_df.YearBuilt.mean():.1f}")

# Visualise the data without the outliers
gunitx = sns.lmplot(data=housex_df, x='YearBuilt', y='logPrice', fit_reg=False)
gunitx.set(title='Melbourne House Market (Exc. YearBuilt >=1900)')

# Now the data look more reasonable after we dropped the outliers, we proceed
# with estimating the hedonic model

# %% Use the formula approach of statsmodel to estimate

# import the formula module of statsmodels

# specify the formula (i.e. the model)
housexreg = smf.ols('logPrice ~ YearBuilt', data=housex_df)

# now, fit (ie. estimate) the model
housexresult = housexreg.fit()

print(housexresult.summary())
print(f"b1_hat = {housexresult.params['YearBuilt']}")
print(f"SE(b1_hat) = {housexresult.bse['YearBuilt']}")


# %% Plotting predicted (yhat) vs actual (y)

# Drop missing observations from whole sample
df_plot = housex_df.dropna(subset=['logPrice', 'YearBuilt'])

# Plot predicted values
fig, ax = plt.subplots()
ax.scatter(df_plot['YearBuilt'], df_plot['logPrice'],
           alpha=0.5, label='observed')
ax.scatter(df_plot['YearBuilt'], housexresult.predict(),
           alpha=0.5, label='predicted')
ax.set_xlabel('YearBuilt')
ax.set_ylabel('Log(Price)')
ax.set_title('Predicted vs actual values for simple OLS regression')
ax.legend()

# %% Comparing estimation with and without outliers

# specify the formula (i.e. the model) with outliers
housereg = smf.ols('logPrice ~ YearBuilt', data=house_df)

# now, fit (ie. estimate) the model
houseresult = housereg.fit()

# print(unitresult.summary())
# print(f"b1_hat = {unitresult.params['YearBuilt']}")
# print(f"SE(b1_hat) = {unitresult.bse['YearBuilt']}")


info_dict = {'R-squared': lambda x: "{:.2f}".format(x.rsquared),
             'No. observations': lambda x: "{0:d}".format(int(x.nobs))}

results_compare = summary_col(results=[housexresult, houseresult],
                              float_format='%0.4f',
                              stars=True,
                              model_names=['Without outliers',
                                           'With outliers'],
                              info_dict=info_dict,
                              regressor_order=['YearBuilt',
                                               'const'])

results_compare.add_title('Table - OLS Regressions - House & BuildingArea')

print(results_compare)


# %% Townhouse - Simple linear regression model: log(price) = b0 + b1 Distance + u

# Load Melbourne housing price data
melb_df = pd.read_csv('Melbourne_housing_FULL.csv')
melb_df.info()

# compute log(price)
melb_df['logPrice'] = np.log(melb_df['Price'])

# we want to model only Townhouse (i.e. exclude units and Towntownhouses)
townhouse_df = melb_df[melb_df.Type == 't']
print(f"Average townhouse price = AU$ {townhouse_df.Price.mean():.0f}")
print(f"Average townhouse distance = {townhouse_df.Distance.mean():.1f} km2")

# Visualisation using seaborn lmplot
gunit = sns.lmplot(data=townhouse_df, x='Distance',
                   y='logPrice', fit_reg=False)
gunit.set(title='Melbourne Towntownhouse Market - Distance')

# %% Exclude a few outliers with distance >=24
townhousex_df = townhouse_df[townhouse_df.Distance < 24]
print(
    f"Average townhouse price (exc. Distance>=24) = AU$ {townhousex_df.Price.mean():.0f}")
print("Average building area (exc. Distance >=24)" +
      f" = {townhousex_df.Distance.mean():.1f} km2")

# Visualise the data without the outliers
gunitx = sns.lmplot(data=townhousex_df, x='Distance',
                    y='logPrice', fit_reg=False)
gunitx.set(title='Melbourne Townhouse Market (Exc. Distance >=24)')

# Now the data look more reasonable after we dropped the outliers, we proceed
# with estimating the hedonic model

# %% Use the formula approach of statsmodel to estimate

# import the formula module of statsmodels

# specify the formula (i.e. the model)
townhousexreg = smf.ols('logPrice ~ Distance', data=townhousex_df)

# now, fit (ie. estimate) the model
townhousexresult = townhousexreg.fit()

print(townhousexresult.summary())
print(f"b1_hat = {townhousexresult.params['Distance']}")
print(f"SE(b1_hat) = {townhousexresult.bse['Distance']}")


# %% Plotting predicted (yhat) vs actual (y)

# Drop missing observations from whole sample
df_plot = townhousex_df.dropna(subset=['logPrice', 'Distance'])

# Plot predicted values
fig, ax = plt.subplots()
ax.scatter(df_plot['Distance'], df_plot['logPrice'],
           alpha=0.5, label='observed')
ax.scatter(df_plot['Distance'], townhousexresult.predict(),
           alpha=0.5, label='predicted')
ax.set_xlabel('Distance (km2)')
ax.set_ylabel('Log(Price)')
ax.set_title(
    'Predicted vs actual values for simple OLS regression - townhouse & Distance')
ax.legend()

# %% Comparing estimation with and without outliers

# specify the formula (i.e. the model) with outliers
townhousereg = smf.ols('logPrice ~ Distance', data=townhouse_df)

# now, fit (ie. estimate) the model
townhouseresult = townhousereg.fit()

# print(townhouseresult.summary())
# print(f"b1_hat = {unitresult.params['Distance']}")
# print(f"SE(b1_hat) = {unitresult.bse['Distance']}")


info_dict = {'R-squared': lambda x: "{:.2f}".format(x.rsquared),
             'No. observations': lambda x: "{0:d}".format(int(x.nobs))}

results_compare = summary_col(results=[townhousexresult, townhouseresult],
                              float_format='%0.4f',
                              stars=True,
                              model_names=['Without outliers',
                                           'With outliers'],
                              info_dict=info_dict,
                              regressor_order=['Distance',
                                               'const'])

results_compare.add_title('Table - OLS Regressions - Townhouse & Distance')

print(results_compare)


# %% Simple linear regression model: log(price) = b0 + b1 Rooms + u

townhouse_df = melb_df[melb_df.Type == 'h']
print(f"Average townhouse price = AU$ {townhouse_df.Price.mean():.0f}")
print(f"Average townhouse Rooms = {townhouse_df.Rooms.mean():.1f} km2")

# Visualisation using seaborn lmplot
gunit = sns.lmplot(data=townhouse_df, x='Rooms', y='logPrice', fit_reg=False)
gunit.set(title='Melbourne townhouse Market - Rooms')

# %% Exclude a few outliers with Rooms >=7
townhousex_df = townhouse_df[townhouse_df.Rooms < 7]
print(
    f"Average townhouse price (exc. Rooms>=7) = AU$ {townhousex_df.Price.mean():.0f}")
print("Average building area (exc. Rooms >=7)" +
      f" = {townhousex_df.Rooms.mean():.1f}")

# Visualise the data without the outliers
gunitx = sns.lmplot(data=townhousex_df, x='Rooms', y='logPrice', fit_reg=False)
gunitx.set(title='Melbourne townhouse Market (Exc. Rooms >=7)')

# Now the data look more reasonable after we dropped the outliers, we proceed
# with estimating the hedonic model

# %% Use the formula approach of statsmodel to estimate

# import the formula module of statsmodels

# specify the formula (i.e. the model)
townhousexreg = smf.ols('logPrice ~ Rooms', data=townhousex_df)

# now, fit (ie. estimate) the model
townhousexresult = townhousexreg.fit()

print(townhousexresult.summary())
print(f"b1_hat = {townhousexresult.params['Rooms']}")
print(f"SE(b1_hat) = {townhousexresult.bse['Rooms']}")


# %% Plotting predicted (yhat) vs actual (y)

# Drop missing observations from whole sample
df_plot = townhousex_df.dropna(subset=['logPrice', 'Rooms'])

# Plot predicted values
fig, ax = plt.subplots()
ax.scatter(df_plot['Rooms'], df_plot['logPrice'], alpha=0.5, label='observed')
ax.scatter(df_plot['Rooms'], townhousexresult.predict(),
           alpha=0.5, label='predicted')
ax.set_xlabel('Rooms')
ax.set_ylabel('Log(Price)')
ax.set_title('Predicted vs actual values for simple OLS regression')
ax.legend()

# %% Comparing estimation with and without outliers

# specify the formula (i.e. the model) with outliers
townhousereg = smf.ols('logPrice ~ Rooms', data=townhouse_df)

# now, fit (ie. estimate) the model
townhouseresult = townhousereg.fit()

# print(unitresult.summary())
# print(f"b1_hat = {unitresult.params['BuildingArea']}")
# print(f"SE(b1_hat) = {unitresult.bse['BuildingArea']}")


info_dict = {'R-squared': lambda x: "{:.2f}".format(x.rsquared),
             'No. observations': lambda x: "{0:d}".format(int(x.nobs))}

results_compare = summary_col(results=[townhousexresult, townhouseresult],
                              float_format='%0.4f',
                              stars=True,
                              model_names=['Without outliers',
                                           'With outliers'],
                              info_dict=info_dict,
                              regressor_order=['Rooms',
                                               'const'])

results_compare.add_title('Table - OLS Regressions - Townhouse & Rooms')

print(results_compare)


# %% Simple linear regression model: log(price) = b0 + b1 Bedroom2 + u

townhouse_df = melb_df[melb_df.Type == 'h']
print(f"Average townhouse price = AU$ {townhouse_df.Price.mean():.0f}")
print(f"Average townhouse Bedroom2 = {townhouse_df.Bedroom2.mean():.1f} km2")

# Visualisation using seaborn lmplot
gunit = sns.lmplot(data=townhouse_df, x='Bedroom2',
                   y='logPrice', fit_reg=False)
gunit.set(title='Melbourne townhouse Market - Bedroom2')

# %% Exclude a few outliers with Bedroom2 >=7
townhousex_df = townhouse_df[townhouse_df.Bedroom2 < 7]
print(
    f"Average townhouse price (exc. Bedroom2>=7) = AU$ {townhousex_df.Price.mean():.0f}")
print("Average building area (exc. Bedroom2 >=7)" +
      f" = {townhousex_df.Bedroom2.mean():.1f}")

# Visualise the data without the outliers
gunitx = sns.lmplot(data=townhousex_df, x='Bedroom2',
                    y='logPrice', fit_reg=False)
gunitx.set(title='Melbourne townhouse Market (Exc. Bedroom2 >=7)')

# Now the data look more reasonable after we dropped the outliers, we proceed
# with estimating the hedonic model

# %% Use the formula approach of statsmodel to estimate

# import the formula module of statsmodels

# specify the formula (i.e. the model)
townhousexreg = smf.ols('logPrice ~ Bedroom2', data=townhousex_df)

# now, fit (ie. estimate) the model
townhousexresult = townhousexreg.fit()

print(townhousexresult.summary())
print(f"b1_hat = {townhousexresult.params['Bedroom2']}")
print(f"SE(b1_hat) = {townhousexresult.bse['Bedroom2']}")


# %% Plotting predicted (yhat) vs actual (y)

# Drop missing observations from whole sample
df_plot = townhousex_df.dropna(subset=['logPrice', 'Bedroom2'])

# Plot predicted values
fig, ax = plt.subplots()
ax.scatter(df_plot['Bedroom2'], df_plot['logPrice'],
           alpha=0.5, label='observed')
ax.scatter(df_plot['Bedroom2'], townhousexresult.predict(),
           alpha=0.5, label='predicted')
ax.set_xlabel('Bedroom2')
ax.set_ylabel('Log(Price)')
ax.set_title('Predicted vs actual values for simple OLS regression')
ax.legend()

# %% Comparing estimation with and without outliers

# specify the formula (i.e. the model) with outliers
townhousereg = smf.ols('logPrice ~ Bedroom2', data=townhouse_df)

# now, fit (ie. estimate) the model
townhouseresult = townhousereg.fit()

# print(unitresult.summary())
# print(f"b1_hat = {unitresult.params['BuildingArea']}")
# print(f"SE(b1_hat) = {unitresult.bse['BuildingArea']}")


info_dict = {'R-squared': lambda x: "{:.2f}".format(x.rsquared),
             'No. observations': lambda x: "{0:d}".format(int(x.nobs))}

results_compare = summary_col(results=[townhousexresult, townhouseresult],
                              float_format='%0.4f',
                              stars=True,
                              model_names=['Without outliers',
                                           'With outliers'],
                              info_dict=info_dict,
                              regressor_order=['Bedroom2',
                                               'const'])

results_compare.add_title('Table - OLS Regressions - Townhouse & Bedroom2')

print(results_compare)


# %% Simple linear regression model: log(price) = b0 + b1 Bathroom + u

townhouse_df = melb_df[melb_df.Type == 'h']
print(f"Average townhouse price = AU$ {townhouse_df.Price.mean():.0f}")
print(f"Average townhouse Bathroom = {townhouse_df.Bathroom.mean():.1f} km2")

# Visualisation using seaborn lmplot
gunit = sns.lmplot(data=townhouse_df, x='Bathroom',
                   y='logPrice', fit_reg=False)
gunit.set(title='Melbourne townhouse Market - Bathroom')

# %% Exclude a few outliers with Bathroom >=6
townhousex_df = townhouse_df[townhouse_df.Bathroom < 6]
print(
    f"Average townhouse price (exc. Bathroom>=6) = AU$ {townhousex_df.Price.mean():.0f}")
print("Average building area (exc. Bathroom >=6)" +
      f" = {townhousex_df.Bathroom.mean():.1f}")

# Visualise the data without the outliers
gunitx = sns.lmplot(data=townhousex_df, x='Bathroom',
                    y='logPrice', fit_reg=False)
gunitx.set(title='Melbourne townhouse Market (Exc. Bathroom >=6)')

# Now the data look more reasonable after we dropped the outliers, we proceed
# with estimating the hedonic model

# %% Use the formula approach of statsmodel to estimate

# import the formula module of statsmodels

# specify the formula (i.e. the model)
townhousexreg = smf.ols('logPrice ~ Bathroom', data=townhousex_df)

# now, fit (ie. estimate) the model
townhousexresult = townhousexreg.fit()

print(townhousexresult.summary())
print(f"b1_hat = {townhousexresult.params['Bathroom']}")
print(f"SE(b1_hat) = {townhousexresult.bse['Bathroom']}")


# %% Plotting predicted (yhat) vs actual (y)

# Drop missing observations from whole sample
df_plot = townhousex_df.dropna(subset=['logPrice', 'Bathroom'])

# Plot predicted values
fig, ax = plt.subplots()
ax.scatter(df_plot['Bathroom'], df_plot['logPrice'],
           alpha=0.5, label='observed')
ax.scatter(df_plot['Bathroom'], townhousexresult.predict(),
           alpha=0.5, label='predicted')
ax.set_xlabel('Bathroom')
ax.set_ylabel('Log(Price)')
ax.set_title('Predicted vs actual values for simple OLS regression')
ax.legend()

# %% Comparing estimation with and without outliers

# specify the formula (i.e. the model) with outliers
townhousereg = smf.ols('logPrice ~ Bathroom', data=townhouse_df)

# now, fit (ie. estimate) the model
townhouseresult = townhousereg.fit()

# print(unitresult.summary())
# print(f"b1_hat = {unitresult.params['BuildingArea']}")
# print(f"SE(b1_hat) = {unitresult.bse['BuildingArea']}")


info_dict = {'R-squared': lambda x: "{:.2f}".format(x.rsquared),
             'No. observations': lambda x: "{0:d}".format(int(x.nobs))}

results_compare = summary_col(results=[townhousexresult, townhouseresult],
                              float_format='%0.4f',
                              stars=True,
                              model_names=['Without outliers',
                                           'With outliers'],
                              info_dict=info_dict,
                              regressor_order=['Bathroom',
                                               'const'])

results_compare.add_title('Table - OLS Regressions - Townhouse & Bathroom')

print(results_compare)


# %% Simple linear regression model: log(price) = b0 + b1 Car + u

townhouse_df = melb_df[melb_df.Type == 'h']
print(f"Average townhouse price = AU$ {townhouse_df.Price.mean():.0f}")
print(f"Average townhouse Car = {townhouse_df.Car.mean():.1f} km2")

# Visualisation using seaborn lmplot
gunit = sns.lmplot(data=townhouse_df, x='Car', y='logPrice', fit_reg=False)
gunit.set(title='Melbourne townhouse Market - Car')

# %% Exclude a few outliers with Car >=6
townhousex_df = townhouse_df[townhouse_df.Car < 8]
print(
    f"Average townhouse price (exc. Car>=8) = AU$ {townhousex_df.Price.mean():.0f}")
print("Average building area (exc. Car >=8)" +
      f" = {townhousex_df.Car.mean():.1f}")

# Visualise the data without the outliers
gunitx = sns.lmplot(data=townhousex_df, x='Car', y='logPrice', fit_reg=False)
gunitx.set(title='Melbourne townhouse Market (Exc. Car >=8)')

# Now the data look more reasonable after we dropped the outliers, we proceed
# with estimating the hedonic model

# %% Use the formula approach of statsmodel to estimate

# import the formula module of statsmodels

# specify the formula (i.e. the model)
townhousexreg = smf.ols('logPrice ~ Car', data=townhousex_df)

# now, fit (ie. estimate) the model
townhousexresult = townhousexreg.fit()

print(townhousexresult.summary())
print(f"b1_hat = {townhousexresult.params['Car']}")
print(f"SE(b1_hat) = {townhousexresult.bse['Car']}")


# %% Plotting predicted (yhat) vs actual (y)

# Drop missing observations from whole sample
df_plot = townhousex_df.dropna(subset=['logPrice', 'Car'])

# Plot predicted values
fig, ax = plt.subplots()
ax.scatter(df_plot['Car'], df_plot['logPrice'], alpha=0.5, label='observed')
ax.scatter(df_plot['Car'], townhousexresult.predict(),
           alpha=0.5, label='predicted')
ax.set_xlabel('Car')
ax.set_ylabel('Log(Price)')
ax.set_title('Predicted vs actual values for simple OLS regression')
ax.legend()

# %% Comparing estimation with and without outliers

# specify the formula (i.e. the model) with outliers
townhousereg = smf.ols('logPrice ~ Car', data=townhouse_df)

# now, fit (ie. estimate) the model
townhouseresult = townhousereg.fit()

# print(unitresult.summary())
# print(f"b1_hat = {unitresult.params['BuildingArea']}")
# print(f"SE(b1_hat) = {unitresult.bse['BuildingArea']}")


info_dict = {'R-squared': lambda x: "{:.2f}".format(x.rsquared),
             'No. observations': lambda x: "{0:d}".format(int(x.nobs))}

results_compare = summary_col(results=[townhousexresult, townhouseresult],
                              float_format='%0.4f',
                              stars=True,
                              model_names=['Without outliers',
                                           'With outliers'],
                              info_dict=info_dict,
                              regressor_order=['Car',
                                               'const'])

results_compare.add_title('Table - OLS Regressions - Townhouse & Car')

print(results_compare)


# %% Simple linear regression model: log(price) = b0 + b1 Landsize + u

townhouse_df = melb_df[melb_df.Type == 'h']
print(f"Average townhouse price = AU$ {townhouse_df.Price.mean():.0f}")
print(f"Average townhouse Landsize = {townhouse_df.Landsize.mean():.1f} km2")

# Visualisation using seaborn lmplot
gunit = sns.lmplot(data=townhouse_df, x='Landsize',
                   y='logPrice', fit_reg=False)
gunit.set(title='Melbourne townhouse Market - Landsize')

# %% Exclude a few outliers with Landsize >=2000
townhousex_df = townhouse_df[townhouse_df.Landsize < 2000]
print(
    f"Average townhouse price (exc. Landsize>=2000) = AU$ {townhousex_df.Price.mean():.0f}")
print("Average building area (exc. Landsize >=2000)" +
      f" = {townhousex_df.Landsize.mean():.1f}")

# Visualise the data without the outliers
gunitx = sns.lmplot(data=townhousex_df, x='Landsize',
                    y='logPrice', fit_reg=False)
gunitx.set(title='Melbourne townhouse Market (Exc. Landsize >=2000)')

# Now the data look more reasonable after we dropped the outliers, we proceed
# with estimating the hedonic model

# %% Use the formula approach of statsmodel to estimate

# import the formula module of statsmodels

# specify the formula (i.e. the model)
townhousexreg = smf.ols('logPrice ~ Landsize', data=townhousex_df)

# now, fit (ie. estimate) the model
townhousexresult = townhousexreg.fit()

print(townhousexresult.summary())
print(f"b1_hat = {townhousexresult.params['Landsize']}")
print(f"SE(b1_hat) = {townhousexresult.bse['Landsize']}")


# %% Plotting predicted (yhat) vs actual (y)

# Drop missing observations from whole sample
df_plot = townhousex_df.dropna(subset=['logPrice', 'Landsize'])

# Plot predicted values
fig, ax = plt.subplots()
ax.scatter(df_plot['Landsize'], df_plot['logPrice'],
           alpha=0.5, label='observed')
ax.scatter(df_plot['Landsize'], townhousexresult.predict(),
           alpha=0.5, label='predicted')
ax.set_xlabel('Landsize')
ax.set_ylabel('Log(Price)')
ax.set_title('Predicted vs actual values for simple OLS regression')
ax.legend()

# %% Comparing estimation with and without outliers

# specify the formula (i.e. the model) with outliers
townhousereg = smf.ols('logPrice ~ Landsize', data=townhouse_df)

# now, fit (ie. estimate) the model
townhouseresult = townhousereg.fit()

# print(unitresult.summary())
# print(f"b1_hat = {unitresult.params['BuildingArea']}")
# print(f"SE(b1_hat) = {unitresult.bse['BuildingArea']}")


info_dict = {'R-squared': lambda x: "{:.2f}".format(x.rsquared),
             'No. observations': lambda x: "{0:d}".format(int(x.nobs))}

results_compare = summary_col(results=[townhousexresult, townhouseresult],
                              float_format='%0.4f',
                              stars=True,
                              model_names=['Without outliers',
                                           'With outliers'],
                              info_dict=info_dict,
                              regressor_order=['Landsize',
                                               'const'])

results_compare.add_title('Table - OLS Regressions - townhouse & Landsize')

print(results_compare)


# %% Simple linear regression model: log(price) = b0 + b1 BuildingArea + u

townhouse_df = melb_df[melb_df.Type == 'h']
print(f"Average townhouse price = AU$ {townhouse_df.Price.mean():.0f}")
print(
    f"Average townhouse BuildingArea = {townhouse_df.BuildingArea.mean():.1f} km2")

# Visualisation using seaborn lmplot
gunit = sns.lmplot(data=townhouse_df, x='BuildingArea',
                   y='logPrice', fit_reg=False)
gunit.set(title='Melbourne townhouse Market - BuildingArea')

# %% Exclude a few outliers with BuildingArea >=400
townhousex_df = townhouse_df[townhouse_df.BuildingArea < 400]
print(
    f"Average townhouse price (exc. BuildingArea>=400) = AU$ {townhousex_df.Price.mean():.0f}")
print("Average building area (exc. BuildingArea >=400)" +
      f" = {townhousex_df.BuildingArea.mean():.1f}")

# Visualise the data without the outliers
gunitx = sns.lmplot(data=townhousex_df, x='BuildingArea',
                    y='logPrice', fit_reg=False)
gunitx.set(title='Melbourne townhouse Market (Exc. BuildingArea >=400)')

# Now the data look more reasonable after we dropped the outliers, we proceed
# with estimating the hedonic model

# %% Use the formula approach of statsmodel to estimate

# import the formula module of statsmodels

# specify the formula (i.e. the model)
townhousexreg = smf.ols('logPrice ~ BuildingArea', data=townhousex_df)

# now, fit (ie. estimate) the model
townhousexresult = townhousexreg.fit()

print(townhousexresult.summary())
print(f"b1_hat = {townhousexresult.params['BuildingArea']}")
print(f"SE(b1_hat) = {townhousexresult.bse['BuildingArea']}")


# %% Plotting predicted (yhat) vs actual (y)

# Drop missing observations from whole sample
df_plot = townhousex_df.dropna(subset=['logPrice', 'BuildingArea'])

# Plot predicted values
fig, ax = plt.subplots()
ax.scatter(df_plot['BuildingArea'], df_plot['logPrice'],
           alpha=0.5, label='observed')
ax.scatter(df_plot['BuildingArea'],
           townhousexresult.predict(), alpha=0.5, label='predicted')
ax.set_xlabel('BuildingArea')
ax.set_ylabel('Log(Price)')
ax.set_title('Predicted vs actual values for simple OLS regression')
ax.legend()

# %% Comparing estimation with and without outliers

# specify the formula (i.e. the model) with outliers
townhousereg = smf.ols('logPrice ~ BuildingArea', data=townhouse_df)

# now, fit (ie. estimate) the model
townhouseresult = townhousereg.fit()

# print(unitresult.summary())
# print(f"b1_hat = {unitresult.params['BuildingArea']}")
# print(f"SE(b1_hat) = {unitresult.bse['BuildingArea']}")


info_dict = {'R-squared': lambda x: "{:.2f}".format(x.rsquared),
             'No. observations': lambda x: "{0:d}".format(int(x.nobs))}

results_compare = summary_col(results=[townhousexresult, townhouseresult],
                              float_format='%0.4f',
                              stars=True,
                              model_names=['Without outliers',
                                           'With outliers'],
                              info_dict=info_dict,
                              regressor_order=['BuildingArea',
                                               'const'])

results_compare.add_title('Table - OLS Regressions - Townhouse & BuildingArea')

print(results_compare)


# %% Simple linear regression model: log(price) = b0 + b1 YearBuilt + u

townhouse_df = melb_df[melb_df.Type == 'h']
print(f"Average townhouse price = AU$ {townhouse_df.Price.mean():.0f}")
print(f"Average townhouse YearBuilt = {townhouse_df.YearBuilt.mean():.1f} km2")

# Visualisation using seaborn lmplot
gunit = sns.lmplot(data=townhouse_df, x='YearBuilt',
                   y='logPrice', fit_reg=False)
gunit.set(title='Melbourne townhouse Market - YearBuilt')

# %% Exclude a few outliers with YearBuilt <=1900
townhousex_df = townhouse_df[townhouse_df.YearBuilt > 1900]
print(
    f"Average townhouse price (exc. YearBuilt>=1900) = AU$ {townhousex_df.Price.mean():.0f}")
print("Average building area (exc. YearBuilt >=1900)" +
      f" = {townhousex_df.YearBuilt.mean():.1f}")

# Visualise the data without the outliers
gunitx = sns.lmplot(data=townhousex_df, x='YearBuilt',
                    y='logPrice', fit_reg=False)
gunitx.set(title='Melbourne townhouse Market (Exc. YearBuilt >=1900)')

# Now the data look more reasonable after we dropped the outliers, we proceed
# with estimating the hedonic model

# %% Use the formula approach of statsmodel to estimate

# import the formula module of statsmodels

# specify the formula (i.e. the model)
townhousexreg = smf.ols('logPrice ~ YearBuilt', data=townhousex_df)

# now, fit (ie. estimate) the model
townhousexresult = townhousexreg.fit()

print(townhousexresult.summary())
print(f"b1_hat = {townhousexresult.params['YearBuilt']}")
print(f"SE(b1_hat) = {townhousexresult.bse['YearBuilt']}")


# %% Plotting predicted (yhat) vs actual (y)

# Drop missing observations from whole sample
df_plot = townhousex_df.dropna(subset=['logPrice', 'YearBuilt'])

# Plot predicted values
fig, ax = plt.subplots()
ax.scatter(df_plot['YearBuilt'], df_plot['logPrice'],
           alpha=0.5, label='observed')
ax.scatter(df_plot['YearBuilt'], townhousexresult.predict(),
           alpha=0.5, label='predicted')
ax.set_xlabel('YearBuilt')
ax.set_ylabel('Log(Price)')
ax.set_title('Predicted vs actual values for simple OLS regression')
ax.legend()

# %% Comparing estimation with and without outliers

# specify the formula (i.e. the model) with outliers
townhousereg = smf.ols('logPrice ~ YearBuilt', data=townhouse_df)

# now, fit (ie. estimate) the model
townhouseresult = townhousereg.fit()

# print(unitresult.summary())
# print(f"b1_hat = {unitresult.params['YearBuilt']}")
# print(f"SE(b1_hat) = {unitresult.bse['YearBuilt']}")


info_dict = {'R-squared': lambda x: "{:.2f}".format(x.rsquared),
             'No. observations': lambda x: "{0:d}".format(int(x.nobs))}

results_compare = summary_col(results=[townhousexresult, townhouseresult],
                              float_format='%0.4f',
                              stars=True,
                              model_names=['Without outliers',
                                           'With outliers'],
                              info_dict=info_dict,
                              regressor_order=['YearBuilt',
                                               'const'])

results_compare.add_title('Table - OLS Regressions - Townhouse & BuildingArea')

print(results_compare)


# %% Multiple linear hedonic price model for House
# specify the formula (i.e. the model) without outliers
housemreg = smf.ols('logPrice ~ BuildingArea + Bedroom2 + Bathroom + Car' +
                    '+ Landsize + YearBuilt + Distance + Rooms', data=housex_df)

# now, fit (ie. estimate) the model
housemresult = housemreg.fit()

resultsm_compare_house = summary_col(results=[housexresult, housemresult],
                                     float_format='%0.4f',
                                     stars=True,
                                     model_names=['Simple Hedonic',
                                                  'Multiple Hedonic'],
                                     info_dict=info_dict,
                                     regressor_order=['BuildingArea',
                                                      'Bedroom2',
                                                      'Bathroom',
                                                      'Car',
                                                      'Landsize',
                                                      'YearBuilt',
                                                      'Distance',
                                                      'Rooms',
                                                      'const'])

resultsm_compare_house.add_title('Table - OLS Regressions- House')

print(resultsm_compare_house)


# %% Multiple linear hedonic price model for Townhouse
# specify the formula (i.e. the model) without outliers
townhousemreg = smf.ols('logPrice ~ BuildingArea + Bedroom2 + Bathroom + Car' +
                        '+ Landsize + YearBuilt + Distance + Rooms', data=townhousex_df)

# now, fit (ie. estimate) the model
townhousemresult = townhousemreg.fit()

resultsm_compare_townhouse = summary_col(results=[townhousexresult, townhousemresult],
                                         float_format='%0.4f',
                                         stars=True,
                                         model_names=['Simple Hedonic',
                                         'Multiple Hedonic'],
                                         info_dict=info_dict,
                                         regressor_order=['BuildingArea',
                                                          'Bedroom2',
                                                          'Bathroom',
                                                          'Car',
                                                          'Landsize',
                                                          'YearBuilt',
                                                          'Distance',
                                                          'Rooms',
                                                          'const'])

resultsm_compare_townhouse.add_title('Table - OLS Regressions- Townhouse')

print(resultsm_compare_townhouse)


# %% Create a set of regional dummy variables
# Create regional dummy variables
extmelb_df = pd.get_dummies(melb_df, columns=['Regionname'], drop_first=True)

# Remove spaces and other invalid characters from column names
extmelb_df.columns = extmelb_df.columns.str.replace(' ', '_')
extmelb_df.columns = extmelb_df.columns.str.replace('-', '_')
extmelb_df.columns = extmelb_df.columns.str.replace('(', '')
extmelb_df.columns = extmelb_df.columns.str.replace(')', '')

print(extmelb_df.columns)

# Fit the original model without regional dummy variables
original_formula = 'Price ~ BuildingArea + Bedroom2 + Bathroom + Car + Landsize + YearBuilt + Distance + Rooms'
original_model = smf.ols(formula=original_formula, data=extmelb_df).fit()
original_adj_r_squared = original_model.rsquared_adj


# %% For house type
house_extmelb_df = extmelb_df[extmelb_df.Type == 'h']

house_extmelb_df.columns = house_extmelb_df.columns.str.replace(' ', '_')
house_extmelb_df.columns = house_extmelb_df.columns.str.replace('-', '_')
house_extmelb_df.columns = house_extmelb_df.columns.str.replace('(', '')
house_extmelb_df.columns = house_extmelb_df.columns.str.replace(')', '')

# Fit the original model without regional dummy variables
# Calculate the adjusted R-squared for the house model
house_adj_r_squared = housemresult.rsquared_adj


# Fit the extended model with regional dummy variables
house_all_region_dummy_vars = [
    col for col in house_extmelb_df.columns if col.startswith('Regionname_')]
house_extended_formula = original_formula + \
    ' + ' + ' + '.join(house_all_region_dummy_vars)
house_extended_model = smf.ols(
    formula=house_extended_formula, data=house_extmelb_df).fit()
house_extended_adj_r_squared = house_extended_model.rsquared_adj


# Compare R-squared Adj values
print(f"House original R-squared Adj: {house_adj_r_squared:.4f}")
print(f"House Extended R-squared Adj: {house_extended_adj_r_squared:.4f}")

if house_extended_adj_r_squared > house_adj_r_squared:
    print("The Extended House Model improves on the goodness-of-fit of the model")
else:
    print("The Original House Model is better or equally good in terms of R-squared Adj.")


# %% For townhouse type
townhouse_extmelb_df = extmelb_df[extmelb_df.Type == 't']

townhouse_extmelb_df.columns = townhouse_extmelb_df.columns.str.replace(
    ' ', '_')
townhouse_extmelb_df.columns = townhouse_extmelb_df.columns.str.replace(
    '-', '_')
townhouse_extmelb_df.columns = townhouse_extmelb_df.columns.str.replace(
    '(', '')
townhouse_extmelb_df.columns = townhouse_extmelb_df.columns.str.replace(
    ')', '')


# Fit the original model without regional dummy variable
# Calculate the adjusted R-squared for the towntownhouse model
townhouse_adj_r_squared = townhousemresult.rsquared_adj


# Fit the extended model with regional dummy variables
townhouse_all_region_dummy_vars = [
    col for col in townhouse_extmelb_df.columns if col.startswith('Regionname_')]
townhouse_extended_formula = original_formula + \
    ' + ' + ' + '.join(townhouse_all_region_dummy_vars)
townhouse_extended_model = smf.ols(
    formula=townhouse_extended_formula, data=townhouse_extmelb_df).fit()
townhouse_extended_adj_r_squared = townhouse_extended_model.rsquared_adj


# Compare R-squared Adj values
print(f"Townhouse original R-squared Adj: {townhouse_adj_r_squared:.4f}")
print(
    f"Townhouse Extended R-squared Adj: {townhouse_extended_adj_r_squared:.4f}")

if townhouse_extended_adj_r_squared > townhouse_adj_r_squared:
    print("The Extended Townouse Model improves on the goodness-of-fit of the model")
else:
    print("The Original Townhouse Model is better or equally good in terms of R-squared Adj.")


# %% For unit
# Create regional dummy variables for each property type
unit_df = melb_df[melb_df.Type == 'u']
unit_extmelb_df = extmelb_df[extmelb_df.Type == 'u']

unit_extmelb_df.columns = unit_extmelb_df.columns.str.replace(' ', '_')
unit_extmelb_df.columns = unit_extmelb_df.columns.str.replace('-', '_')
unit_extmelb_df.columns = unit_extmelb_df.columns.str.replace('(', '')
unit_extmelb_df.columns = unit_extmelb_df.columns.str.replace(')', '')


# Fit the original model without regional dummy variable - Unit
# Calculate the adjusted R-squared for the townunit model
unitmreg = smf.ols('logPrice ~ BuildingArea + Bedroom2 + Bathroom + Car' +
                   '+ Landsize + YearBuilt + Distance ', data=unit_df)

# now, fit (ie. estimate) the model
unitmresult = unitmreg.fit()


unit_adj_r_squared = unitmresult.rsquared_adj


# Fit the extended model with regional dummy variables
unit_all_region_dummy_vars = [
    col for col in unit_extmelb_df.columns if col.startswith('Regionname_')]
unit_extended_formula = original_formula + \
    ' + ' + ' + '.join(unit_all_region_dummy_vars)
unit_extended_model = smf.ols(
    formula=unit_extended_formula, data=unit_extmelb_df).fit()
unit_extended_adj_r_squared = unit_extended_model.rsquared_adj


# Compare R-squared Adj values
print(f"Unit original R-squared Adj: {unit_adj_r_squared:.4f}")
print(f"Unit Extended R-squared Adj: {unit_extended_adj_r_squared:.4f}")

if unit_extended_adj_r_squared > unit_adj_r_squared:
    print("The Extended Unit Model improves on the goodness-of-fit of the model")
else:
    print("The Original Unit Model is better or equally good in terms of R-squared Adj.")


# %% For all properties
# Fit the extended model with regional dummy variables
all_region_dummy_vars = [
    col for col in extmelb_df.columns if col.startswith('Regionname_')]
extended_formula = original_formula + ' + ' + ' + '.join(all_region_dummy_vars)
extended_model = smf.ols(formula=extended_formula, data=extmelb_df).fit()
extended_adj_r_squared = extended_model.rsquared_adj

# Compare R-squared Adj values
print(f"Original R-squared Adj: {original_adj_r_squared:.4f}")
print(f"Extended R-squared Adj: {extended_adj_r_squared:.4f}")

if unit_extended_adj_r_squared > extended_adj_r_squared:
    print("The Extended Unit Model has a higher R Square Adj. compared to Combine model")
else:
    print("The Extended Combine Model has a higher R Square Adj. compared to Unit model")


if house_extended_adj_r_squared > extended_adj_r_squared:
    print("The Extended House Model has a higher R Square Adj. compared to Combine mdodel")
else:
    print("The Extended Combine Model has a higher R Square Adj. compared to House model")

if townhouse_extended_adj_r_squared > extended_adj_r_squared:
    print("The Extended Townhouse Model has a higher R Square Adj. compared to Combine model")
else:
    print("The Extended Combine Model has a higher R Square Adj. compared to Townhouse model")
