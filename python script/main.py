#!/usr/bin/env python
# coding: utf-8

# <font size="6"> <div style="text-align: center"> **Time Series Regression on Cloud Analytics Case Study**

# # Introduction

# A consultant's firm client wants to mitigate its costs on AWS platform and, so, we are in charge of finding a solution for this issue based on 2 solutions:
# 1.	Forecasting Cloud Spend for the Next Quarter: The client wants to develop a forecast for their cloud spend to build their budget for the upcoming quarter. This involves using the cost and billing usage report from the past 18 months to predict the expected cloud spend for the next quarter accurately.
# 2.	Identifying Spend Areas for Optimization Efforts: The client wants to understand which specific AWS services are driving growth in their spend so that they can prioritize optimization efforts. He also wants to know if the efforts on Product and/or on Service should be prioritized and which ones.
# 
# We'll decompose the answer of the 2 problems into 4 parts:
# 
# 
# 
# *   Data Exploration and Analysis
# *   Feature Engineering
# *   Forecasting models using ML and DL to answer question 1
# *   Cluster Analysis to answer question 2
# 
# 
# 
# 
# 

# In[6]:


# import libraries
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import numpy as np
import datetime as dt
import itertools
import random
import re
import math

from sklearn.metrics import *
from sklearn.preprocessing import MinMaxScaler
from plotnine import *
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans

from sklearn.metrics import mean_absolute_percentage_error as mape
from sklearn.multioutput import MultiOutputRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import cross_val_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

import warnings
warnings.filterwarnings("ignore")


# ## Data Upload

# In[7]:


# import and see nb rows and columns of the input dataset
df_raw = pd.read_csv('/content/drive/MyDrive/cost_data.csv',  sep='|')
df_raw.shape


# ## Exploratory Data Analysis

# ### Global Analysis

# In[8]:


# To verify uniqueness of records
df_raw.duplicated().sum()


# In[9]:


# verify columns and their dtypes
df_raw.info()


# In[10]:


# remove spaces and paranthesis in columns name
df_raw.columns = df_raw.columns.str.replace(' ','_').str.replace('[()]', '')
df_raw.columns


# In[11]:


# brief look at the raw dataset
df_raw.head(5)


# In[12]:


# transform usage_date to datetime

df_raw['usage_date'] = pd.to_datetime(df_raw['usage_date'])
df_raw.info()


# In[13]:


# describe and see summary statistics for all the variables of our dataset

def summary(df):
  desc = df.describe(include = 'all').T.rename(columns={"count": "count_non_nulls"})
  desc['count_nulls'] = df.shape[0]-desc['count_non_nulls']
  new_index_list = ['count_non_nulls', 'count_nulls']
  new_index=new_index_list + [c for c in desc.columns if c not in new_index_list]
  desc = desc.reindex(new_index, axis='columns')

  return desc

desc_ini = summary(df_raw)
desc_ini


# So far, we can see that this dataset is relatively large with low dimensionality in terms of columns. We have a date column ('usage_date'), which will be key to perform the time series forecasting, 2 measure float continuous variables ('usage_amount' and 'amortized_cost') and 8 remaining categorical variables (some related with AWS service and the others related client’s infrastructure components).
# **The variable 'amortized_cost' which is what we want to predict in the forecast - dependent variable of our models.**

# We see some coluns with null values as well and we have to decide later what to do with those missing values. Some categorical values have a lot of cardinality as well (or distinct values), which is not a good indicator for creating any model. However, there are some dominant values and there can be a lot of insignificant categories in each variable.
# As far as the continuous variables we can see, is that there are negative values, which don't make sense and will be removed (unless is some kind of correction) and the percentile 25 is 0 in both cases, indicating, right away, **that we have a lot of free costs usage and a lot of operations with an insignicficant amount of usage**.

# Below, let's try to find relationships between variables and do exploratory analysis on what can be cleaned as well or redo, in order to reduce some volumetry for the database (which will easy the modelling part) as well to do some treatments on the categorical fields - feature engineering.

# ###Variable Analysis / Feature Engineering

# #### Date Variable - *Usage Date*

# This is the date variable that is key to model the forecasting with a certain lag the future 6 months of costs. This is has a daily distribution and goes from 2021-10-01 to 2023-03-31.

# In[14]:


# let's check if we have all days

date_gb = df_raw.groupby('usage_date', dropna=False).agg({'amortized_cost':'count'}).reset_index()

date_gb['date_gap'] = date_gb['usage_date'].diff()
date_gb


# In[15]:


date_gb.date_gap.value_counts()


# Indeed we have all consecutive days from first to last date!

# It might be relevant to include variables like the weekday, the month, or even the quarter, to see if there is some **seasonality** in terms of higher or lower costs. For the year, it doesnt make sense to check because we have less months for 2021 and 2023 compared to the full 2022.

# In[16]:


df_initial = df_raw.copy()
df_initial['month'] = pd.DatetimeIndex(df_initial['usage_date']).month
df_initial['quarter'] = pd.DatetimeIndex(df_initial['usage_date']).quarter
df_initial['weekday'] = pd.DatetimeIndex(df_initial['usage_date']).weekday # 0 monday to 6 sunday
df_initial


# **Let's check now the distribution of days with the dependent variable, the amortized cost.**

# In[19]:


# plot the relationship between both variables
def scatterplot(df, x, y, title, figsize = (10,10), ylim = (0,None)):
  plt.figure(figsize=figsize)
  plt.scatter(df[x], df[y])
  plt.xlabel(x)
  plt.ylabel(y)
  plt.ylim(ylim)
  plt.title(title)

  return plt.show()

scatterplot(df_initial, 'usage_date', 'amortized_cost', "Cost per Day")


# In[20]:


# now zoomed in

scatterplot(df_initial, 'usage_date', 'amortized_cost', "Cost per Day", ylim = (0,100))


# Right now, its difficult to reach any conclusion, as expected, because we have many data points. What we can see already are outliers.
# **Let's check the same but for the other new date columns (filtering for 2022 only so that we avoid bias with incomplete 2021 and 2023 years).**

# In[21]:


#zoomed in as well from now on to capture the least amount of outliers

df_2022 = df_initial[pd.DatetimeIndex(df_initial['usage_date']).year == 2022]


scatterplot(df_2022, 'month', 'amortized_cost', "Cost per Month", ylim = (0,100))


# In[22]:


scatterplot(df_2022, 'quarter', 'amortized_cost', "Cost per Quarter", ylim = (0,100))


# In[23]:


scatterplot(df_2022, 'weekday', 'amortized_cost', "Cost per Weekday", ylim = (0,100))


# So, with all these graphs, it's noticeable that:
# 
# 
# *   There seems to have more concentration of costs in the beggining of the year than in the middle of the year (Q1 vs Q3).
# *   As expected, July and August seem to be the months where there are less costs (due to reduced working load between working people and resources usage, as its presumable).
# *   Regarding the week day, it seems more or less uniform between them.
# 
# 
# 
# 

# #### Dependent Variable - Amortized Cost

# This is the variable we want to predict based on several other independent variables, like the Product Tag and many others, as we will see.

# In[24]:


sns.boxplot(
    data=df_initial, x="amortized_cost",
    notch=True, showcaps=False,
    flierprops={"marker": "x"},
    boxprops={"facecolor": (.4, .6, .8, .5)},
    medianprops={"color": "coral"},
)


# The median is very close to 0 as seen here in the boxplot above (we can't even see a box given the huge dispersion of outliers).
# So, now, let's remove some outliers based on the 99 percentile.

# In[25]:


df2 = df_initial[df_initial.amortized_cost < df_initial.amortized_cost.quantile(.99)]

summary(df2)


# Even though we removed only 17377 rows, we see that the maximum value now for the dependent variable (trimmed to the 99% observation) is only about 34, which is a very different level of magnitude compared to other values.

# #### *Usage Amount*

# This can be potentially one of the many independent variables and it's the only other float continuous type. It's interesting to check for outliers in this variables as well and **the relationship it has with the amortized_cost.**

# In[26]:


sns.boxplot(
    data=df_initial, x="usage_amount",
    notch=True, showcaps=False,
    flierprops={"marker": "x"},
    boxprops={"facecolor": (.4, .6, .8, .5)},
    medianprops={"color": "coral"},
)


# The same kind of conclusions apply here, where we have some extreme values that bias the distribution of values visually, so, **let's remove the 1% higher percentile as well.**

# In[27]:


df3 = df2[df2.usage_amount < df2.usage_amount.quantile(.99)]
summary(df3)


# Still, we see that maximum values can be considered outliers when compared to the 75 percentile (same applies to the cost variable), however, let's keep them because they can be useful for the models.

# Now, it's time to remove some more rows based on the lack of coeherency between these 2 variables. In my opinion, **having negative costs or usage amount don't make any sense, and, observations with 0 usage and 0 costs are not relevant for our objective, so, they will be deleted.**

# In[28]:


df4 = df3[~((df3['amortized_cost'] <= 0) & (df3['usage_amount'] <= 0) | (df3['amortized_cost'] < 0) | (df3['usage_amount'] < 0))]
summary(df4)


# In[29]:


# relationship of usage with cost, values filtered to be closer to the median (on the usage), so that we can try to extrapolate something

mpl.rcParams['agg.path.chunksize'] = 10000

plt.figure(figsize=(10,10))
plt.scatter(df4['usage_amount'], df4['amortized_cost'])
plt.xlabel('usage_amount')
plt.ylabel('amortized_cost')
plt.ylim((0, 1))
plt.xlim((0, 0.30))
plt.title('Cost per usage')

plt.show()


# By analysing the above graph, we conclude it's not easy to find a direct relationship between these 2 variables. On one hand, we see one line that indicates a positive correlation between them (whenever the usage goes up, the costs as well), but we see a lot of free usage as well, where we use free products, most likely, or, on the other hand, we have can have little usage and spend a lot (more expensive products).

# #### *Line Item type*

# This variable indicates us what type of service are we using. It's pretty simple with only 4 distinct classes and let's see if it's useful for discovering relations with the dependent variable.

# In[30]:


plt.figure(figsize=(15,8))
splot = sns.countplot(x="line_item_type",data=df4)
for p in splot.patches:
  splot.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., p.get_height()),  ha = 'center', va = 'center', xytext = (0, 5), textcoords = 'offset points')


# In[31]:


plt.figure(figsize=(15,8))
splot = sns.barplot(data=df4, x="line_item_type", y="amortized_cost")
for p in splot.patches:
  splot.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., p.get_height()),  ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')


# As we see, there is a huge dominance in "Usage" kind of line item with more than 98% of totalcount of rows. As far as cost is concerned, we see actually being the category with the least average of it (probably most of them happen when we have free costs (=0)). **This means, that it can be useful to still consider this variable instead of just ignoring it. However, we will merge all other 3 values into one - "other"**. Still, later, it's useful to discover if this new category has some meaning whatsoever.

# In[32]:


df5 = df4.copy()
df5['line_item_type'] = np.where(df5['line_item_type'] != 'Usage', 'Other', df5['line_item_type'])
df5.groupby('line_item_type', dropna=False).agg({'amortized_cost':'count', 'usage_amount':'sum'}).reset_index()


# #### *Product Code*

# This variable indicates us what AWS service/product was used on each daily operation. It contains 58 distinct values right now, so let's assess its relevance and distribution.

# In[33]:


product_grouped = df5.groupby('product_code', dropna=False).agg({'amortized_cost':['count', 'sum']}).reset_index()#.sort_values('amortized_cost', ascending=False)
product_grouped.columns = product_grouped.columns.map('_'.join)
product_grouped['pct_count'] = round(product_grouped['amortized_cost_count'] * 100/sum(product_grouped['amortized_cost_count']), 1)
product_grouped['pct_cost'] = round(product_grouped['amortized_cost_sum'] * 100/sum(product_grouped['amortized_cost_sum']), 1)#.sort_values('pct_cost', ascending=False)
product_grouped = product_grouped.sort_values('pct_cost', ascending=False)
product_grouped


# In[34]:


# what if we choose just the values which appear either more than 1% of total rows or total costs?
pg_new = product_grouped[(product_grouped['pct_count'] > 1) | (product_grouped['pct_cost'] > 1)]
pg_new


# As we can see, the dominance in this field is not so high as the previous one. However, there are a lot of irrelevant values whether because they are not representative in the whole column or they don't have many costs associated.
# What we will do, in this case, is to take the first 20 values (representing more than 1% of either costs or count) and the remaining we put as "other").

# In[35]:


df6 = df5.copy()
df6['product_code'] = np.where(df6['product_code'].isin(list(pg_new['product_code_'])), df6['product_code'], 'Other')
df6.groupby('product_code', dropna=False).agg({'amortized_cost':'count', 'usage_amount':'sum'}).reset_index()


# #### *Line Item Description*

# This variable contains 2 kinds of info in the same string:
# - the cost per something (some unit usage)
# - the place where it was (an instance, a country, etc).
# 
# As we are going to see now, this information can be irrelevant for the fact that we might have the cost per usage now (it's basically the 2 variables we have divided) - **so can be redudant to have this data again** and the remaining can be incorporated in other variables already directly. Also, there are a lot of different values in this field which can create too much noise to our models.

# In[36]:


# let's confirm that the cost per unit is similar to what we have already as a ratio

df_lid = df6[['amortized_cost', 'usage_amount', 'line_item_description']].copy()

df_lid['description_price'] = df_lid.line_item_description.str.extract('([0-9][.]*[0-9]*)').astype('float')
df_lid['price_per_unit'] = np.round(np.where(df_lid['usage_amount'] == 0, 0, df_lid['amortized_cost']/df_lid['usage_amount']).astype('float'), 3)
df_lid.head(5)


# In[37]:


df_lid.describe().T


# In[38]:


lid_grouped = df6.groupby('line_item_description', dropna=False).agg({'amortized_cost':['count', 'sum']}).reset_index()
lid_grouped.columns = lid_grouped.columns.map('_'.join)
lid_grouped['pct_count'] = round(lid_grouped['amortized_cost_count'] * 100/sum(lid_grouped['amortized_cost_count']), 1)
lid_grouped['pct_cost'] = round(lid_grouped['amortized_cost_sum'] * 100/sum(lid_grouped['amortized_cost_sum']), 1)
lid_grouped = lid_grouped.sort_values('pct_count', ascending=False)
lid_grouped


# To conclude, we see that the price per unit seems to match the amount of amortized cost per usage amount (and what we have as usage_amount seem to match the amount of GBs consumed).  Also, the values are really spread out and with few meaning for all (there is no dominant distinct value).
# **Therefore, it looks like this variable might be irrelevant to our work.**

# #### *Usage Type*

# This variable is related with another AWS service detail. Its mode is "Data Volume" and let's see remaining cardinality distribution.

# In[39]:


plt.figure(figsize=(30,8))
splot = sns.countplot(x="usage_type",data=df6)
for p, item in zip(splot.patches, splot.get_xticklabels()):
  splot.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., p.get_height()),  ha = 'center', va = 'center', xytext = (0, 5), textcoords = 'offset points')
  item.set_rotation(45)


# In[40]:


# relate with the cost variable
plt.figure(figsize=(30,8))
splot = sns.barplot(data=df6, x="usage_type", y="amortized_cost")
for p, item in zip(splot.patches, splot.get_xticklabels()):
  splot.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., p.get_height()),  ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')
  item.set_rotation(45)


# As we see, there is a **dominance** in terms of value distribution within its class in **"Data Volume" and "Operations"**. This doesn't translate into higher average costs of course, as we know that the majority of costs are close to 0.
# - We observe, as well, that we have some "Operations" values related between them (specifying the time span in parenthesis), but probably grouping all of them is a good idea. Same applies to the "Instance Hour" values.
# - We have some nulls in our dataset that by itself might not have the most impact but still, can be useful so we'll keep them.
# - We will aggregate the remaining values to "Other" as well as the nulls!

# In[41]:


df7 = df6.copy()
df7['usage_type'] = df7['usage_type'].fillna('Other')
df7['usage_type'] = np.where(df7['usage_type'].str.contains('Operations'), 'Operations', df7['usage_type'])
df7['usage_type'] = np.where(df7['usage_type'].str.contains('Instance Hour'), 'Instance Hour', df7['usage_type'])

list_to_keep = ['Operations', 'Data Volume', 'Storage', 'Instance Hour']
df7['usage_type'] = np.where(df7['usage_type'].isin(list_to_keep), df7['usage_type'], 'Other')


plt.figure(figsize=(20,8))
splot = sns.countplot(x="usage_type",data=df7)
for p in splot.patches:
  splot.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., p.get_height()),  ha = 'center', va = 'center', xytext = (0, 5), textcoords = 'offset points')


# #### *Usage Type AWS*

# This variable is related with the previous one but with a lot of cardinality, just like the "line_item_description".

# In[42]:


df_uta = df7[['amortized_cost', 'usage_amount', 'usage_type', 'usage_type_aws']].copy()

top_values = df_uta.groupby('usage_type_aws', dropna=False).agg({'amortized_cost':['count', 'sum']}).reset_index()
top_values.columns = top_values.columns.map('_'.join)
top_values = top_values.sort_values('amortized_cost_count', ascending=False)
top_values.head(20)


# As seen with the top 20 values in terms of frequency, we have some options to deal with this column:
# - First, we can specify the type of usage, so for instance have values like "Requests - Tiers2, DataTransfer-Out-Bytes, DNS-Queries, etc etc";
# - OR, we can split and create a new column with the info of the name of the server where the data is being moved: "USW1, EUC1, etc etc".
# 
# 
# **The problem here is that there are many correlated values within each distribution (for instance we can have EUC1 server, but on other we have EUC1-USW2), which can indicate that both servers have interactions here.**
# 
# Even if we do treatments, it's difficult to get to a correct distribution or else we get a high cardinality still without any dominant value. **Thus, it's better to leave out this variable for now.**

# In[43]:


################# test to use usage type aws for server - will not be used

df8 = df7.copy()

# get either the left string before the first '-' or ':'. Also, put it all uppercase so we dont have discrepancies
df8['server'] = df8['usage_type_aws'].str.replace(':', '-').str.split('-').str[0]
df8['server'] = df8['server'].str.upper()
uta_group = df8.groupby('server', dropna=False).agg({'amortized_cost':'count', 'usage_amount':'sum'}).reset_index()
uta_group


# #### *Operation*

# This variable indicates us other detail of an AWS service. Right now, after cleaning, it contains only 1 null and 354 distinct values which is quite a high number. Let's see how can we fix this possible issue.

# In[44]:


op_grouped = df7.groupby('operation', dropna=False).agg({'amortized_cost':['count', 'sum']}).reset_index()
op_grouped.columns = op_grouped.columns.map('_'.join)
op_grouped = op_grouped.sort_values('amortized_cost_sum', ascending=False)
op_grouped.head(20)


# **We can do similar treatments here as done in the past.** First, lets consider all strings before ":" or "-", because we have operations like "CreateDBInstance:0002" and "	CreateDBInstance:0014", for instance, which might not be so relevant to distinguish this detail. So, we'll group them all and, then, we'll consider a large cumulative percentage of distincts and then treat remaining categories as "Other".

# In[45]:


df8 = df7.copy()

# the null = 'unknown' because its a meaning of a null and we have 'None' and 'none' in the dataset which mean the same
df8['operation'] = df8['operation'].fillna('Unknown')
df8['operation'] = df8['operation'].str.replace(':', '-').str.split('-').str[0]
df8['operation'] = df8['operation'].str.replace('none', 'None')
op_grouped_treated = df8.groupby('operation', dropna=False).agg({'amortized_cost':['count', 'sum']}).reset_index()
op_grouped_treated.columns = op_grouped_treated.columns.map('_'.join)
op_grouped_treated = op_grouped_treated.sort_values('amortized_cost_count', ascending=False)

op_grouped_treated['pct_count'] = (op_grouped_treated['amortized_cost_count']/sum(op_grouped_treated['amortized_cost_count']))*100
op_grouped_treated['pct_cost'] = (op_grouped_treated['amortized_cost_sum']/sum(op_grouped_treated['amortized_cost_sum']))*100
op_grouped_treated['count_cumsum'] = round(op_grouped_treated['pct_count'].cumsum(), 1)

op_grouped_treated


# In[46]:


# Let's define a 90% threshold on the 'count_cumsum' column to retain the distinct categories and the remaining will be 'Other'

op_list_keep = list(op_grouped_treated[(op_grouped_treated['count_cumsum'] <= 90)]['operation_'])
len(op_list_keep)


# In[47]:


df8['operation'] = np.where(df8['operation'].isin(op_list_keep), df8['operation'], 'Other')
df8.groupby('operation', dropna=False).agg({'amortized_cost':'count', 'usage_amount':'sum'}).reset_index()


# **We managed to keep still a high cardinality (88 distinct values) but more manageable now!**

# #### *Environment Tag*

# This variable is now related with which portions of the client infrastructure are using the services and what's their work environment. Pretty straigthforward with 15 distinct string values but with a lot of nulls. Let's inspect:

# In[48]:


plt.figure(figsize=(15,8))
splot = sns.countplot(x="environment_tag",data=df8.fillna('Missing'))
for p in splot.patches:
  splot.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., p.get_height()),  ha = 'center', va = 'center', xytext = (0, 5), textcoords = 'offset points')


# In[49]:


plt.figure(figsize=(15,8))
splot = sns.barplot(data=df8.fillna('Missing'), x="environment_tag", y="amortized_cost")
for p in splot.patches:
  splot.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., p.get_height()),  ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')


# Some interesting results we have here:
# - First, the nulls dominate our distribution
# - We have a '¯\_(ツ)_/¯' value which is comic and it can be relatable to the nulls as well (we don't know either), so let's group them and call it 'Unknown'
# - Like other cols, some values here can be aggregated so that we have less environment tags.

# In[50]:


df9 = df8.copy()

df9['environment_tag'] = df9['environment_tag'].fillna('unknown') # fill nulls
df9['environment_tag'] = df9['environment_tag'].str.replace('¯\_(ツ)_/¯', 'unknown', regex=False) # replace other value to match nulls
df9['environment_tag'] = df9['environment_tag'].str.replace('production', 'prod') # replace production to prod
df9['environment_tag'] = df9['environment_tag'].str.replace('_', '-').str.split('-').str[0] # take left of '_' or '-'
df9['environment_tag'] = df9['environment_tag'].str.replace('01', '') # remove numbers from string
df9['environment_tag'] = df9['environment_tag'].str.replace('us', 'unknown') # put the "us" values into the "unknown" category because we don't know his environment still
df9['environment_tag'] = df9['environment_tag'].str.lower() # lowercase all
df9 = df9[df9['environment_tag'] != 'corporate']# remove "corporate" row as it is only 1 observation and it quite some cost, so can be seen as outlier (also, this might not be a valid environment tag)


# In[51]:


plt.figure(figsize=(15,8))
splot = sns.countplot(x="environment_tag",data=df9)
for p in splot.patches:
  splot.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., p.get_height()),  ha = 'center', va = 'center', xytext = (0, 5), textcoords = 'offset points')


# #### *Product Tag*

# This is the final independent variable and crutial for our forecast model. As requested, we have to include at least this attribute to the model, so, let's check its importance:

# In[52]:


plt.figure(figsize=(15,8))
splot = sns.countplot(x="product_tag",data=df9.fillna('Missing'))
for p in splot.patches:
  splot.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., p.get_height()),  ha = 'center', va = 'center', xytext = (0, 5), textcoords = 'offset points')


# In[53]:


plt.figure(figsize=(15,8))
splot = sns.barplot(data=df9.fillna('Missing'), x="product_tag", y="amortized_cost")
for p in splot.patches:
  splot.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., p.get_height()),  ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')


# As we see above, the values range from "Product 1" to "Product 8". The Product 2 doesnt have any relevance so, we can remove those small lines.
# **As far as the nulls are concerned, it's "safe" to remove them as well, since they contain no meaningful importance and they can bias the forecast as well, since the mean for the amortized cost is quite high when compared to all the other values (outlier)**. We'll keep the other categories as they are even though some might have smaller expression in the data we have.

# In[54]:


df10 = df9.copy()

df10 = df10[~((df10['product_tag'].isnull()) | (df10['product_tag'] == 'Product 2'))]

summary(df10)


# ### New features - One Hot Encoding

# Now, to turn our models more efficient or even possible (in case of a time series regression), we have to transform the categorical variables into int variables. To do this, we can do several ways, namely by:
# - Encoding our data and creating a multi flag inside each column, which lowers the dimensionality but it might "confuse" our models (in the sense that higher number cardinality doesn't mean anything in this case);
# - Target encoding, which actually could be very useful and it keeps the dimensionality low as well (it's like the previous method but it weights regarding the distribution of values in each class)
# - **One Hot Encoding**, which is basically creating multiple binary flags representing each class of each variable.
# This is a very popular method, so let's check its appeal.

# In[55]:


# But first, remove the 2 columns and groupby all the others so we have less volumetry
df11 = df10.copy()

df11 = df11.drop(columns = ['line_item_description', 'usage_type_aws'])

df11 = df11.groupby([c for c in df11.columns if c not in ['amortized_cost', 'usage_amount']], dropna=False)             .agg({'amortized_cost':'sum', 'usage_amount':'sum'}).reset_index()

summary(df11)


# **Voilá! The volumetry of the database drastically changed when we removed both columns and by changing the granularity to a higher level, we have much less rows (around 388k)**, which can improve our models' performance. However, this has a cost where we can loose information of each single point with both removed columns information.

# In[56]:


df_encoded = pd.get_dummies(df11, columns = ['line_item_type', 'product_code', 'usage_type', 'operation', 'environment_tag', 'product_tag'])

# replace spaces in col names with "_" and lowercase it
df_encoded.columns = df_encoded.columns.str.replace(' ','_')
df_encoded.columns = df_encoded.columns.str.lower()

print(df_encoded.columns)


# In[57]:


df_encoded.shape


# We have 136 columns in total now and although some dummies might be irrelelvant (in a sense that per field with X distinct values we only need X-1 dummies (if all are 0 then we match the last value). However, since we have fields with a lot of dimensions in them, it might be more useful to still add X variables per categoric field.

# ##Data Normalization

# This step is very important, as we need to rescale our data for modelling reasons (with exception of the date column). This means that standardizing and putting all of them in the same scale (namely usage_amount, amortized_cost and remaining dummy variables) is much needed to improve accuracy.
# 
# ---
# 
# 
# **The method we are going to choose is the normalization, doing a min max scaling between 0 and 1**.
# Other methods like Standard Scaler could be used but this method is more intuitive and easier to interpret and also less sensitive to outliers, which can be important in this case, as we still decided to kept some possible outliers in our data.

# In[58]:


# Create an instance of the MinMaxScaler class
min_max_scaler = MinMaxScaler()

# Normalize our dataset (returns an array)
normalized_cols = [c for c in df_encoded.columns if c != 'usage_date']

normalized_df = df_encoded.copy()

normalized_df[normalized_cols] = min_max_scaler.fit_transform(normalized_df[normalized_cols])

# Convert the array above back into a pandas DataFrame
normalized_df = pd.DataFrame(data=normalized_df, columns=normalized_df.columns)

#reindex to have the dependent variable as a last column
new_index=[c for c in normalized_df.columns if c != 'amortized_cost'] + ['amortized_cost']
normalized_df = normalized_df.reindex(new_index, axis='columns')

summary(normalized_df)


# ##Time Series Feature Engineering

# - For the next step, we'll expand further the variables we have but reduce drastically the volume of our DB, having into account that we want to model based on our daily data but with **multi lagged variables**.
# For this, we need to resample our data at a daily granularity doing the mean of all the features (so, it works like an aggregation step).
# 
# - Then, it will be useful to assess if an the best approach should be an **Auto Regressive model** (so using the target variable: amortized_cost as predictors/independent variables as well) or if it's better to not mix with it and keep it only as the target.
# 
# - Finally, we'll remove some features based on the importance of each variable before doing our forecasting model.

# In[59]:


def date_granularity(df, time_gran):
  # setting time as index

  df_new = df.copy()
  df_new['usage_date'] = pd.to_datetime(df_new['usage_date'])

  df_new.set_index('usage_date', inplace=True)

  # resampling to daily data and summing all variables
  df_new = df_new.resample(time_gran).sum()

  # normalize again the data
  min_max_scaler = MinMaxScaler()
  daily_norm = min_max_scaler.fit_transform(df_new)

  df_new = pd.DataFrame(data=daily_norm, columns = df_new.columns, index = df_new.index)

  return df_new

daily_df = date_granularity(normalized_df, 'D')
summary(daily_df)


# ###EDA with different Time Spans

# In[60]:


# plotting again the cost per day

plt.figure(figsize = (10,10))

viz_df = daily_df.reset_index()

plt.plot(viz_df['usage_date'], viz_df['amortized_cost'], label='cost')
plt.plot(viz_df['usage_date'], viz_df['usage_amount'], label = 'usage amount')

plt.legend()
plt.show()


# As seen above, we can take a **completely different conclusion than before**. In fact, there seems to be a generalized upwards trend from 2021 onwards and this might be due to the fact that we were in COVID years until end of 2022 where life tended to become normal like before and, so, maybe costs and usage rose. Despite these spikes in may 2022 and October 2022 (these can be outliers), we see staber floatings and upwards trend from  6 to 6 months. This info may be very useful for our modelling and indicates very well what should be our lag number (so having a 6 months lag span). *There seems to be a very good correlation between the cost and the usage as well!*
# **To further compare, we'll do the same graphs for weekly data and for monthly data and decide later on the best time granularity to use.**

# In[61]:


# plotting the cost per week

week_df = date_granularity(normalized_df, 'W')
viz_df = week_df.reset_index()

plt.figure(figsize = (10,10))
plt.plot(viz_df['usage_date'], viz_df['amortized_cost'], label='cost')
plt.plot(viz_df['usage_date'], viz_df['usage_amount'], label = 'usage amount')

plt.legend()
plt.show()


# In[62]:


# plotting the cost per month

month_df = date_granularity(normalized_df, 'M')
viz_df = month_df.reset_index()

plt.figure(figsize = (10,10))
plt.plot(viz_df['usage_date'], viz_df['amortized_cost'], label='cost')
plt.plot(viz_df['usage_date'], viz_df['usage_amount'], label = 'usage amount')

plt.legend()
plt.show()


# Analysing monthly graph, we don't see much findings in terms of patterns like we saw with daily data. For weekly, we see more interesting conclusions with a better pattern but still a bit weird, specially in the final weeks of our dataset.
# 
# **Thus, we will keep on using daily data.**

# ###Correlation between Variables and Features Removal

# Let's assess for a Pearson's correlation test (using an heatmap) and check for redudant variables or other possible relationships between them.
# Since, we have lots of variables, we are only going to assess the **relationship between the continuous variables and product_tag**, since these are requirement for our independent variables.

# In[63]:


# since we have many variables, lets remove the variable "operation..." for this part only
corr = daily_df[[c for c in daily_df.columns if c.startswith('product_tag') or c in ['usage_amount', 'amortized_cost']]].corr()
figure = plt.figure(figsize=(10,10))
sns.heatmap(corr, annot=True, fmt = '.1g')


# As expected, the paradigm changes now. We see much more correlation when we resample the data and there are even variables pratically all correlated: this is the case for usage_amount and amortized_cost (proven in the graph above) as well as the product_tag_4 and product_tag_6, with almost 1 in Pearson correlation.
# *Almost all seem to have a positive mid to strong connection with the dependent variable, with exception of product_tag_product_1 which seems to be negatively correlated*.
# 
# 
# 
# 
# **Thus, we shall remove redundant variables with correlation higher than 0.9 to the target variable and those who have null values (means that its normalized values are always 0 or close) and we'll do this for the whole dataset (not just for these features).**

# In[64]:


corr_matrix = daily_df.corr()
corr_matrix


# In[65]:


# fill the ones all null so they are removed
corr_matrix = daily_df.corr().abs().fillna(0.99)

keep_corr = corr_matrix[corr_matrix <= 0.9]
redundant_corr = corr_matrix[corr_matrix > 0.9]
redundant_corr


# In[66]:


cols_to_keep = list(keep_corr[keep_corr.index == 'amortized_cost'].dropna(axis=1).columns) + ['amortized_cost']
cols_to_keep


# In[67]:


len(cols_to_keep)


# To conclude, we will remove a lot of variables now because most of them were extremely high correlated with the costs variable, making them redundant and could potentially hurt the model or at least don't provide useful information.
# **The usage_amount was one of those variables given its pretty strong positive correlation. Also, the product tag of product 7 was proven to be redundant too.**
# 
# 
# We'll keep 47 columns in total before getting to more features to be created (lag variables) and we proceed with further feature importance and removal, having into account Auto Regressive models (i.e. with the help of lagged values for the target variable to predict its self values).

# In[68]:


daily_df = daily_df[cols_to_keep]
daily_df


# # Time Series Forecast Modelling

# Now, its time for the modelling part given the data and the treatments we already have.
# 
# For the purpose of being a MultiVariate Time Series Regression model, I decided to explore one simpler ML model and a DL model:
# - Light GBM model
# - LSTM model

# ##LightGBM Model

# LightGBM (LGBM) is a gradient boosting framework that is known for its efficiency, speed, and accuracy in handling large-scale datasets.
# It is capable of capturing complex nonlinear relationships between the independent and target variables. For feature importance is also really helpful and we can set a ranking like we'll see more ahead.
# 
# **A test using an autoregressive lagged model (i.e. using the own target lagged variable to predict the future values of itself) will be made and, finally, a non AR model will be assessed as well to remove any possible bias and try to include as most as possible the influence of "Product_tag" and other columns so we can reach conclusions.**
# 
# 

# ###Autoregressive Model

# In[69]:


def time_delay_embedding(series, n_lags, horizon):

  if series.name is None:
      name = 'Series'
  else:
      name = series.name

  n_lags_iter = list(range(n_lags, -horizon, -1))

  df_list = [series.shift(i) for i in n_lags_iter]
  df = pd.concat(df_list, axis=1).dropna()

  df.columns = [f'{name}(t-{j - 1})'
                if j > 0 else f'{name}(t+{np.abs(j) + 1})'
                for j in n_lags_iter]

  df.columns = [re.sub('t-0', 't', x) for x in df.columns]

  return df


# In[70]:


def error_metrics(Y_ts, predictions):

  # Mean Squared Error (MSE)
  mse = mean_squared_error(Y_ts, predictions)

  # Root Mean Squared Error (RMSE)
  rmse = np.sqrt(mse)

  # Mean Absolute Error (MAE)
  mae = mean_absolute_error(Y_ts, predictions)

  # Mean Absolute Percentage Error (MAPE)
  mape = mean_absolute_percentage_error(Y_ts, predictions)

  print("Mean Squared Error (MSE):", mse)
  print("Root Mean Squared Error (RMSE):", rmse)
  print("Mean Absolute Error (MAE):", mae)
  print("Mean Absolute Percentage Error (MAPE):", mape)


# In[71]:


# horizon will be 6 months * 30 days = 180 days
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error as mape
from sklearn.multioutput import MultiOutputRegressor

from lightgbm import LGBMRegressor
from sklearn.model_selection import cross_val_score

def LGBM_model(df, n_lags, target_var='amortized_cost'):

  print('Nr of lags choosen:', n_lags, '\n')
  list_cols = []
  for col in df:
      col_df = time_delay_embedding(df[col], n_lags=n_lags, horizon=180)
      list_cols.append(col_df)

  # concatenating all variables and drop nulls
  new_df = pd.concat(list_cols, axis=1).dropna()

  # defining target (Y) and explanatory variables (X)
  predictor_variables = new_df.columns.str.contains('\(t\-')
  target_variables = new_df.columns.str.contains(f'{target_var}\(t\+')

  X = new_df.iloc[:, predictor_variables]
  Y = new_df.iloc[:, target_variables]

  # train/test split with 0.7/0.3 standard split
  X_tr, X_ts, Y_tr, Y_ts = train_test_split(X, Y, test_size=0.3, shuffle=False)

  # fitting a LGBM model without feature engineering
  model_wo_fe = MultiOutputRegressor(LGBMRegressor(random_state=0, n_estimators=100))
  model_wo_fe.fit(X_tr, Y_tr)

  # getting forecasts for the test set
  predictions = model_wo_fe.predict(X_ts)

  # computing all the error metrics
  error_metrics(Y_ts, predictions)

  print('------------------------', '\n')


# In[72]:


## see the minimum MAPE to choose the number of lags (choose between 1 day and 6 months to cover many possibilities)

target_var = 'amortized_cost'

lags = [1,7,14,30,60,90,180]

for lag in lags:
  LGBM_model(daily_df, lag)


# Therefore, upon these results, the best number of lags to chose seems to be 180 days, because its the one that minimizes the average percentage difference between the predicted and actual values (MAPE) and the Mean Abolute Error (MAE).
# 
# This value is in line on the graphs we saw above, that we could better capture a whole 6 months before trend to predict the next 6 months. The issues that we can have is that, first, we exploded many variables, as we create more lagged features and, second, we might be overfitting our model.**Therefore, we'll choose the 30 days lagged solution, which seem the most viable given the possible overfiting and overdimensionality issues.**
# 
# Next, let's see the most important features (top 100) and check if the autoregressive target  and other time variables had a big impact on the model and, overshadowed the other independent variables.

# In[73]:


## choosing the optimal nr of lag = 30
n_lags=30
target_var = 'amortized_cost'

list_cols = []
for col in daily_df:
    col_df = time_delay_embedding(daily_df[col], n_lags=n_lags, horizon=180)
    list_cols.append(col_df)

# concatenating all variables and drop nulls
new_df = pd.concat(list_cols, axis=1).dropna()

# defining target (Y) and explanatory variables (X)
predictor_variables = new_df.columns.str.contains('\(t\-')
target_variables = new_df.columns.str.contains(f'{target_var}\(t\+')

X = new_df.iloc[:, predictor_variables]
Y = new_df.iloc[:, target_variables]

# train/test split with 70/30 standard split
X_tr, X_ts, Y_tr, Y_ts = train_test_split(X, Y, test_size=0.3, shuffle=False)

# fitting a LGBM model without feature engineering
model_wo_fe = MultiOutputRegressor(LGBMRegressor(random_state=0, n_estimators=100))
model_wo_fe.fit(X_tr, Y_tr)

# getting forecasts for the test set
predictions = model_wo_fe.predict(X_ts)

# computing all the error metrics
error_metrics(Y_ts, predictions)


# In[74]:


new_df.shape


# In[75]:


# plot actual + forecast values
def plot_forecast(predictions, df_initial):
  forecast_values = np.mean(predictions, axis=0) # do average of predictions
  forecast_dates = pd.date_range(start=df_initial.index[-1] + pd.DateOffset(1), periods=len(predictions[1]), freq='D')

  figure = plt.figure(figsize=(10,10))
  plt.title('Amortized Cost per day')
  plt.plot(df_initial.index, df_initial['amortized_cost'], color='blue', label = 'Actual')
  plt.plot(forecast_dates, forecast_values, color='red', label='Forecast')
  plt.xlabel('Date')
  plt.ylabel('Cost')
  plt.legend()
  return plt.show()

plot_forecast(predictions, daily_df)


# Even though we see a big drop right in the first observation (like it happened twice in the past), we see a similar pattern of forecasted normalized values based mostly on more recent months.

# In[76]:


# getting the importance of each feature - top 100

avg_imp = pd.DataFrame([x.feature_importances_
                        for x in model_wo_fe.estimators_]).mean()

# getting the top 100 features
n_top_features = 100

importance_scores = pd.Series(dict(zip(X_tr.columns, avg_imp)))
top_features = importance_scores.sort_values(ascending=False)[:n_top_features]
top_features_nm = top_features.index

# subsetting training and testing sets by those features
X_tr_top = X_tr[top_features_nm]
X_ts_top = X_ts[top_features_nm]

# re-fitting the lgbm model
model_top_features = MultiOutputRegressor(LGBMRegressor(random_state=0, n_estimators=100))
model_top_features.fit(X_tr_top, Y_tr)

# getting forecasts for the test set
preds_top_feats = model_top_features.predict(X_ts_top)


# plot top 100 features


imp_df = importance_scores.sort_values(ascending=True)[-n_top_features:].reset_index()
imp_df.columns = ['Feature', 'Importance']
imp_df['Feature'] = pd.Categorical(imp_df['Feature'], categories=imp_df['Feature'])


plot = ggplot(imp_df, aes(x='Feature', y='Importance')) +        geom_bar(fill='#58a63e', stat='identity', position='dodge') +        theme_classic(
           base_size=10) + \
       theme(
           plot_margin=.25,
           axis_text=element_text(size=8),
           axis_text_x=element_text(size=8),
           axis_title=element_text(size=8),
           legend_text=element_text(size=8),
           legend_title=element_text(size=8),
           legend_position='top') + \
       xlab('') + \
       ylab('Importance') + coord_flip() + \
       theme(figure_size=(50, 50))

plot


# As seen, almost all important features include the own variable (with several T-X lags) and the effect of seasonality as well with the weekday and months lagged.
# The only most relevant variable here for our problem is the importance of the **Product Tag - Product 5**.
# 
# **In order to remove these bias and try to assess other important variables to forecast (even though it might produce a worse model), we'll do the same modelling but now removing the autoregressive target variable as well as the date columns.**

# ###Non AR Model

# In[77]:


daily_df_new = daily_df.copy()
daily_df_new = daily_df_new.drop(['month', 'quarter', 'weekday'], axis=1)


# In[78]:


## choosing the optimal nr of lag = 30
from sklearn.metrics import *
n_lags=30
target_var = 'amortized_cost'

list_cols = []
for col in daily_df_new:
    col_df = time_delay_embedding(daily_df_new[col], n_lags=n_lags, horizon=180)
    list_cols.append(col_df)

# concatenating all variables and drop nulls
new_df = pd.concat(list_cols, axis=1).dropna()

# defining target (Y) and explanatory variables (X)
# explanatory variables now shouldnt include the target

predictor_variables = (new_df.columns.str.contains('\(t\-')) & (~new_df.columns.str.contains(target_var))
target_variables = new_df.columns.str.contains(f'{target_var}\(t\+')

X = new_df.iloc[:, predictor_variables]
Y = new_df.iloc[:, target_variables]

# train/test split with 70/30 standard split
X_tr, X_ts, Y_tr, Y_ts = train_test_split(X, Y, test_size=0.3, shuffle=False)

# fitting a LGBM model without feature engineering
model_wo_fe = MultiOutputRegressor(LGBMRegressor(random_state=0, n_estimators=100))
model_wo_fe.fit(X_tr, Y_tr)

# getting forecasts for the test set
predictions = model_wo_fe.predict(X_ts)

# computing all the error metrics
error_metrics(Y_ts, predictions)


# In[79]:


# plot actual + forecast values

plot_forecast(predictions, daily_df_new)


# The same pattern applies here, so we can conclude that without the date and the own cost effect, **we have variables in our dataset good enough to predict the future costs!**

# In[80]:


# getting the importance of each feature - top 100

avg_imp = pd.DataFrame([x.feature_importances_
                        for x in model_wo_fe.estimators_]).mean()

n_top_features = 100

importance_scores = pd.Series(dict(zip(X_tr.columns, avg_imp)))
top_features = importance_scores.sort_values(ascending=False)[:n_top_features]
top_features_nm = top_features.index

# subsetting training and testing sets by those features
X_tr_top = X_tr[top_features_nm]
X_ts_top = X_ts[top_features_nm]

# re-fitting the lgbm model
model_top_features = MultiOutputRegressor(LGBMRegressor(random_state=0, n_estimators=100))
model_top_features.fit(X_tr_top, Y_tr)

# getting forecasts for the test set
preds_top_feats = model_top_features.predict(X_ts_top)


# plot top 100 features

imp_df = importance_scores.sort_values(ascending=True)[-n_top_features:].reset_index()
imp_df.columns = ['Feature', 'Importance']
imp_df['Feature'] = pd.Categorical(imp_df['Feature'], categories=imp_df['Feature'])


plot = ggplot(imp_df, aes(x='Feature', y='Importance')) +        geom_bar(fill='#58a63e', stat='identity', position='dodge') +        theme_classic(
           base_size=10) + \
       theme(
           plot_margin=.25,
           axis_text=element_text(size=8),
           axis_text_x=element_text(size=8),
           axis_title=element_text(size=8),
           legend_text=element_text(size=8),
           legend_title=element_text(size=8),
           legend_position='top') + \
       xlab('') + \
       ylab('Importance') + coord_flip() + \
       theme(figure_size=(50, 50))

plot


# Now there's more balance in terms of variable importance, not only in product, but also operations related. Different lags have influence as well and in mixed fashion.

# Next, let's experiment a Deep Learning model to complement or possibly change this analysis and assess which model performs best, so that conclusions can be taken on a best accurate cost forecast and which were the main drivers of this spend increase, so that the company can focus mostly on some specific usages.

# ## LSTM Model

# LSTM (Long Short-Term Memory) is a type of recurrent neural network (RNN) architecture that has several advantages for data modelling, including time series forecasting. It is very good to capture long term dependencies and that is the case for our dataset where we have to predict 6 months ahead based on previous 1 and half years (in days). It also deals well with noisy data. The issue that can arise here is that we can add more complexity than what is needed for the model (as the LGBM already performed quite well).
# 
# **We'll experiment using only a non AutoRegressive model to compare with the latter LGBM.**

# In[81]:


# import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import datetime as dt
import itertools
import random

from sklearn.preprocessing import MinMaxScaler


def summary(df):
  desc = df.describe(include = 'all').T.rename(columns={"count": "count_non_nulls"})
  desc['count_nulls'] = df.shape[0]-desc['count_non_nulls']
  new_index_list = ['count_non_nulls', 'count_nulls']
  new_index=new_index_list + [c for c in desc.columns if c not in new_index_list]
  desc = desc.reindex(new_index, axis='columns')

  return desc


# In[82]:


# import libraries
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import numpy as np
import datetime as dt
import itertools
import random
import re

from sklearn.metrics import *
from sklearn.preprocessing import MinMaxScaler
from plotnine import *
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_absolute_percentage_error as mape
from sklearn.multioutput import MultiOutputRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import cross_val_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
#import shap


# In[83]:


def time_delay_embedding(series, n_lags, horizon):

  if series.name is None:
      name = 'Series'
  else:
      name = series.name

  n_lags_iter = list(range(n_lags, -horizon, -1))

  df_list = [series.shift(i) for i in n_lags_iter]
  df = pd.concat(df_list, axis=1).dropna()

  df.columns = [f'{name}(t-{j - 1})'
                if j > 0 else f'{name}(t+{np.abs(j) + 1})'
                for j in n_lags_iter]

  df.columns = [re.sub('t-0', 't', x) for x in df.columns]

  return df


# In[84]:


def error_metrics(Y_ts, predictions):

  # Mean Squared Error (MSE)
  mse = mean_squared_error(Y_ts, predictions)

  # Root Mean Squared Error (RMSE)
  rmse = np.sqrt(mse)

  # Mean Absolute Error (MAE)
  mae = mean_absolute_error(Y_ts, predictions)

  # Mean Absolute Percentage Error (MAPE)
  mape = mean_absolute_percentage_error(Y_ts, predictions)

  print("Mean Squared Error (MSE):", mse)
  print("Root Mean Squared Error (RMSE):", rmse)
  print("Mean Absolute Error (MAE):", mae)
  print("Mean Absolute Percentage Error (MAPE):", mape)


# In[85]:


# import and see nb rows and columns of the input dataset
daily_df_new = pd.read_excel('/content/drive/MyDrive/daily_df_new.xlsx')
daily_df_new = daily_df_new.set_index('usage_date')
daily_df_new.shape


# ###Non AR Model

# In[86]:


n_lags=30
target_var = 'amortized_cost'

list_cols = []
for col in daily_df_new:
    col_df = time_delay_embedding(daily_df_new[col], n_lags=n_lags, horizon=180)
    list_cols.append(col_df)

# concatenating all variables and drop nulls
new_df = pd.concat(list_cols, axis=1).dropna()

# defining target (Y) and explanatory variables (X)
# explanatory variables now shouldnt include the target

predictor_variables = (new_df.columns.str.contains('\(t\-')) & (~new_df.columns.str.contains(target_var))
target_variables = new_df.columns.str.contains(f'{target_var}\(t\+')

X = new_df.iloc[:, predictor_variables]
Y = new_df.iloc[:, target_variables]

# split the same way into train test
X_tr, X_ts, Y_tr, Y_ts = train_test_split(X, Y, test_size=0.3, shuffle=False, random_state=0)

n_features = X_tr.shape[1]
X_tr_reshaped = X_tr.to_numpy().reshape(-1, 1, n_features)
X_ts_reshaped = X_ts.to_numpy().reshape(-1, 1, n_features)

# Build the LSTM model
model = Sequential()
model.add(LSTM(64, input_shape=(1, n_features)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
history = model.fit(X_tr_reshaped, Y_tr, epochs=100, batch_size=32, validation_data=(X_ts_reshaped, Y_ts))

# Evaluate the model
train_loss = model.evaluate(X_tr_reshaped, Y_tr)
test_loss = model.evaluate(X_ts_reshaped, Y_ts)
print("Train Loss:", train_loss)
print("Test Loss:", test_loss)

# Make predictions
train_predictions = model.predict(X_tr_reshaped)
test_predictions = model.predict(X_ts_reshaped)

# Get the loss values from the history object
train_loss = history.history['loss']
test_loss = history.history['val_loss']

# Plot the loss values
plt.plot(train_loss, label='Train Loss')
plt.plot(test_loss, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()


# In[87]:


# now the same but with less epochs since the increase doesnt help to decrease the error anymore
n_lags=30
target_var = 'amortized_cost'

list_cols = []
for col in daily_df_new:
    col_df = time_delay_embedding(daily_df_new[col], n_lags=n_lags, horizon=180)
    list_cols.append(col_df)

# concatenating all variables and drop nulls
new_df = pd.concat(list_cols, axis=1).dropna()

# defining target (Y) and explanatory variables (X)
# explanatory variables now shouldnt include the target

predictor_variables = (new_df.columns.str.contains('\(t\-')) & (~new_df.columns.str.contains(target_var))
target_variables = new_df.columns.str.contains(f'{target_var}\(t\+')

X = new_df.iloc[:, predictor_variables]
Y = new_df.iloc[:, target_variables]

# split the same way into train test
X_tr, X_ts, Y_tr, Y_ts = train_test_split(X, Y, test_size=0.3, shuffle=False, random_state=0)

n_features = X_tr.shape[1]
X_tr_reshaped = X_tr.to_numpy().reshape(-1, 1, n_features)
X_ts_reshaped = X_ts.to_numpy().reshape(-1, 1, n_features)

# Build the LSTM model
model = Sequential()
model.add(LSTM(64, input_shape=(1, n_features)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_tr_reshaped, Y_tr, epochs=20, batch_size=32, validation_data=(X_ts_reshaped, Y_ts))

# Train the model
history = model.fit(X_tr_reshaped, Y_tr, epochs=20, batch_size=32, validation_data=(X_ts_reshaped, Y_ts))

# Evaluate the model
train_loss = model.evaluate(X_tr_reshaped, Y_tr)
test_loss = model.evaluate(X_ts_reshaped, Y_ts)
print("Train Loss:", train_loss)
print("Test Loss:", test_loss)

# Make predictions
train_predictions = model.predict(X_tr_reshaped)
test_predictions = model.predict(X_ts_reshaped)

# Get the loss values from the history object
train_loss = history.history['loss']
test_loss = history.history['val_loss']

# Plot the loss values
plt.plot(train_loss, label='Train Loss')
plt.plot(test_loss, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()


# As seen above, the model learned quite quickly its pattern and we don't gain much information/model improvement, when we iterate over more than **20 epochs**. The error seems relatively low already and slightly decreasing in the train and in the test as well, which is a good indicator.
# 
# However, they have quite some gap between them, which can mean that the model is slightly overfitted and, as we saw before, this is no surprise, as we still have highly correlated features and further PCA or feature reduction could have been done in order to avoid this.
# 
# Nevertheless, this model seems to be an improvement over the LGBM due to its MSE metric being lower for the test set (0.0221 vs 0.0492), even though we lose interpretability using this method.

# In[90]:


train_forecast_avg = np.mean(train_predictions, axis=1)
test_forecast_avg = np.mean(test_predictions, axis=1)

train_time_index = Y_tr.index
test_time_index = Y_ts.index

# Plot the actual values
plt.plot(train_time_index, np.mean(Y_tr, axis=1), label='Actual Train')
plt.plot(test_time_index, np.mean(Y_ts, axis=1), label='Actual Test')

# Plot the average forecasted values
plt.plot(train_time_index, train_forecast_avg, label='Forecast Train (Avg)')
plt.plot(test_time_index, test_forecast_avg, label='Forecast Test (Avg)')


plt.xlabel('Time')
plt.ylabel('Value')
plt.title('Actual vs. Forecast (Average)')


plt.legend()
plt.show()


# Finally, we'll see the most relevant features in the used models to complement the analysis.
# 
# Since this is a black box model with a lot of complex Neural Network relationships, then it's more difficult to do this analysis. A Shapley additive explanation method was tried to assess feature importance for this LSTM but without success, thus, **let's evaluate feature importance in the LGM model:**

# In[91]:


# plot again feature importance for the LGBM model
plot = ggplot(imp_df, aes(x='Feature', y='Importance')) +        geom_bar(fill='#58a63e', stat='identity', position='dodge') +        theme_classic(
           base_size=10) + \
       theme(
           plot_margin=.25,
           axis_text=element_text(size=8),
           axis_text_x=element_text(size=8),
           axis_title=element_text(size=8),
           legend_text=element_text(size=8),
           legend_title=element_text(size=8),
           legend_position='top') + \
       xlab('') + \
       ylab('Importance') + coord_flip() + \
       theme(figure_size=(50, 50))

plot


# As we can see, **The Product 5 labeled in Product Tag** is the most important feature (without accounting for the own cost feature and with different time lags. Moreover, **Send and List Subsriptions Operations** and **different Product Codes** (like Amazon MSK or "Others"), also contributte fairly well for our time series model.

# **Note: Both of these models can be further exploited with hyperparameter fine tuning in order to have a better overall model and more adapted to this data. Further ML and DL models can be studeid and added as well.**

# #Unsupervised Learning - Clustering

# To answer to the second question of prioritization for optimization efforts (in terms of AWS service and/or infrastructure components), an unsupervised method was used, in order to group sets of data depending on the averages spend, in order to check for patterns in data.
# 
# For simplicity and good understandability, a **K-Means** algorithm was used in order to complement our data analysis and understand the main drivers of spend for the client.

# ##K-means Algoritm

# In[92]:


import warnings
warnings.filterwarnings("ignore")

from sklearn.cluster import KMeans

df_cluster = daily_df_new.copy()

min_clusters = 1
max_clusters = 10

wcss = []

# Perform K-means clustering for different numbers of clusters
for n_clusters in range(min_clusters, max_clusters+1):
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(df_cluster)
    wcss.append(kmeans.inertia_)


plt.plot(range(min_clusters, max_clusters+1), wcss, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.title('Elbow Method - Determining Number of Clusters')
plt.show()


# One good method to determine the optimal number of clusters is the **elbow method**, which is based on the sum of squared distance (SSE) between data points and their assigned clusters’ centroids. To find this number of clusters, we observe the elbow graphs and find more or less what is the position of the "elbow" formed by this line plot.
# 
# 
# **In this case, the optimal seem to be 3 groups of clusters and that's what we are going to choose!**

# In[93]:


X = daily_df_new.iloc[:,:-1]

# Define the number of clusters = 3
n_clusters = 3

# Initialize and fit the K-means clustering model
kmeans = KMeans(n_clusters=n_clusters, random_state=0)
kmeans.fit(X)

# Get the cluster labels for each data point
cluster_labels = kmeans.labels_

df_cluster['cluster'] = cluster_labels

for cluster in range(n_clusters):
    cluster_data = df_cluster[df_cluster['cluster'] == cluster]
    print(f"Cluster {cluster+1} - Size: {len(cluster_data)}")


# In[94]:


cluster_summary = df_cluster.groupby('cluster')['amortized_cost'].mean()

print(cluster_summary)


# In[95]:


# Calculate aggregated statistics for the target variable based on each cluster
cluster_stats = df_cluster.groupby('cluster')['amortized_cost'].agg(['mean', 'median', 'min', 'max'])


cluster_stats.plot(kind='bar', figsize=(10, 6))
plt.xlabel('Cluster')
plt.ylabel('Amortized Cost')
plt.title('Cluster Statistics - Amortized Cost')
plt.legend(loc='upper right')
plt.show()


# In[96]:


df_cluster.to_excel('cluster.xlsx')


# Here we can see well defined all 3 groups of data where we have **low average costs (Cluster 0), medium costs (cluster 1) and high average costs (Cluster 2)**. Let's see per variable, which one had the most importance.

# In[97]:


# Calculate the average of each column within each cluster
cluster_averages = df_cluster.groupby('cluster').mean()

# Exclude the cluster and cost columns
columns = df_cluster.columns[:-2]

# Calculate the number of rows for the subplot grid
n_cols = 4
n_rows = math.ceil(len(columns)/n_cols)


fig, axes = plt.subplots(n_rows, n_cols, figsize=(80, 100))

# Iterate over each column and plot the clusters
for i, column in enumerate(columns):
    # Calculate the subplot position
    row = int(i / n_cols)
    col = i % n_cols

    # Create the subplot
    ax = axes[row, col] if n_rows > 1 else axes[col]

    # Plot the average 'amortized_cost' per cluster and column
    sns.scatterplot(data=df_cluster, x=column, y='amortized_cost', hue='cluster', palette='viridis', ax=ax, alpha=0.5)
    sns.scatterplot(data=cluster_averages, x=column, y='amortized_cost', marker='X', color='red', ax=ax)

    ax.set_title(f'Average Cost per Cluster - {column}')
    ax.set_xlabel(column)
    ax.set_ylabel('Amortized Cost')
    ax.legend().remove()

plt.tight_layout()
plt.show()


# In[98]:


# Now, filter only when cluster group = 0 (where we have the most average costs) and see top average of attributes
# Also, here is more visible the top important variables:

df_cluster_agg = df_cluster[df_cluster['cluster'] == 0].groupby('cluster').mean().T.sort_values(0, ascending=False)
df_cluster_agg


# By the analysis of above graphs and the aggregate table, **we conclude that both client’s infrastructure components (product (tag) & environment (tag)) and the AWS service information** are key drivers for the increase in costs that's been happening.
# 
# On average, certain operations like "List Subscriptions", "create snapshots" and "VPN connections" are costly for the client. Other operations of course are included.
# 
# **In terms of infrustructure, we see products 4, 6 and 5 as the most relevant regrding the most costs increase and the UAT environment is often related with mosts costs as well. All of these are mostly related with the usage as well, because we know that the higher the usage in these products or services, the higher will be the costs. Thus, the client should focus on spending less resources in these fields and optimize both product and AWS service usage to control expenses.**

# # Conclusions

# **To sum up this exercise, we analysed and concluded that the client should:**
# 
# 
# *   Focus on both AWS service and infrastructure usage amounts, to control better its costs.
# *   Special focus on controlling Operations such as Listing Subscriptions, Creating Snapshots of data and providing VPN acesses
# 
# 
# *   Regarding the Product and environment side: Products 4,6,5 and UAT working env should be prioritized as these are usually associated with high costs per usage or high usages
# 
# 
# 
# *   In terms of forecasting costs, we discovered that the main drivers on the product side is clearly the Product 5 (with different time lags), but also other Product Codes and "Send" Operations are big drivers when we want to forecast future costs based on the past data we have available.

# # References

# 
# *   https://www.activestate.com/blog/top-10-tools-for-hyperparameter-optimization-in-python/
# 
# *   https://dataconomy.com/2022/11/25/time-series-forecasting-machine-learning/
# 
# 
# *   https://machinelearningmastery.com/clustering-algorithms-with-python/
# 
# *   https://github.com/austinnottexas/ConvolutionalCloudWorkloadAnalysis
# 
# *   https://docs.aws.amazon.com/cur/latest/userguide/data-dictionary.html
# 
# *   https://towardsdatascience.com/11-dimensionality-reduction-techniques-you-should-know-in-2021-dcb9500d388b
# *   https://machinelearningmastery.com/data-leakage-machine-learning/
# 
# 
# *   https://aws.amazon.com/pt/route53/faqs/
# 
# 
# *   https://www.kaggle.com/code/vprokopev/mean-likelihood-encodings-a-comprehensive-study/notebook
# 
# 
# *   https://blog.paperspace.com/time-series-forecasting-regression-and-lstm/
# 
# *   https://blog.paperspace.com/time-series-forecasting-autoregressive-models-smoothing-methods/
# 
# *   https://machinelearningmastery.com/feature-selection-time-series-forecasting-python/
# 
# *   https://shap.readthedocs.io/en/latest/generated/shap.Explainer.html
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 

# In[ ]:




