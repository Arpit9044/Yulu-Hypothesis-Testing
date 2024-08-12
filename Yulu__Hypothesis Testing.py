#!/usr/bin/env python
# coding: utf-8

# In[1380]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind,f_oneway,levene,kruskal,shapiro,chi2_contingency

import warnings
warnings.filterwarnings("ignore")


# In[1381]:


df=pd.read_csv(r"C:\Users\hp\Downloads\Yulu.csv")


# In[1382]:


df


# In[1383]:


df.shape


# In[1384]:


df.info()


# In[1385]:


df.describe()


# In[1386]:


df[df.duplicated()]


# In[1387]:


df.isna().sum().sum()


# # Outliers Detection and Removal from the target variable

# In[1388]:


sns.boxplot(y=df['count'])


# In[1389]:


#As we can see there are a lot of outliers in the data
#Outliers Detection::
IQR=np.percentile(df['count'],75)-np.percentile(df['count'],25)
upper_whisker=np.percentile(df['count'],75)+(1.5*IQR)
df[df['count']>upper_whisker]


# In[1390]:


#We are removing the outliers as there only  300 out of 11k Rows
df=df[df['count']<upper_whisker]
sns.boxplot(y=df['count'])


# # Basic Analysis

# In[1391]:


df


# In[1392]:


plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
sns.histplot(df['temp'])
plt.subplot(1,2,2)
sns.boxplot(y=df['temp'])
plt.show()


# In[1393]:


plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
sns.histplot(df['atemp'])
plt.subplot(1,2,2)
sns.boxplot(y=df['atemp'])
plt.show()


# In[1394]:


plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
sns.histplot(df['humidity'],bins=15)
plt.subplot(1,2,2)
sns.boxplot(y=df['humidity'])
plt.show()


# In[1395]:


plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
sns.histplot(df['windspeed'])
plt.subplot(1,2,2)
sns.boxplot(y=df['windspeed'])
plt.show()


# In[1396]:


plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
sns.histplot(df['casual'])
plt.subplot(1,2,2)
sns.boxplot(y=df['casual'])
plt.show()


# In[1397]:


plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
sns.histplot(df['registered'])
plt.subplot(1,2,2)
sns.boxplot(y=df['registered'])
plt.show()


# In[1398]:


plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
sns.histplot(df['count'])
plt.subplot(1,2,2)
sns.boxplot(y=df['count'])
plt.show()


# In[1399]:


sns.countplot(x=df['season'])


# In[1400]:


sns.countplot(x=df['workingday'])


# In[1401]:


sns.countplot(x=df['weather'])


# In[1402]:


df['weather'].value_counts()


# In[1403]:


#As weather 4 has only 1 data point, we will remove this
df=df[df['weather']!=4]


# In[1404]:


df['weather'].value_counts()


# In[1405]:


df.corr()


# In[1406]:


plt.figure(figsize=(12,7))
sns.heatmap(df.corr(),annot=True)
plt.show()


# In[1407]:


#'temp' and 'atemp' are highly correlated, we will remove atemp.
# Similarly 'Casual' and 'registered are also highly correlated with the target variable.'
df.drop(['atemp','casual','registered'],axis=1,inplace=True)
df


# # Working day effect on number of bikes used.

# In[1408]:


df


# In[1409]:


a=df.groupby('workingday')['count'].mean()
a


# In[1410]:


sns.barplot(x=a.index,y=a.values)
plt.show()


# In[1411]:


sns.histplot(df['count'])


# In[1412]:


#As we can see that this distribution does not look normal
#Let us test for the same
from statsmodels.graphics.gofplots import qqplot
qqplot(df['count'],line='s')
plt.show()


# - We can see that the data is not purely gaussian in nature but still we will go ahead with the test.

# In[1413]:


#Let us check this with the help of shapiro test as well
#Ho---->Data is Gaussian
#Ha---->Data is not Gaussian
p_value=shapiro(df['count'])[1]


# In[1414]:


if p_value<alpha:
     print('Reject Null Hypothesis')
else:
    print('Fail to reject Null Hypothesis')    


# - Shapiro also shows that the data is not gaussian

# In[1455]:


#As we can see there is a clear difference between the demand of bikes on working and non-working day.
#Let us check this through hypothesis testing
#Ho - working_day_count >= non_working_day_count 
#Ha - working_day_count < non_working_day_count

#Preparing the two samples for ttest :
working_day_count=df.loc[df['workingday']==1,'count'].sample(2999)
non_working_day_count=df.loc[df['workingday']==0,'count'].sample(2999)

#performing the two sample ttest
t_stat,p_value=ttest_ind(working_day_count,non_working_day_count,alternative='less')
print(p_value)


#Comparing p_value with significance level
alpha=0.05
if p_value<alpha:
    print('Reject Ho')
else:
    print('Fail to reject Ho')


# - So, On non-working days, no of bike rides are higher than that of working day.

# # Season effect on number of bikes used.

# In[1416]:


df


# In[1417]:


a=df.groupby('season')['count'].mean()
a


# In[1418]:


sns.barplot(x=a.index,y=a.values)


# - We can see that there are four different seasons i.e. we will get 4 different sample and perform Anova test to check whether the variance of all the four samples are equal or not.

# In[1419]:


samp1=df.loc[df['season']==1,'count'].sample(1000) #No of bikes used when season is 1
samp2=df.loc[df['season']==2,'count'].sample(1000)  #No of bikes used when season is 2
samp3=df.loc[df['season']==3,'count'].sample(1000)  #No of bikes used when season is 3
samp4=df.loc[df['season']==4,'count'].sample(1000)  #No of bikes used when season is 4


# In[1420]:


# Before appling anova test, let us check whether the variances of all the samples are equal
#Ho--->all the samples have equal variances
#Ha--->One or more sample has different variance
p_value=levene(samp1,samp2,samp3,samp4)[1]
p_value


# In[1421]:


if p_value < alpha:
    print('Reject Null Hypothesis')
else:
    print('Fail to reject Null Hypothesis')


# We can see that the variances of these samples are not equal but we will go ahead with ANOVA.

# In[1422]:


# Ho- All samples have equal mean.
# Ha- One or more sample has different mean.
f_stat,p_value=f_oneway(samp1,samp2,samp3,samp4)
print(f_stat,p_value)


# In[1423]:


if p_value < alpha:
    print('Reject Null Hypothesis')
else:
    print('Fail to reject Null Hypothesis')


# We will also apply the Kruskal Wallis Test as the variances are not equal.

# In[1424]:


p_value=kruskal(samp1,samp2,samp3,samp4)[1]
p_value


# In[1425]:


if p_value < alpha:
    print('Reject Null Hypothesis')
else:
    print('Fail to reject Null Hypothesis')


# This also suggests that one or more sample is having their mean diiferent.

# Let us deep dive and perform multiple 2- Sample ttest to compare two samples at atime

# In[1426]:


print(samp1.mean(),samp2.mean(),samp3.mean(),samp4.mean())


# In[1427]:


print(samp1.mean(),samp2.mean())


# In[1428]:


#Ttest between sample 1 and sample 2
#We see that samp2 mean is higher than that of samp1.Let us see whether this is statistically significant or not.
# Ho -----> bikes used in season1 >= bikes used in season2
# Ha ----->  bikes used in season1  < bikes used in season2
t_stat,p_value=ttest_ind(samp1,samp2,alternative='less')
print(t_stat,p_value)


# In[1429]:


if p_value < alpha:
    print('Reject Null Hypothesis')
else:
    print('Fail to reject Null Hypothesis')


# So,the conclusion is that this difference is statistically significant i.e. in season2 customers order more bikes than season1

# In[1430]:


print(samp1.mean(),samp3.mean())


# In[1431]:


#Ttest between sample 1 and sample 3
#We see that samp3 mean is higher than that of samp1.Let us see whether this is statistically significant or not.
# Ho -----> bikes used in season1 >= bikes used in season3
# Ha ----->  bikes used in season1  < bikes used in season3
t_stat,p_value=ttest_ind(samp1,samp3,alternative='less')
print(t_stat,p_value)


# In[1432]:


if p_value < alpha:
    print('Reject Null Hypothesis')
else:
    print('Fail to reject Null Hypothesis')


# So,the conclusion is that this difference is statistically significant i.e. in season2 customers order more bikes than season1

# In[1433]:


print(samp1.mean(),samp4.mean())


# In[1434]:


#Ttest between sample 1 and sample 4
#We see that samp4 mean is higher than that of samp1.Let us see whether this is statistically significant or not.
# Ho -----> bikes used in season1 >= bikes used in season4
# Ha ----->  bikes used in season1  < bikes used in season4
t_stat,p_value=ttest_ind(samp1,samp4,alternative='less')
print(t_stat,p_value)


# In[1435]:


if p_value < alpha:
    print('Reject Null Hypothesis')
else:
    print('Fail to reject Null Hypothesis')


# So,the conclusion is that this difference is statistically significant i.e. in season4 customers order more bikes than season1
# - Therefore, season1 has the lowest number of customers that have avalied the bike.

# In[1436]:


print(samp2.mean(),samp3.mean())


# In[1437]:


#Ttest between sample 2 and sample 3
#We see that samp3 mean is higher than that of samp2.Let us see whether this is statistically significant or not.
# Ho -----> bikes used in season2 >= bikes used in season3
# Ha ----->  bikes used in season2  < bikes used in season3
t_stat,p_value=ttest_ind(samp2,samp3,alternative='less')
print(t_stat,p_value)


# In[1438]:


if p_value < alpha:
    print('Reject Null Hypothesis')
else:
    print('Fail to reject Null Hypothesis')


# So,the conclusion is that this difference is statistically significant i.e. in season3 customers order more bikes than season2

# In[1439]:


print(samp2.mean(),samp4.mean())


# In[1440]:


#Ttest between sample 2 and sample 4
#We see that samp2 mean is higher than that of samp1.Let us see whether this is statistically significant or not.
# Ho -----> bikes used in season4 >= bikes used in season2
# Ha ----->  bikes used in season4  < bikes used in season2
t_stat,p_value=ttest_ind(samp2,samp4,alternative='greater')
print(t_stat,p_value)


# In[1441]:


if p_value < alpha:
    print('Reject Null Hypothesis')
else:
    print('Fail to reject Null Hypothesis')


# So,the conclusion is that this difference is statistically significant i.e. in season2 customers order more bikes than season4

# - So this is the decreasing order of demand or number of customers booking the bike as per the seasons:
#     - fall
#     - summer
#     - winter
#     - spring
#     
# Therefore, in fall season the number of bikes availed is the most and in spring it's the least.

# # Holiday effect on number of bikes used.

# In[1461]:


a=df.groupby('holiday')['count'].mean()
a


# In[1462]:


sns.barplot(a.index,a.values)


# In[1463]:


no_holiday=df.loc[df['holiday']==0,'count'].sample(299)
holiday=df.loc[df['holiday']==1,'count'].sample(299)
print(holiday.mean(),no_holiday.mean())


# In[1464]:


# Ho----> bikes used on holidays <= bikes used on non-holidays
# Ha----> bikes used on holidays > bikes used on non-holidays
#As there are two samples, we will go ahead with the ttest
t_stat,p_value=ttest_ind(no_holiday,holiday,alternative='less')
print(t_stat,p_value)


# In[1465]:


if p_value < alpha:
    print('Reject Null Hypothesis')
else:
    print('Fail to reject Null Hypothesis')


# - There is no effect of holiday on number of bikes availed.

# # weather effect on number of bikes used.

# In[1125]:


a=df.groupby('weather')['count'].mean()
a


# In[1126]:


sns.barplot(a.index,a.values)


# In[1127]:


samp1=df.loc[df['weather']==1,'count'].sample(500) #No of bikes used when weather is 1
samp2=df.loc[df['weather']==2,'count'].sample(500) #No of bikes used when weather is 2
samp3=df.loc[df['weather']==3,'count'].sample(500) #No of bikes used when weather is 3


# In[1128]:


# Before appling anova test, let us check whether the variances of all the samples are equal
#Ho--->all the samples have equal variances
#Ha--->One or more sample has different variance
p_value=levene(samp1,samp2,samp3)[1]
p_value


# In[1129]:


if p_value < alpha:
    print('Reject Null Hypothesis')
else:
    print('Fail to reject Null Hypothesis')


# - We will still go ahead with the ANOVA even after the variances are not equal

# In[1130]:


# Ho- All samples have same mean.
# Ha- One or more sample has different mean.
f_stat,p_value=f_oneway(samp1,samp2,samp3)
print(f_stat,p_value)


# In[1131]:


if p_value < alpha:
    print('Reject Null Hypothesis')
else:
    print('Fail to reject Null Hypothesis')


# We will also apply the Kruskal Wallis Test as the variances are not equal.

# In[1132]:


p_value=kruskal(samp1,samp2,samp3)[1]
p_value


# In[1133]:


if p_value < alpha:
    print('Reject Null Hypothesis')
else:
    print('Fail to reject Null Hypothesis')


# This also suggests that one or more sample is having their mean diiferent.

# Let us deep dive and perform multiple 2- Sample ttest to compare two samples at atime

# In[1134]:


print(samp1.mean(),samp2.mean(),samp3.mean())


# In[1135]:


print(samp1.mean(),samp2.mean())


# In[1136]:


#Ttest between sample 1 and sample 2
#We see that samp1 mean is higher than that of samp1.Let us see whether this is statistically significant or not.
# Ho -----> bikes used in weather1 <= bikes used in weather2
# Ha ----->  bikes used in weather1 > bikes used in weather1
t_stat,p_value=ttest_ind(samp1,samp2,alternative='greater')
print(t_stat,p_value)


# In[1137]:


if p_value < alpha:
    print('Reject Null Hypothesis')
else:
    print('Fail to reject Null Hypothesis')


# So,the conclusion is that this difference is statistically significant i.e. in weather1 customers order more bikes than weather2

# In[1138]:


print(samp1.mean(),samp3.mean())


# In[1139]:


#Ttest between sample 1 and sample 3
#We see that samp1 mean is higher than that of samp1.Let us see whether this is statistically significant or not.
# Ho -----> bikes used in weather1 <= bikes used in weather3
# Ha ----->  bikes used in weather1  > bikes used in weather3
t_stat,p_value=ttest_ind(samp1,samp3,alternative='greater')
print(t_stat,p_value)


# In[1140]:


if p_value < alpha:
    print('Reject Null Hypothesis')
else:
    print('Fail to reject Null Hypothesis')


# So,the conclusion is that this difference is statistically significant i.e. in weather1 customers order more bikes than weather3

# In[1141]:


print(samp2.mean(),samp3.mean())


# In[1142]:


#Ttest between sample 2 and sample 3
#We see that samp2 mean is higher than that of samp3.Let us see whether this is statistically significant or not.
# Ho -----> bikes used in weather2 <= bikes used in weather3
# Ha ----->  bikes used in weather2 > bikes used in weather3
t_stat,p_value=ttest_ind(samp2,samp3,alternative='greater')
print(t_stat,p_value)


# In[1143]:


if p_value < alpha:
    print('Reject Null Hypothesis')
else:
    print('Fail to reject Null Hypothesis')


# So,the conclusion is that this difference is statistically significant i.e. number of bikes booked in weather 2 is greater than that of weather 3

# - So this is the decreasing order of demand or number of customers booking the bike as per the weather:
#     - Clear, Few clouds, partly cloudy
#     - Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist
#     - Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds
#     - Heavy Rain + Ice Pellets + Thunderstorm + Mist, Snow + Fog
#     
# Therefore, in "Clear, Few clouds, partly cloudy season", the number of bikes availed is the most and in "Heavy Rain + Ice Pellets + Thunderstorm + Mist, Snow + Fog" it's the least.

# # Effect of Temperature on Number of Bikes Booked

# In[1144]:


sns.scatterplot(df['temp'],df['count'])


# In[1145]:


#As data does not seem linear, we will apply spearman correlation test.
from scipy.stats import spearmanr
spearmanr(df['temp'],df['count'])


# In[1146]:


print(df['temp'].max(),df['temp'].min(),df['temp'].mean())


# - Looking at the range of temperature and the mean, it seems like the location to which this data belongs to a cold place and people like to travel with bike when temperature ison a relatively higher side.
# 
# - Hence,there is a positive correlation between temp and number of bikes booked.

# In[1147]:


sns.histplot(df['temp'],bins=5)


# In[1148]:


#Let's divide the temperature in 5 bins and then see which kind of temperature shows high demand of bike
df['temp_bins']=pd.cut(df['temp'],bins=5,labels=['very_low','low','medium','high','very_high'])
df


# In[1149]:


df['temp_bins'].value_counts()


# In[1165]:


a=df.groupby('temp_bins')['count'].mean()
a


# In[1166]:


sns.barplot(a.index,a.values)


# - When the temperature is above medium the demand of the bikes increase
# - Quite afew customers also rent bike in low temperatures.

# # Effect of Humidity on Number of Bikes Booked

# In[1466]:


sns.scatterplot(df['humidity'],df['count'])


# In[1467]:


#As data does not seem linear, we will apply spearman correlation test.
from scipy.stats import spearmanr
spearmanr(df['humidity'],df['count'])


# In[1468]:


print(df['humidity'].max(),df['humidity'].min(),df['humidity'].mean())


# - Hence,there is a negative correlation between humidity and number of bikes booked.

# In[1469]:


sns.histplot(df['humidity'],bins=5)


# In[1473]:


#Let's divide the temperature in 5 bins and then see which kind of temperature shows high demand of bike
df['humidity_bins']=pd.cut(df['humidity'],bins=3,labels=['low','medium','high'])
df


# In[1474]:


df['humidity_bins'].value_counts()


# In[1476]:


a=df.groupby('humidity_bins')['count'].mean()
a


# In[1477]:


sns.barplot(a.index,a.values)


# - When the humidity is above medium the demand of the bikes is high.
# - As the humidity increases, bikes rented decrease 

# # Effect of windspeed on Number of Bikes Booked

# In[1155]:


sns.scatterplot(df['windspeed'],df['count'])


# In[1156]:


from scipy.stats import spearmanr
spearmanr(df['windspeed'],df['count'])


# In[1157]:


print(df['windspeed'].max(),df['windspeed'].min(),df['windspeed'].mean())


# - There is a low positive correlation between windspeed and number of bikes rented.

# # Effect of Weather on Season

# In[1158]:


contingency=pd.crosstab(df['weather'],df['season'])
contingency


# In[1159]:


#Ho---->There is no realtion between weather and season
#Ha---->There is a relationship between weather and season
p_value=chi2_contingency(contingency)[1]
p_value


# In[1160]:


if p_value < alpha:
    print('Reject Null Hypothesis')
else:
    print('Fail to reject Null Hypothesis')


# - There is a realtionship between weather and season i.e both are dependent variables.

# # Recommendations:
# 
# ### 1) Weather 4 is extremely hard and unfavourable to travelling.Also,there is only 1 data point that corresponds to weather 4 so it  should not be at the company's focus at all.
# 
# ### 2) Yulu should focus mainly on weather 1 and 2 because that is where the maximum bikes are rented.
# 
# ### 3) Company should focus on the different season in this decreasing order:
#   ###       - fall          
#   ###       - summer
#   ###       - winter
#   ###       - spring
#      
# ### 4)The demand of bikes is high during non-working days.
# 
# ### 5) The demand of bike has a positive correlation with temperature.As the temperature rises, the bike rides also increase.
# 
# ### 6) There is a negative correlation of bikes rented w.r.t. humidity.In high humidity,less bikes are rented.
# 
# 
# ### I recommend the company to keep their stock according to these recommendations.

# In[ ]:




