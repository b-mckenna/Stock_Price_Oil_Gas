## 1-Load Data
#
# The oil primary dataset includes an excel spreadsheet with oil price and date in a daily frequency. The stock data comes in the shape of a csv file with also daily frequency.
# We wil load the data, read and transform it into a master dataframe.
#

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import warnings; warnings.simplefilter('ignore')
%matplotlib inline


# Read oil price and transform data


xls_file=pd.ExcelFile("/home/cdsw/input/RBRTEd.xls") # Read Excel
brendan=xls_file.parse("Data 1") # Read sheet Data 1
brendan.columns=brendan.iloc[1] # set row 1 as column name
brendan=brendan.ix[2:] # remove first 2 rows
brendan["Date"]=brendan["Date"].astype('datetime64[ns]') # Convert column to date format
brendan.columns=["date","oil_price"]
brendan.head()


brendan.info()

# Now we have just loaded, transformed and checked our oil data.


## Before we start our analysis, we need to read and transform our *share price data*
shares=["RDSB.L","BP.L","CNE.L","PMO.L","CLR","FP.PA","REP.MC","ENGI.PA","SLB.PA"]

# We'll store the share price data in a master dataframe

all_data=pd.DataFrame()
for index in range(len(shares)):
  stock=pd.DataFrame()
  # 1. Read files
  stock=pd.read_csv("/home/cdsw/input/"+shares[index]+".csv")
  # 2. Transform data
  stock=stock[["Date","Close"]]
  stock["Date"]=stock["Date"].astype('datetime64[ns]')
  stock.columns=["date","share_price"]
  test=pd.DataFrame(brendan)
  output=stock.merge(test,on="date",how="left")
  stock["oil_price"]=output["oil_price"]
  stock['share_price']=pd.to_numeric(stock['share_price'], errors='coerce').dropna(0)
  stock['oil_price']=pd.to_numeric(stock['oil_price'], errors='coerce').dropna(0)
  stock["year"]=pd.to_datetime(stock["date"]).dt.year
  stock["name"]=shares[index]
  stock = stock.dropna()
  # 3. Feature Engineering. Create new column with scaled share price from 0 to 1. This will help us comparing companies later on.
  from sklearn.preprocessing import MinMaxScaler
  scaler=MinMaxScaler()
  stock["share_price_scaled"]=scaler.fit_transform(stock["share_price"].to_frame())
  # 4. Append data to a master dataframe
  all_data=all_data.append(stock)
  all_data.head()


## 2. Data Analysis
# To explore the universe, we will start with some practical recipes to make sense of our data. This analysis contains a few of the tools with the purpose of exploring different visualisations that can be useful in a Machine Learning problem. 
# It is not going to be a detailed analysis and I am not going to bring additional features like key events or other metrics to try to explain the patterns in the plots. Again, the idea is to show you a glimpse of the potential of data analysis with python.
# Here is an outline of the main charts that we will make:
# * 2.1. Simple line plot oil price
# * 2.2. Pairplot on BP share price from years 2000 to 2017 using a color gradient for different years
# * 2.3. Pairplot on BP share price using last five years
# * 2.4. Violin plot of the oil price
# * 2.5. Violin plot of the share price of several oil & gas companies
# * 2.6. Jointplot comparison of Premier Oil and Continental
# * 2.7. Plot of oil price vs share price of different companies using different templates
## 2.1 Simple line plot oil price

brendan[['date','oil_price']].set_index('date').plot(color="green", linewidth=1.0)

# To begin with a more advanced data analysis, we will create a pairplot using seaborn to analyze BP share price.

## 2.2.- Pairplot on BP share price from years 2000 to 2017 using a color gradient for different years
#==============================================================================
# Pairplot using master data table (all_data) with a filter on BP share price
#==============================================================================

palette=sns.cubehelix_palette(18, start=2, rot=0, dark=0, light=.95, reverse=False)
sns.pairplot(all_data[all_data['name']=="BP.L"].drop(["share_price_scaled"],axis=1),
             hue="year",palette=palette,size=4,markers="o",
             plot_kws=dict(s=50, edgecolor="b", linewidth=0))

# The pairplot shows all the pairwise relationships in a dataset and the univariate distribution of the data for each variable. It gives us a reasonable idea about variable relationships. I have also built a palette with a gradient color with increasing darkness with time. Have a look at the combination of oil price vs BP share price in the top-center plot.
# First of all, see how it evolves in a zigzag with both time and oil price. 
# Then see where we are today, in 2019. It looks like unexplored terrain! 
# Notice also the differences in distribution symmetries comparing the oil price and share prices. Bear in mind this is just a basic analysis on a single oil company. Soon we will compare between different companies.
# Now that we've made a plot for BP, you can try to do it for the rest of the companies or comparing different types of data.
#
# There is much more information that can be extracted from this plot, but we will stop here and keep going to next step. Lets try to filter in the last five years where we are covering a large spectrum in the oil price and see if we see more.

## 2.3.- Pairplot on BP share price using last five years
#==============================================================================
# Pairplot on less data 2013 to 2017 using Royal Dutch Shell (LON) stock price
#==============================================================================
# Just for the last 5 years

all_data13=all_data[all_data["year"]>2012]
palette=sns.cubehelix_palette(5, start=2, rot=0, dark=0, light=.95, reverse=False)
sns.pairplot(all_data13[all_data13['name']=="RDSB.L"].drop(["share_price_scaled"],axis=1),
             hue="year",palette=palette,size=4,markers="o",
             plot_kws=dict(s=50, edgecolor="b", linewidth=0))


## 2.4. Violin plot of the oil price
# Now, we will build a few violin plots to learn more about the sensitivity of each company's stock to the oil price. Essentially this is a box plot along with the probability density of data at different values.
### Violin Plot Oil price on last 5 years

sns.set_style("whitegrid")
palette=sns.cubehelix_palette(5, start=2.8, rot=0, dark=0.2, light=0.8, reverse=False)

sns.violinplot(x="year", y="oil_price", data=all_data13[all_data13['name']=="RDSB.L"],
               inner="quart", palette=palette, trim=True)

## 2.5. Violin plot of the share price of several Oil and Gas companies
### Violin Plot Oil price on last 5 years
#

sns.factorplot(x="year", y="share_price_scaled", col='name', col_wrap=3,kind="violin",
               split=True, data=all_data13,inner="quart", palette=palette, trim=True,size=4,aspect=1.2)
sns.despine(left=True)


## 2.6. Jointplot comparison of Premier Oil and Continental
# The following plot is an attempt to draw two variables with bivariate and univariate graphs.
# This is really just an alternative way of visualizing data using a jointplot. 
### Joint plot using 5 years for Premier Oil

sns.jointplot("oil_price", "share_price",data=all_data13[all_data13['name']=="PMO.L"],kind="kde",
              hue="year",size=6,ratio=2,color="red").plot_joint(sns.kdeplot, zorder=0, n_levels=20)

### Joint plot using 5 years for Continental

sns.jointplot("oil_price", "share_price",data=all_data13[all_data13['name']=="CLR"],kind="kde",
              hue="year",size=6,ratio=2,color="blue").plot_joint(sns.kdeplot, zorder=0, n_levels=20)

# There's a difference in share price distribution for the two companies and the shape of the density chart.
## 2.7. Oil price vs share price of different companies using different templates
# The next analysis will do a grid of charts for all companies to check if we see any patterns.
### lmplot using using 5 years for all companies

sns.lmplot(x="oil_price", y="share_price_scaled", col="name",ci=None, col_wrap=3, 
           data=all_data13, order=1,line_kws={'color': 'blue'},scatter_kws={'color': 'grey'}).set(ylim=(0, 1))

# We don't see too much on that chart. Let's add different colors for each year and see if correlations are telling us anything.

palette=sns.cubehelix_palette(5, start=2, rot=0, dark=0, light=.95, reverse=False)
sns.lmplot(x="oil_price", y="share_price_scaled",hue="year", col="name",ci=None, 
           col_wrap=3, data=all_data13, order=1,palette=palette,size=4).set(ylim=(0, 1))

## 3. Machine Learning and Prediction
# Here is an outline of the machine learning problems that we will solve:
# * 3.1. Cluster analysis on Shell data
# * 3.2. Linear regression on Royal Dutch Shell share price vs oil price
# * 3.3. Random Forest on Royal Dutch Shell share price vs oil price

# A potential application of this algorithms would be to evaluate the _relative value_ of the share compared to the oil price. Thus, it can give you an indication if the share is overpriced or undervalued. 

# However, the objective of this exercise is to provide the tools to unlock the potential of Machine Learning in the oil industry.

## 3.1. Cluster analysis on Shell data
# In the following example we will divide the data from Royal Dutch Shell into 6 groups using cluster analysis. Clustering is the task of grouping a set of objects in such a way that objects in the same group are more similar to each other than to those in other groups. If you want to understand a bit more about clustering, see references.
### Unsupervised Learning - Cluster analysis on Shell data

from sklearn.cluster import KMeans

shell=pd.DataFrame()
shell=all_data13[all_data13['name']=="RDSB.L"]

# We need to scale also oil price, so clustering is not influenced by the relative size of one axis.
shell["oil_price_scaled"]=scaler.fit_transform(shell["oil_price"].to_frame())
shell["cluster"] = KMeans(n_clusters=6, random_state=1).fit_predict(shell[["share_price_scaled","oil_price_scaled"]])

# The 954 most common RGB monitor colors https://xkcd.com/color/rgb/
colors = ["baby blue", "amber", "scarlet", "grey","milk chocolate", "windows blue"]
palette=sns.xkcd_palette(colors)

sns.lmplot(x="oil_price", y="share_price_scaled",ci=None,palette=palette, hue="cluster",fit_reg=0 ,data=shell)

# There are many application of practical problems using cluster analysis. In this example we are just using it for data visualization and grouping.

## 3.2.- Linear regression on Royal Dutch Shell share price vs oil price
# Next we will construct a simple linear regression model using supervised learning. The objective is to evaluate the prediction of data from the last 100 days using data trained from years 2016-2019 (excluding test data). Train data is the data used to construct the model and test data is the data we are trying to predict.

### Supervised learning linear regression

from sklearn import linear_model

# 1. Data preparation

shell15=pd.DataFrame()
shell15=all_data13[(all_data13['name']=="RDSB.L") & (all_data13['year']>2015 )] 
shell15=shell15[["share_price","oil_price"]].reset_index()

# Just using 1 variable for linear regression. Use randomforest to try with multiple variables

train = shell15[:-100]
test = shell15[-100:]

x_train=train["oil_price"].to_frame()
y_train=train['share_price'].to_frame()
x_test=test["oil_price"].to_frame()
y_test=test['share_price'].to_frame()

# 2. Create linear regression object

regr = linear_model.LinearRegression()

# 3. Train the model using the training sets

regr.fit(x_train,y_train)

print("Coefficients: ",  float(regr.coef_))
print("Mean squared error: %.2f"
      % np.mean((regr.predict(x_train) - y_train) ** 2))

plt_train=plt.scatter(x_train, y_train,  color='grey')
plt_test=plt.scatter(x_test, y_test,  color='green')
plt.plot(x_train, regr.predict(x_train), color='black', linewidth=3)
plt.plot(x_test,regr.predict(x_test),  color='black', linewidth=3)
plt.xlabel("oil_price")
plt.ylabel("share_price")
plt.legend((plt_train, plt_test),("train data", "test data"))
plt.show()

# In the chart above you can see an approximation of how Linear Regression is fit and trying to predict results from test data. It looks like the prediction data is quite off for lower oil prices. The mean square error of this predictive method is 23210.67. Lets see how a more sofisticated method does on this topic.

## 3.3. Random Forest on Royal Dutch Shell share price vs oil price
# Random forest is an ensemble tool which takes a subset of observations and a subset of variables to build a decision trees. It builds multiple such decision tree and amalgamate them together to get a more accurate and stable prediction.
# Random forest algorithm accepts more than one variable in the input data to predict the output. It runs very efficiently on large databases, its very accurate, can handle many input variables, it has effective methods for estimating missing data and many more advantages. The main disadvantage is overfitting for some tasks or some sets of data. That leads with innacurate predictions. It is also biased in favor of categorical attributes(if used) with more levels. In anycase we are gonna give it a go.
# In top of the oil price, we are going to use other variables to predict the share price of Shell. These are going to be the prices of Premier Oil, Cairn Energy, TOTAL and ENGIE. I know this doesn't make much sense, but we just want to see how to construct a model of this type. It will allow us to see the impact of each one on the final prediction.

from sklearn.ensemble import RandomForestRegressor

# 1. Data preparation
shell15=pd.DataFrame()
shell15=all_data13[(all_data13['name']=="RDSB.L") & (all_data13['year']>2015 )]
shell15=shell15[["share_price","oil_price"]].reset_index()

shell15['PMO.L']=all_data13[(all_data13['name']=="PMO.L")][-373:].reset_index()['share_price']
shell15['CNE.L']=all_data13[(all_data13['name']=="CNE.L")][-373:].reset_index()['share_price']
shell15['FP.PA']=all_data13[(all_data13['name']=="FP.PA")][-373:].reset_index()['share_price']
shell15['ENGI.PA']=all_data13[(all_data13['name']=="ENGI.PA")][-373:].reset_index()['share_price']

train = shell15[:-100]
test = shell15[-100:]

x_train=train[["oil_price","PMO.L","CNE.L","FP.PA","ENGI.PA"]]
y_train=train['share_price']

x_test=test[["oil_price","PMO.L","CNE.L","FP.PA","ENGI.PA"]] 
y_test=test['share_price'].to_frame()

# 2. Create Randomforest object usinig a max depth=5
regressor = RandomForestRegressor(n_estimators=200, max_depth=5 )

# 3. Train data
clf=regressor.fit(x_train, y_train)

# 4. Predict!
y_pred=regressor.predict(x_test)
y_pred=pd.DataFrame(y_pred)

# We are going to have a look at how fitted data looks like:

plt_train=plt.scatter(x_train["oil_price"],y_train,   color='grey')
plt_pred=plt.scatter(shell15["oil_price"], regressor.predict(shell15[["oil_price","PMO.L","CNE.L","FP.PA","ENGI.PA"]]),  color='black')

plt.xlabel("oil_price")
plt.ylabel("share_price")
plt.legend((plt_train,plt_pred),("train data","prediction"))
plt.show()

# The model looks really good just predicting the training data. Probably with quite a bit of overfitting. There are many parameters to tune, but a key one is max depth. This will provide the depth of the trees. The higher the number the more overfitting you will have, depending on the type of data. We will have a look now to how this model predicts or test data.

plt_train=plt.scatter(x_train["oil_price"],y_train,   color='grey')
plt_test=plt.scatter(x_test["oil_price"],y_test,   color='green')
plt_pred=plt.scatter(x_test["oil_price"], y_pred,  color='black')
plt.xlabel("oil_price")
plt.ylabel("share_price")
plt.legend((plt_train, plt_test,plt_pred),("train data", "test data","prediction"))
plt.show()
print("Mean squared error: %.2f"
      % np.mean((regressor.predict(x_train) - y_train) ** 2))

# The prediction on the test data looks much better now, still somehow innacurate for lower oil price environment. If you see the mean squared error, we manage to reduce the error from 23210 to 2709. That is 10 times lower than using linear regression.
# It is always worth to give it a check to the importance of each parameter:

importances=regressor.feature_importances_
indices=list(x_train)
print("Feature ranking:")

for f in range(x_train.shape[1]):
    print("Feature %s (%f)" % (indices[f], importances[f]))

f, (ax1) = plt.subplots(1, 1, figsize=(8, 6), sharex=True)
sns.barplot(indices, importances, palette="BrBG", ax=ax1)
ax1.set_ylabel("Importance")

# It's interesting to see how the importance of the share price of TOTAL is higher than the oil price. This is mostly because they are similar size companies that behave in a similar way.
#
#Just as a summary, you have now the tools to start your own little project or even to understand how this works. Through this article, I hope I have helped you to start thinking more on how you can unlock the value of machine learning in your area.
#





