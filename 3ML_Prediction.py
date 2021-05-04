#====================================================
# Part 3: Machine Learning
#====================================================
# * 3.1. Cluster analysis
# * 3.2. Linear regression
# * 3.3. Random Forest

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#%matplotlib inline
import numpy as np
from sklearn.cluster import KMeans
from sklearn import linear_model
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

# Loading the data
shareprice = pd.read_csv("/Users/brendanmckenna/Dropbox/Projects/OG/Data/SharePricePostProcessed.csv")
shareprice["date"]=pd.to_datetime(shareprice['date'])
master_df13=shareprice[shareprice["year"]>2012]

## 3.1. Cluster analysis
# Clustering is the task of grouping a set of objects based on degree of similarity. Here, we divide data from Royal Dutch Shell into 6 groups using cluster analysis. 
# Unsupervised Learning - Cluster analysis on Shell data
shell=pd.DataFrame()
shell=master_df13[master_df13['name']=="RDSB.L"]

# Scale oil price to ensure it isn't influenced by the relative size of one axis.
shell["oil_price_scaled"] = scaler.fit_transform(shell["oil_price"].to_frame())
shell["cluster"] = KMeans(n_clusters=4, random_state=1).fit_predict(shell[["share_price_scaled","oil_price_scaled"]])

colors = ["baby blue", "amber", "scarlet", "grey","milk chocolate", "windows blue"]
palette=sns.xkcd_palette(colors)

sns.lmplot(x="oil_price", y="share_price_scaled",ci=None,palette=palette, hue="cluster",fit_reg=0 ,data=shell)

## 3.2.- Linear regression
# Univariant linear regression on Royal Dutch Shell share price vs oil price
# Supervised learning linear regression

# 1. Data preparation

shell15=pd.DataFrame()
shell15=master_df13[(master_df13['name']=="RDSB.L") & (master_df13['year']>2015 )] 
shell15=shell15[["share_price","oil_price"]].reset_index()

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
shell15=master_df13[(master_df13['name']=="RDSB.L") & (master_df13['year']>2015 )]
shell15=shell15[["share_price","oil_price"]].reset_index()

shell15['PMO.L']=master_df13[(master_df13['name']=="PMO.L")][-373:].reset_index()['share_price']
shell15['CNE.L']=master_df13[(master_df13['name']=="CNE.L")][-373:].reset_index()['share_price']
shell15['FP.PA']=master_df13[(master_df13['name']=="FP.PA")][-373:].reset_index()['share_price']
shell15['ENGI.PA']=master_df13[(master_df13['name']=="ENGI.PA")][-373:].reset_index()['share_price']

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
