import matplotlib.pyplot as plt
#%matplotlib inline
import numpy as np
import pandas as pd
import seaborn as sns

shareprice=pd.read_csv("/Users/brendanmckenna/Dropbox/Projects/OG/Data/SharePricePostProcessed.csv")
shareprice["date"]=pd.to_datetime(shareprice['date'])

print(shareprice.head())

## 2.1 Simple line plot of the oil price
shareprice[['date','oil_price']].set_index('date').plot(xlabel="Date",ylabel="Oil Price",color="green",title="Historical Oil Prices",linewidth=1.0)

## 2.2.- Pairplot on BP share price from years 2000 to 2017 using a color gradient for different years
#==============================================================================
# Pairplot using master data table (shareprice) with a filter on BP share price
#==============================================================================

palette=sns.cubehelix_palette(18, start=2, rot=0, dark=0, light=.95, reverse=False)
sns.pairplot(shareprice[shareprice['name']=="BP.L"].drop(["share_price_scaled"],axis=1),
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

master_df13=shareprice[shareprice["year"]>2012]
palette=sns.cubehelix_palette(5, start=2, rot=0, dark=0, light=.95, reverse=False)
sns.pairplot(master_df13[master_df13['name']=="RDSB.L"].drop(["share_price_scaled"],axis=1),
             hue="year",palette=palette,size=4,markers="o",
             plot_kws=dict(s=50, edgecolor="b", linewidth=0))


## 2.4. Violin plot of the oil price
# Now, we will build a few violin plots to learn more about the sensitivity of each company's stock to the oil price. Essentially this is a box plot along with the probability density of data at different values.
### Violin Plot Oil price on last 5 years

sns.set_style("whitegrid")
palette=sns.cubehelix_palette(5, start=2.8, rot=0, dark=0.2, light=0.8, reverse=False)

sns.violinplot(x="year", y="oil_price", data=master_df13[master_df13['name']=="RDSB.L"],
               inner="quart", palette=palette, trim=True)

## 2.5. Violin plot of the share price of several Oil and Gas companies
### Violin Plot Oil price on last 5 years
#

sns.factorplot(x="year", y="share_price_scaled", col='name', col_wrap=3,kind="violin",
               split=True, data=master_df13,inner="quart", palette=palette, trim=True,size=4,aspect=1.2)
sns.despine(left=True)


## 2.6. Jointplot comparison of Premier Oil and Engie
# The following plot is an attempt to draw two variables with bivariate and univariate graphs.
# This is really just an alternative way of visualizing data using a jointplot. 
### Joint plot using 5 years for Premier Oil

sns.jointplot("oil_price", "share_price",data=master_df13[master_df13['name']=="PMO.L"],kind="kde",
              hue="year",size=6,ratio=2,color="red").plot_joint(sns.kdeplot, zorder=0, n_levels=20)

### Joint plot using 5 years for Engie

sns.jointplot("oil_price", "share_price",data=master_df13[master_df13['name']=="ENGI.PA"],kind="kde",
              hue="year",size=6,ratio=2,color="blue").plot_joint(sns.kdeplot, zorder=0, n_levels=20)

# There's a difference in share price distribution for the two companies and the shape of the density chart.
## 2.7. Oil price vs share price of different companies using different templates
# The next analysis will do a grid of charts for all companies to check if we see any patterns.
### lmplot using using 5 years for all companies

sns.lmplot(x="oil_price", y="share_price_scaled", col="name",ci=None, col_wrap=3, 
           data=master_df13, order=1,line_kws={'color': 'blue'},scatter_kws={'color': 'grey'}).set(ylim=(0, 1))

# We don't see too much on that chart. Let's add different colors for each year and see if correlations are telling us anything.

palette=sns.cubehelix_palette(5, start=2, rot=0, dark=0, light=.95, reverse=False)
sns.lmplot(x="oil_price", y="share_price_scaled",hue="year", col="name",ci=None, 
           col_wrap=3, data=master_df13, order=1,palette=palette,size=4).set(ylim=(0, 1))

plt.show()
