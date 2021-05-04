import matplotlib.pyplot as plt
#%matplotlib inline
import numpy as np
import pandas as pd
import seaborn as sns

shareprice=pd.read_csv("/Users/brendanmckenna/Dropbox/Projects/OG/Data/SharePricePostProcessed.csv")
shareprice["date"]=pd.to_datetime(shareprice['date'])

# Simple line plot of the oil price
shareprice[['date','oil_price']].set_index('date').plot(xlabel="Date",ylabel="Oil Price",color="green",title="Historical Oil Prices",linewidth=1.0)

# Designing the color pallete object for the charts
palette=sns.cubehelix_palette(18, start=2, rot=0, dark=0, light=.95, reverse=False)

# Pairplot on BP share price from years 2000 to 2017. We're using the master data table (shareprice) with a filter on BP share price.
# We are using the pairplot to evaluate the relationship of three variables: share price, oil price, and year. 
sns.pairplot(shareprice[shareprice['name']=="BP.L"].drop(["share_price_scaled"],axis=1),
             hue="year",palette=palette,size=4,markers="o",
             plot_kws=dict(s=50, edgecolor="b", linewidth=0)).fig.suptitle("BP", y=1.08)


## Pairplot on BP share price using last five years
master_df13=shareprice[shareprice["year"]>2012]

palette=sns.cubehelix_palette(5, start=2, rot=0, dark=0, light=.95, reverse=False)

sns.pairplot(master_df13[master_df13['name']=="BP.L"].drop(["share_price_scaled"],axis=1),
             hue="year",palette=palette,size=4,markers="o",
             plot_kws=dict(s=50, edgecolor="b", linewidth=0))

# Pairplot on Royal Dutch Shell (LON) share price from years 2013 to 2017
sns.pairplot(master_df13[master_df13['name']=="RDSB.L"].drop(["share_price_scaled"],axis=1),
             hue="year",palette=palette,size=4,markers="o",
             plot_kws=dict(s=50, edgecolor="b", linewidth=0))

plt.show()

# Violin plot of the Royal Dutch Shell to review how sensitive it's share price is to the oil price. 
# We see a box plot and the probability density of data at different values.

sns.set_style("whitegrid")
palette=sns.cubehelix_palette(5, start=2.8, rot=0, dark=0.2, light=0.8, reverse=False)
sns.violinplot(x="year", y="oil_price", data=master_df13[master_df13['name']=="RDSB.L"],
               inner="quart", palette=palette, trim=True)

# Violin plot of multiple companies
sns.factorplot(x="year", y="share_price_scaled", col='name', col_wrap=3,kind="violin",
               split=True, data=master_df13,inner="quart", palette=palette, trim=True,size=4,aspect=1.2)
sns.despine(left=True)

plt.show()

# Joint plots
sns.jointplot("oil_price", "share_price",data=master_df13[master_df13['name']=="PMO.L"],kind="kde",
              hue="year",size=6,ratio=2,color="red").plot_joint(sns.kdeplot, zorder=0, n_levels=20).fig.suptitle("Premier Oil", y=1.08)

sns.jointplot("oil_price", "share_price",data=master_df13[master_df13['name']=="ENGI.PA"],kind="kde",
              hue="year",size=6,ratio=2,color="blue").plot_joint(sns.kdeplot, zorder=0, n_levels=20).fig.suptitle("Engie", y=1.08)

plt.show()

# lmplot
sns.lmplot(x="oil_price", y="share_price_scaled", col="name",ci=None, col_wrap=3, 
           data=master_df13, order=1,line_kws={'color': 'blue'},scatter_kws={'color': 'grey'}).set(ylim=(0, 1))

# Let's add different colors for each year and see if correlations are telling us anything.
palette=sns.cubehelix_palette(5, start=2, rot=0, dark=0, light=.95, reverse=False)
sns.lmplot(x="oil_price", y="share_price_scaled",hue="year", col="name",ci=None, 
           col_wrap=3, data=master_df13, order=1,palette=palette,size=4).set(ylim=(0, 1))

plt.show()
