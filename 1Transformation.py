import pandas as pd
import warnings; warnings.simplefilter('ignore')

## 1-Load Data

# Read in oil price datasets
oilpricedata=pd.read_excel("/Users/brendanmckenna/Dropbox/Projects/OG/Data/RBRTEd.xls", sheet_name=1)

# Transform data
oilpricedata.columns=oilpricedata.iloc[1] # set row 1 as column name
print(oilpricedata.head())

oilpricedata=oilpricedata.drop([0,1])
oilpricedata["Date"]=pd.to_datetime(oilpricedata['Date'])
oilpricedata.columns=["date","oil_price"]
print("Oil price data")
print(oilpricedata.head())
oilpricedata.info()
oilpricedata.to_csv("/Users/brendanmckenna/Dropbox/Projects/OG/Data/OilPricePostProcessed.csv", index=False)


# Read in share price data
shares=["RDSB.L","BP.L","CNE.L","PMO.L","FP.PA","REP.MC","ENGI.PA","SLB.PA"]

master_df=pd.DataFrame()

for index in range(len(shares)):
  stock=pd.DataFrame()
  # 1. Read files
  stock=pd.read_csv("/Users/brendanmckenna/Dropbox/Projects/OG/Data/"+shares[index]+".csv")
  # 2. Transform data
  stock=stock[["Date","Close"]]
  stock["Date"]=pd.to_datetime(stock['Date'])
  stock.columns=["date","share_price"]
  test=pd.DataFrame(oilpricedata)
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
  master_df=master_df.append(stock)

master_df.to_csv("/Users/brendanmckenna/Dropbox/Projects/OG/Data/SharePricePostProcessed.csv", index=False)