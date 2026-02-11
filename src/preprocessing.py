import pandas as pd
import numpy as np
import os


from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Raw Data
os.chdir(r"D:\1.Work\WorkStation\WorkSpace\Projects\Customer-Churn-Prediction")  # Set the working directory to the project folder to avoid issues with file paths
cwd = os.getcwd()
rawdata_path = os.path.join(cwd, "data", "raw",'data.csv')
data = pd.read_csv(rawdata_path)

# Preprocessing
data.drop(columns='Unnamed: 0',inplace=True)
data.drop_duplicates(inplace=True)

num_cols = [i for i in data.columns if data[i].dtype in ['int64','float64','int32','float32']]

# Feature Engineering
data2 = data.copy()

data2.rename(columns={'Charge  Amount':'Charge Amount'},inplace=True)
data2.rename(columns={'Call  Failure':'Call Failure'},inplace=True)
data2.rename(columns={'Subscription  Length':'Subscription Length'},inplace=True)

data2['Usage'] = data2['Seconds of Use']*data2['Frequency of use']
data2['Usage'] = data2['Usage'].apply(lambda x: np.sqrt(x))
data2.drop(columns=['Seconds of Use','Frequency of use'],inplace=True)

data2['Userbase'] = data2['Distinct Called Numbers']*data2['Customer Value']
data2['Userbase'] = data2['Userbase'].apply(lambda x: np.sqrt(x))
data2.drop(columns=['Distinct Called Numbers','Customer Value'],inplace=True)

data2['Usage'] = data2['Usage'].apply(lambda x: float('{:.2f}'.format(x)))
data2['Userbase'] = data2['Userbase'].apply(lambda x: float('{:.2f}'.format(x)))

data2 = data2[['Call Failure','Complains','Subscription Length','Charge Amount','Frequency of SMS','Tariff Plan','Status','Age','Usage','Userbase','Churn']]

# Transformation
data2['Call Failure'] = data2['Call Failure'].apply(lambda x: float('{:.2f}'.format(np.log10(1+x))))
data2['Subscription Length'] = data2['Subscription Length'].apply(lambda x: float('{:.2f}'.format(np.sqrt(x))))
data2['Charge Amount'] = data2['Charge Amount'].apply(lambda x: float('{:.2f}'.format(np.log10(1+x))))
data2['Frequency of SMS'] = data2['Frequency of SMS'].apply(lambda x: float('{:.2f}'.format(np.log10(1+x))))
data2['Tariff Plan'] = data2['Tariff Plan'].apply(lambda x: x-1)
data2['Status'] = data2['Status'].apply(lambda x: x-1)
data2['Usage'] = data2['Usage'].apply(lambda x: float('{:.2f}'.format(np.log10(1+x))))
data2['Userbase'] = data2['Userbase'].apply(lambda x: float('{:.2f}'.format(np.log10(1+x))))

scaler = StandardScaler()
feature = data2['Age'].values.reshape(-1,1)
data2['Age'] = scaler.fit_transform(feature).ravel()
data2['Age'] = data2['Age'].apply(lambda x: '{:.2f}'.format(x))
data2['Age'] = data2['Age'].astype('float')

# Export data to dvc
processed_path = os.path.join(cwd,'data','processed','preprocessed.csv')
data2.to_csv(processed_path,index=False)