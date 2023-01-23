import pandas as pd 
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv(r'D:\Ananthi\Projects\Machine learning\Project\House\Chennai House Price Prediction\Project\House\Chennai House Price Prediction\Chennai.csv')
print(df.head())


#Spli Train and test data

number = LabelEncoder()
df['Sqtft'] = number.fit_transform(df['Sqtft'])
df['Location'] = number.fit_transform(df['Location'])
df['No. of Bedrooms'] = number.fit_transform(df['No. of Bedrooms'])
df['Resale'] = number.fit_transform(df['Resale'])
df['MaintenanceStaff'] = number.fit_transform(df['MaintenanceStaff'])
df['Gymnasium'] = number.fit_transform(df['Gymnasium'])
df['SwimmingPool'] = number.fit_transform(df['SwimmingPool'])
df['LandscapedGardens'] = number.fit_transform(df['LandscapedGardens'])
df['JoggingTrack'] = number.fit_transform(df['JoggingTrack'])
df['RainWaterHarvesting'] = number.fit_transform(df['RainWaterHarvesting'])
df['IndoorGames'] = number.fit_transform(df['IndoorGames'])
df['ShoppingMall'] = number.fit_transform(df['ShoppingMall'])
df['Intercom'] = number.fit_transform(df['Intercom'])
df['SportsFacility'] = number.fit_transform(df['SportsFacility'])
df['ATM'] = number.fit_transform(df['ATM'])
df['ClubHouse'] = number.fit_transform(df['ClubHouse'])
df['School'] = number.fit_transform(df['School'])
df['24X7Security'] = number.fit_transform(df['24X7Security'])
df['PowerBackup'] = number.fit_transform(df['PowerBackup'])
df['CarParking'] = number.fit_transform(df['CarParking'])
df['StaffQuarter'] = number.fit_transform(df['StaffQuarter'])
df['Cafeteria'] = number.fit_transform(df['Cafeteria'])
df['MultipurposeRoom'] = number.fit_transform(df['MultipurposeRoom'])
df['Hospital'] = number.fit_transform(df['Hospital'])
df['WashingMachine'] = number.fit_transform(df['WashingMachine'])
df['AC'] = number.fit_transform(df['AC'])
df['Gasconnection'] = number.fit_transform(df['Gasconnection'])
df['Wifi'] = number.fit_transform(df['Wifi'])
df["Children'splayarea"] = number.fit_transform(df["Children'splayarea"])
df["LiftAvailable"] = number.fit_transform(df["LiftAvailable"])
df['BED'] = number.fit_transform(df['BED'])
df['VaastuCompliant'] = number.fit_transform(df['VaastuCompliant'])
df['Microwave'] = number.fit_transform(df['Microwave'])
df['GolfCourse'] = number.fit_transform(df['GolfCourse'])
df['TV'] = number.fit_transform(df['TV'])
df['DiningTable'] = number.fit_transform(df['DiningTable'])
df['Sofa'] = number.fit_transform(df['Sofa'])
df['Wardrobe'] = number.fit_transform(df['Wardrobe'])
df['Refrigerator'] = number.fit_transform(df['Refrigerator'])
df['Price'] = number.fit_transform(df['Price'])


features = ['Sqtft', 'Location', 'No. of Bedrooms', 'Resale','MaintenanceStaff', 'Gymnasium', 'SwimmingPool', 'LandscapedGardens','JoggingTrack', 'RainWaterHarvesting', 'IndoorGames', 'ShoppingMall','Intercom', 'SportsFacility', 'ATM', 'ClubHouse', 'School','24X7Security', 'PowerBackup', 'CarParking', 'StaffQuarter','Cafeteria', 'MultipurposeRoom', 'Hospital', 'WashingMachine','Gasconnection', 'AC', 'Wifi', "Children'splayarea", "LiftAvailable",'BED', 'VaastuCompliant', 'Microwave', 'GolfCourse', 'TV','DiningTable', 'Sofa', 'Wardrobe', 'Refrigerator']
target = "Price"

features_train, features_test, target_train, target_test = train_test_split(df[features], df[target], test_size = 0.2)

clf = LinearRegression()
clf.fit(features_train, target_train)
print("Acc LR: ",clf.score(features_test, target_test)*250)

#Decision Tree
from sklearn.tree import DecisionTreeRegressor
regressor3=DecisionTreeRegressor(random_state=100,max_depth=3,min_samples_leaf=5)
regressor3.fit(features_train,target_train)
print("Acc Random: ",regressor3.score(features_test, target_test)*250)
 

#2. Gradient Boosting Regressor
from sklearn.ensemble import GradientBoostingRegressor
regressor5=GradientBoostingRegressor(learning_rate=0.001,n_estimators=400)
regressor5.fit(features_train,target_train)
print("Acc GB: ",regressor5.score(features_test, target_test)*200)