import pandas as pd
import numpy as np
housing = pd.read_csv(r"D:\Ananthi\Projects & Documents\Machine learning\Project\House\Chennai House Price Prediction\Chennai.csv")
housing.head()

import matplotlib.pyplot as plt
housing.hist(bins=50, figsize=(10, 8))
plt.show()

housing.rename(columns={"No. of Bedrooms":"Bedrooms","Children'splayarea":"Playarea"}, inplace=True)


from sklearn.preprocessing import LabelEncoder
le_location=LabelEncoder()
housing["Location"]=le_location.fit_transform(housing["Location"])


"""corr_matrix = housing.corr()
print(corr_matrix["Price"].sort_values(ascending=False))
"""
housing_labels=housing["Price"]
housing_labels=np.log(housing["Price"]).values

housing=housing.drop("Price",axis=1)
housing=housing.values



#housing.replace(9,np.nan,inplace=True)
#housing.dropna(axis=0,how="any",inplace=True)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(housing, housing_labels, test_size=0.20, random_state=42)


from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)


from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)



print("LR: ", lin_reg.score(X_test, y_test))


import matplotlib.pyplot as plt
predictions = lin_reg.predict(X_test)  
plt.scatter(y_test,predictions)


# from sklearn import metrics

# print('MAE:', metrics.mean_absolute_error(y_test, predictions)) 
# print('MSE:', metrics.mean_squared_error(y_test, predictions)) 
# print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions))) 



# from sklearn.ensemble import RandomForestClassifier
# classifier=RandomForestClassifier(n_estimators=10,criterion='entropy',random_state=0)
# classifier.fit(X_train,y_train)


from sklearn.svm import SVR
c=SVR(kernel='linear')
c.fit(X_train,y_train)

print("ACC: ",lin_reg.score(X_test, y_test))