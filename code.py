import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle 

#Load dataset
dataset = pd.read_csv(r'C:\Users\VICTUS\Desktop\Dataset\House_data.csv')

#Independent variable and Dependent variable
y = dataset.iloc[:,2]
x = dataset.iloc[:,5:6]

#Split data in training and testing set
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=1/3,random_state=0)

regressor = LinearRegression()
regressor.fit(x_train,y_train)

y_pred = regressor.predict(x_test)

comparison = pd.DataFrame({'Actual': y_test,'Predicted':y_pred})
comparison

plt.scatter(x_train,y_train,color='red')
plt.plot(x_test,y_pred,color='blue')
plt.title('Sq_feet vs Price (Training set)')
plt.ylabel('Price')
plt.show()


plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title('Sq_feet vs Price (Test set)')
plt.ylabel('Price')
plt.show()

m = regressor.coef_
print(m)

c = regressor.intercept_
print(c)


# Save the trained model to disk
filename = 'LRM_house_price.pkl'
with open(filename,'wb') as file:
    pickle.dump(regressor,file)
print("Model has been pickled and saved as LRM_house_price.pkl")
