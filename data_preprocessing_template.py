# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
x = dataset.iloc[:, :-1].to_numpy()
y = dataset.iloc[:, -1].to_numpy()

#Now we need to take care of the Categorical Variables by encoding them
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
State_encoder = LabelEncoder()

x[:, -1] = State_encoder.fit_transform(x[:, -1])

one_hot_encoder = OneHotEncoder(categorical_features = [3] )

x = one_hot_encoder.fit_transform(x).toarray()

#Avoiding the Dummy Trap section
x = x[:, 1:]


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

regressor.fit(x_train, y_train)

# Testing the thing
y_pred = regressor.predict(x_test)


# Applying backwards elimination to reduce characteristics
import statsmodels.api as sm

# Now we need to add a column of "1" for the sm to work

x = np.append(arr = np.ones((len(x),1)).astype(int), values = x , axis = 1 )

x_opt = x[:, [0,1,2,3,4,5]]
regressor_OLS = sm.OLS(endog = y, exog = x_opt ).fit()
regressor_OLS.summary()


x_opt = x[:, [0,1,3,4,5]]
regressor_OLS = sm.OLS(endog = y, exog = x_opt ).fit()
regressor_OLS.summary()


x_opt = x[:, [0,3,4,5]]
regressor_OLS = sm.OLS(endog = y, exog = x_opt ).fit()
regressor_OLS.summary()

x_opt = x[:, [0,3,5]]
regressor_OLS = sm.OLS(endog = y, exog = x_opt ).fit()
regressor_OLS.summary()

x_opt = x[:, [0,3]]
regressor_OLS = sm.OLS(endog = y, exog = x_opt ).fit()
regressor_OLS.summary()



# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""





import statsmodels.formula.api as sm
def backwardElimination(x, SL):
    numVars = len(x[0])
    temp = np.zeros((50,6)).astype(int)
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        adjR_before = regressor_OLS.rsquared_adj.astype(float)
        if maxVar > SL:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    temp[:,j] = x[:, j]
                    x = np.delete(x, j, 1)
                    tmp_regressor = sm.OLS(y, x).fit()
                    adjR_after = tmp_regressor.rsquared_adj.astype(float)
                    if (adjR_before >= adjR_after):
                        x_rollback = np.hstack((x, temp[:,[0,j]]))
                        x_rollback = np.delete(x_rollback, j, 1)
                        print (regressor_OLS.summary())
                        return x_rollback
                    else:
                        continue
    regressor_OLS.summary()
    return x
 
SL = 0.05
x_opt = x[:, [0, 1, 2, 3, 4, 5]]
x_Modeled = backwardElimination(x_opt, SL)
