# Data Preprocessing Template

# Importing the libraries
import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd

def Get_the_Data_set(filename):
    import pandas as pd
    Dataset = pd.read_csv(filename)
    X = Dataset.iloc[:, :-1].to_numpy()
    y = Dataset.iloc[:, -1].to_numpy()
    return Dataset, X, y

# Importing the dataset
dataset, X, y = Get_the_Data_set('50_Startups.csv')

#Now we need to take care of the Categorical Variables by encoding them
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
State_encoder = LabelEncoder()

X[:, -1] = State_encoder.fit_transform(X[:, -1])

one_hot_encoder = OneHotEncoder(categorical_features = [3] )

X = one_hot_encoder.fit_transform(X).toarray()

#Avoiding the Dummy Trap section
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

regressor.fit(x_train, y_train)

# Testing the thing
y_pred = regressor.predict(x_test)


# Applying backwards elimination to reduce characteristics
import statsmodels.api as sm

# Now we need to add a column of "1" for the sm to work

X = np.append(arr = np.ones((len(X),1)).astype(int), values = X , axis = 1 )

x_opt = X[:, [0,1,2,3,4,5]]
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



"""
For Backwards elimination
"""

import statsmodels.regression.linear_model as sm
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
x_opt = X[:, [0, 1, 2, 3, 4, 5]]
x_Modeled = backwardElimination(x_opt, SL)


sm.ols

"""
Forward Elimination
"""





"""
Bidirectional elimination
"""