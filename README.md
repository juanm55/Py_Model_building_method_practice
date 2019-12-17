# Py_Model_building_method_practice
Experimenting with the different methods for building a model

So basically, there are supposed to be several methods for feature selection, this is how this topic is called, however it seems to me more like feature testing and reduction, i.e. we test every combination of features following a certain logic and stop when we achieve the target R2.

##Forward Selection
Start by testing how a model performs (R2) for every variable or feature you have, select the best, then test how models perform with 2 (the one you had and each of the combinations of the remaining ones), keep doing until achieving desired R2

##Backward elimination
Build a model with ALL features, and start taking out features in order by the highest p value, i.e. the ones that really have nothing to do with the behaviour on analysis, stop when all p values are under desired treshold or when desired R2 is achieved.

##Stepwise elimination
Found little resources on this, they only say it's similar to forward selection, but you can also delete features. So I am guessing that it's like, the mean approach, build a model with half of the features, randomly or pre-selected by correlation or any other technique, then take turns adding one or deleting one feature.

It only bothers me that we could end up with a model that has good performance, but has a feature that should not be in there, because the R2 was achieved by the addition of the last highly correlated feature
