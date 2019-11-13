import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn import metrics
import joblib

#load dataset
dataframe = pd.read_csv("csv/dataset.csv")
# print(dataframe.head())

#split to training and test data
dataframe.fillna(0, inplace=True)
x = dataframe.drop(["Label"], axis=1)
y = dataframe["Label"]

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=40)

#Build a model
model = RandomForestClassifier(n_estimators=100,max_depth=5)
model.fit(x_train,y_train)

#save the model so that no need to retrain model again
joblib.dump(model,"rf_malaria_100_5")


#make predictions and get classification report
predictions = model.predict(x_test)
print(metrics.classification_report(predictions,y_test))
