#-------------------------------------------------------------------------
# AUTHOR: your name
# FILENAME: title of the source file
# SPECIFICATION: description of the program
# FOR: CS 4210- Assignment #2
# TIME SPENT: how long it took you to complete the assignment
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard
# dictionaries, lists, and arrays

#Importing some Python libraries
from sklearn.naive_bayes import GaussianNB
import csv

#Reading the training data in a csv file
#--> add your Python code here
training_data = []
with open('weather_training.csv', 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
      if i > 0: #skipping the header
         training_data.append(row)

#Transform the original training features to numbers and add them to the 4D array X.
#For instance Sunny = 1, Overcast = 2, Rain = 3, X = [[3, 1, 1, 2], [1, 3, 2, 2], ...]]
#--> add your Python code here
X= []
features_map = {'Sunny': 1, 'Overcast' : 2, 'Rain' : 3, 'Hot': 3, 'Mild': 2, 'Cool': 1, 'High': 2, 'Normal': 1,
                'Weak':1,'Strong': 2}
#Transform the original training classes to numbers and add them to the vector Y.
#For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
#--> add your Python code here
Y = []
classes_map = {'Yes': 1, 'No':2}
for row in training_data:
    X.append([features_map[row[1]], features_map[row[2]], features_map[row[3]], features_map[row[4]]])
    Y.append(classes_map[row[5]])


#Fitting the naive bayes to the data
clf = GaussianNB(var_smoothing=1e-9)
clf.fit(X, Y)

#Reading the test data in a csv file
#--> add your Python code here
test_data = []
with open('weather_test.csv', 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
      if i > 0: #skipping the header
         test_data.append(row)

#Printing the header of the solution
#--> add your Python code here
with open('weather_test.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    header = next(reader)
    print("Header of the test data:", header)

#Use your test samples to make probabilistic predictions. For instance: clf.predict_proba([[3, 1, 2, 1]])[0]
#--> add your Python code here
for row in test_data[:10]:  # Limit to the first 10 instances
    # Transform the row into the appropriate format for prediction
    probabilities = clf.predict_proba([[features_map[row[1]], features_map[row[2]], features_map[row[3]], features_map[row[4]]]])[0]  # Get the predicted probabilities
    predicted_class = 'Yes' if probabilities[0] > probabilities[1] else 'No'  # Determine the predicted class
    print(f"Instance: {row[0]}, Predicted Class: {predicted_class}, Probabilities: {probabilities}")



