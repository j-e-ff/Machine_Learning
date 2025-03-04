#-------------------------------------------------------------------------
# AUTHOR: Jeffrey Rodas
# FILENAME: decision_tree_2.py
# SPECIFICATION: Your goal is to train, test, and output the performance of the 3 models created by using each training
#                set on the test set provided (contact_lens_test.csv). You must repeat this process 10 times (train and
#                test using a different training set), choosing the average accuracy as the final classification
#                performance of each model.
# FOR: CS 4210- Assignment #2
# TIME SPENT: how long it took you to complete the assignment
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard
# dictionaries, lists, and arrays

#Importing some Python libraries
from sklearn import tree
import csv

dataSets = ['contact_lens_training_1.csv', 'contact_lens_training_2.csv', 'contact_lens_training_3.csv']

for ds in dataSets:

    dbTraining = []
    X = []
    Y = []

    #Reading the training data in a csv file
    with open(ds, 'r') as csvfile:
         reader = csv.reader(csvfile)
         for i, row in enumerate(reader):
             if i > 0: #skipping the header
                dbTraining.append (row)

    #Transform the original categorical training features to numbers and add to the 4D array X.
    #For instance Young = 1, Prepresbyopic = 2, Presbyopic = 3, X = [[1, 1, 1, 1], [2, 2, 2, 2], ...]]
    #--> add your Python code here
    age_map = {'Young': 1, 'Prepresbyopic': 2, 'Presbyopic': 3}
    spectacle_map = {'Myope': 1, 'Hypermetrope': 2}
    astigmatism_map = {'Yes': 1, 'No': 2}
    tear_production_map = {'Normal': 1, 'Reduced': 2}


    #Transform the original categorical training classes to numbers and add to the vector Y.
    #For instance Yes = 1 and No = 2, Y = [1, 1, 2, 2, ...]
    #--> add your Python code here
    lens_map = {'No': 2, 'Yes': 1}

    for row in dbTraining:
        X.append([age_map[row[0]], spectacle_map[row[1]], astigmatism_map[row[2]], tear_production_map[row[3]]])
        Y.append(lens_map[row[4]])

    accuracies = []

    #Loop your training and test tasks 10 times here
    for i in range (10):

       #Fitting the decision tree to the data setting max_depth=3
       clf = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth=5)
       clf = clf.fit(X, Y)

       #Read the test data and add this data to dbTest
       #--> add your Python code here
       dbTest = []
       with open('contact_lens_test.csv', 'r') as csvfile:
           reader = csv.reader(csvfile)
           for j, row in enumerate(reader):
               if j > 0:
                   dbTest.append(row)

       X_test = []
       Y_test = []
       correct_predictions = 0

       for data in dbTest:
           # Transform the features of the test instances to numbers following the same strategy done during training,
           # and then use the decision tree to make the class prediction. For instance: class_predicted = clf.predict([[3, 1, 2, 1]])[0]
           # where [0] is used to get an integer as the predicted class label so that you can compare it with the true label
           # --> add your Python code here
           X_test.append(
               [age_map[data[0]], spectacle_map[data[1]], astigmatism_map[data[2]], tear_production_map[data[3]]])
           Y_test.append(lens_map[data[4]])

       # Compare the prediction with the true label (located at data[4]) of the test instance to start calculating the accuracy.
       # --> add your Python code here
       for j in range(len(X_test)):
           class_predicted = clf.predict([X_test[j]])[0]
           if class_predicted == Y_test[j]:
               correct_predictions += 1

       accuracy = correct_predictions / len(X_test)
       accuracies.append(accuracy)

    #Find the average of this model during the 10 runs (training and test set)
    #--> add your Python code here
    avg_accuracy = sum(accuracies)/len(accuracies)

    #Print the average accuracy of this model during the 10 runs (training and test set).
    #Your output should be something like that: final accuracy when training on contact_lens_training_1.csv: 0.2
    #--> add your Python code here
    print(f"Final accuracy when training on {ds}: {avg_accuracy}")
