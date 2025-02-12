#-------------------------------------------------------------------------
# AUTHOR: Jeffrey Rodas
# FILENAME: decision_tree
# SPECIFICATION: description of the program
# FOR: CS 4210- Assignment #1
# TIME SPENT: ~30 minutes
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard
# dictionaries, lists, and arrays

#importing some Python libraries
from sklearn import tree
import matplotlib.pyplot as plt
import csv
db = []
X = []
Y = []

#conversion of the values 
age_map = {'Young': 1, 'Prepresbyopic': 2, 'Presbyopic':3}
spectacle_map = {'Myope':1, 'Hypermetrope':2}
astigmatism_map = {'Yes':1, 'No':2}
tear_production_map = {'Normal': 1, 'Reduced':2}
lens_map = {'No':0,'Yes':1}

#reading the data in a csv file
with open('contact_lens.csv', 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
      if i > 0: #skipping the header
         db.append (row)
         print(row)
         

#transform the original categorical training features into numbers and add to the 4D array X. For instance Young = 1, Prepresbyopic = 2, Presbyopic = 3
#--> add your Python code here
# X =
#conversion of the values 
age_map = {'Young': 1, 'Prepresbyopic': 2, 'Presbyopic':3}
spectacle_map = {'Myope':1, 'Hypermetrope':2}
astigmatism_map = {'Yes':1, 'No':0}
tear_production_map = {'Normal': 1, 'Reduced':2}  

#transform the original categorical training classes into numbers and add to the vector Y. For instance Yes = 1, No = 2
#--> addd your Python code here
# Y =
lens_map = {'No':0,'Yes':1}

#adding the converted values to X and Y
for row in db:
  X.append([age_map[row[0]],spectacle_map[row[1]],astigmatism_map[row[2]],tear_production_map[row[3]]])
  Y.append(lens_map[row[4]])


#fitting the decision tree to the data
clf = tree.DecisionTreeClassifier(criterion = 'entropy')
clf = clf.fit(X, Y)

#plotting the decision tree
tree.plot_tree(clf, feature_names=['Age', 'Spectacle', 'Astigmatism', 'Tear'], class_names=['Yes','No'], filled=True, rounded=True)
plt.show()
