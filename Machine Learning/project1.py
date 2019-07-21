# Load libraries

import pandas as pd
import numpy as np
from sklearn import model_selection

df = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/iris/bezdekIris.data",
names = ["Sepal Length", "Sepal Width", "Petal Length", "Petal Width", "Class"])

#shuffle our data and we use 121 out of 150 as training data


#split our data in 10 folds
kfold = model_selection.KFold(n_splits=10)

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
	
print("Success")

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csv(url, names=names)

array = dataset.values
X = array[:,0:4]
Y = array[:,4]

validation_size = 0.2
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)



# Test options and evaluation metric
seed = 7
scoring = 'accuracy'

# Spot Check Algorithms
models = []
models.append(("LoR", LogisticRegression()) )
models.append(("LDA", LinearDiscriminantAnalysis()) )
models.append(("QDA", QuadraticDiscriminantAnalysis()) )
models.append(("SVC", SVC()) )
models.append(("LSVC", LinearSVC()) )
models.append(("SGD", SGDClassifier()) )
models.append(("KNN", KNeighborsClassifier()) )
models.append(("GNB", GaussianNB() ))
models.append(("DT", DecisionTreeClassifier()) )
models.append(("RF", RandomForestClassifier()) )

# evaluate each model in turn
results = []
names = []
means=[]
stds=[]
for name, model in models:
     #cross validation among models, score based on accuracy
     cv_results = model_selection.cross_val_score(model, X_train, Y_train, scoring='accuracy', cv=kfold )
     print("\n"+name)
     models.append(name)
     print("Result: "+str(cv_results))
     print("Mean: " + str(cv_results.mean()))
     print("Standard Deviation: " + str(cv_results.std()))
     means.append(cv_results.mean())
     stds.append(cv_results.std())
	
#new_model
x_loc = np.arange(len(models))
width = 0.5
models_graph = plt.bar(x_loc, means, width, yerr=stds)
plt.ylabel('Accuracy')
plt.title('Scores by models')
plt.xticks(x_loc, model_names) # models name on x-axis

#add valve on the top of every bar
def addLabel(rects):
	for rect in rects:
		height = rect.get_height()
		plt.text(rect.get_x() + rect.get_width()/2., 1.05*height,
		'%f' % height, ha='center',
		va='bottom')

addLabel(models_graph)

plt.show()


