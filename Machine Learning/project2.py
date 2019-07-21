# Load libraries
import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
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

print(dataset.shape)

# Split-out validation dataset
array = dataset.values
X = array[:,0:4]
Y = array[:,4]

validation_size = 0.50
seed = 14
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)
print(len(X))
print(len(Y))
print(len(X_validation))
print(len(X_train))
print(len(Y_train))
print(len(Y_validation))



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

for name, model in models:
#cross validation among models, score based on accuracy
	kfold = model_selection.KFold(n_splits=10, random_state=seed)
	cv_results = model_selection.cross_val_score(model, X_train, Y_train, scoring='accuracy', cv=kfold )
	print("\n"+name)
	model_names.append(name)
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