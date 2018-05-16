
# coding: utf-8

# # Import packages and load data

# In[10]:

import pandas as pd


# In[11]:

train = pd.read_csv('train.csv')


# ## Summarize the continuous variables

# In[12]:

train.describe()


# ## Examine the first few rows of data

# In[13]:

train.head()


# ## Check for missing values
# Age will have to be imputed.  Cabin should probably be dropped since so many values are missing.  Rows missing embarked can be dropped since it is only 2 rows.

# In[14]:

train.isnull().sum()


# ## Impute Age using median
# We could do something fancier using median by group or doing some predictive modeling to fill in the missing age values.

# In[15]:

train['ImputedAge'] = train['Age'].fillna(train['Age'].median())
train.head()
train.isnull().sum()


# ## Extract Titles
# The Name column is probably useless on it's own but the title associated with each name could be useful.

# In[16]:

def get_title(name):
    if '.' in name:
        return name.split(',')[1].split('.')[0].strip()
    else:
        return 'Unknown'
    
train['Title'] = train['Name'].map(lambda x: get_title(x))
train.head()


# ## Drop Age and Cabin
# Then check to make sure all remaining data is complete.

# In[17]:

train.drop(['Age', 'Cabin'], axis = 1, inplace = True)
train.dropna(inplace = True)
train.isnull().sum()


# ## Visualize data

# ### Compare continous variables

# In[19]:

import seaborn as sns
from matplotlib import pyplot as plt
continuous = ['ImputedAge', 'SibSp', 'Parch', 'Fare']
categorical = ['Survived', 'Pclass', 'Sex', 'Embarked', 'Title']
g = sns.PairGrid(train[continuous])
g.map(plt.scatter)
plt.show()


# ### Boxplots of continous versus categorical
# Age has only slight correlation with survival.  Very large families are unlikely to survive.  High fare and high Pclass are correlated with survival.  Sex is correlated with survival.  Some titles are correlated with survival.  Let's consider all of these variables as candidates for inclusion in the models.

# In[21]:

sns.boxplot('Survived', 'ImputedAge', data = train)
plt.show()


# In[22]:

sns.boxplot('Survived', 'SibSp', data = train)
plt.show()


# In[23]:

sns.boxplot('Survived', 'Parch', data = train)
plt.show()


# In[25]:

sns.boxplot('Survived', 'Fare', data = train)
plt.show()


# ### Compare categorical variables

# In[27]:

sns.barplot('Pclass', 'Survived', data = train)
plt.show()


# In[28]:

sns.barplot('Sex', 'Survived', data = train)
plt.show()


# In[29]:

sns.barplot('Embarked', 'Survived', data = train)
plt.show()


# In[30]:

sns.barplot('Title', 'Survived', data = train)
plt.show()


# ## Set up data for fitting
# Choose which variables to keep, encode categorical variables, and split into a training set and a test set.

# In[9]:

from sklearn.model_selection import train_test_split
y = train['Survived']
#keep = ['Sex', 'Pclass', 'Title', 'ImputedAge', 'SibSp', 'Parch', 'Fare']
keep = ['Sex', 'Pclass', 'Title', 'ImputedAge', 'SibSp', 'Parch', 'Fare', 'Embarked']
X = train[keep]
X = pd.get_dummies(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# ## Fit a Logistic Regression model
# Fit to the training set.  Score on the test set.

# In[10]:

from sklearn.linear_model import LogisticRegression
import sklearn.preprocessing
model = LogisticRegression()
model.fit(X_train, y_train)
model.score(X_test, y_test)


# ## Use k-fold cross validation

# In[11]:

from sklearn.model_selection import cross_val_score
import numpy as np
scores = cross_val_score(model, X, y, cv = 5)
print(scores)
print(np.mean(scores))


# ## Select hyperparameters by cross validation
# Choose the level of regularization of the logistic regression by doing a grid search over a set of possible values.
# 
# There are multiple ways to quantify the performance of a classification algorithm.  The Kaggle competition for the Titanic dataset uses the prediction accuracy, but it is interesting to consider other metrics.  We consider five different performance metrics:
# 
# 1. Precision: Correctly identified survivors/total identified survivors.  This is a measure of the quality of the positive results.
# 2. Recall: Correctly identified survivors/total survivors.  This is a measure of the quantity of the positive results.
# 3. Accuracy: Fraction of cases correctly identified, whether survivor or not.
# 4. F1: The F-score is the harmonic mean of precision and recall.
# 5. AUC: This is the area under the ROC curve.
# 
# ROC is the receiver operating characteristic.  The logistic regression produces a probability of survivorship for each person in the dataset.  What probability threshold should we use to predict survivorship?  0.5 may seem logical, but we could use other choices.  The ROC is the relationship between the true positive rate and the false positive rate as the threshold is varied.  The AUC is the area under this curve.
# 
# The code block below chooses the best hyperparameters to maximize each of the five metrics in turn.

# In[12]:

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report


# Set the parameters by cross-validation
tuned_parameters = [{'penalty': ['l1', 'l2'], 'C': [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000]}]

scores = ['precision', 'recall', 'accuracy', 'f1', 'roc_auc']

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = GridSearchCV(model, tuned_parameters, cv=5,
                       scoring=score)
    clf.fit(X_train, y_train)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred))
    print()


# The hyperparameters that maximize the AUC:

# In[13]:

clf.best_params_


# ## Plot ROC
# Using the choice of hyperparameters that optimize the AUC we plot the ROC for using cross validation.

# In[14]:

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc
from scipy import interp
import matplotlib.pyplot as plt
from itertools import cycle
import numpy as np
cv = StratifiedKFold(n_splits=6)
classifier = LogisticRegression(**clf.best_params_)
X = X.as_matrix()
y = y.as_matrix()

tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

i = 0
for train, test in cv.split(X, y):
    #print(X[train])
    probas_ = classifier.fit(X[train], y[train]).predict_proba(X[test])
    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    plt.plot(fpr, tpr, lw=1, alpha=0.3,
             label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

    i += 1
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
         label='Luck', alpha=.8)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
plt.plot(mean_fpr, mean_tpr, color='b',
         label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
         lw=2, alpha=.8)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                 label=r'$\pm$ 1 std. dev.')

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()


# # Bundle previous steps into a single function

# In[15]:

from sklearn.svm import SVC
def run_steps(X, y, model_f, tuned_parameters):
    model = model_f()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    model.fit(X_train, y_train)
    print("Score on test using default settings: ", model.score(X_test, y_test))
    scores = cross_val_score(model, X, y, cv = 5)
    print("Scores for 5 fold CV: ", scores)
    print("Mean score for 5 fold CV: ", np.mean(scores))
    
    #tuned_parameters = [{'penalty': ['l1', 'l2'], 'C': [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000]}]

    scores = ['precision', 'recall', 'accuracy', 'f1', 'roc_auc']
    #scores = ['recall', 'accuracy', 'f1', 'roc_auc']

    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()

        clf = GridSearchCV(model, tuned_parameters, cv=5,
                       scoring=score)
        clf.fit(X_train, y_train)

        print("Best parameters set found on development set:")
        print()
        print(clf.best_params_)
        print()
        print("Grid scores on development set:")
        print()
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
        print()

        print("Detailed classification report:")
        print()
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print()
        y_true, y_pred = y_test, clf.predict(X_test)
        print(classification_report(y_true, y_pred))
        print()
        
    print("Best settings for ", scores[-1], ":", clf.best_params_)
    
    cv = StratifiedKFold(n_splits=6)
    if (model_f == SVC):
        classifier = model_f(probability = True, **clf.best_params_)
    else:
        classifier = model_f(**clf.best_params_)
    #X = X.as_matrix()
    #y = y.as_matrix()

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    i = 0
    for train, test in cv.split(X, y):
        #print(X[train])
        probas_ = classifier.fit(X[train], y[train]).predict_proba(X[test])
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=1, alpha=0.3,
             label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

        i += 1
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
         label='Luck', alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b',
         label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
         lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                 label=r'$\pm$ 1 std. dev.')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()


# ## Logistic Regression

# In[16]:

model = LogisticRegression
tuned_parameters = [{'penalty': ['l1', 'l2'], 'C': [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000]}]
run_steps(X, y, model, tuned_parameters)


# ## K Neighbors Classifier

# In[21]:

from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier
tuned_parameters = [{'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 'metric': ['euclidean', 'manhattan', 'chebyshev']}]
run_steps(X, y, model, tuned_parameters)


# ## Support Vector Classifier

# In[22]:

from sklearn.svm import SVC
model = SVC
tuned_parameters = [{'kernel': ['linear'], 'C': [0.03, 0.1, 0.3, 1]}]
run_steps(X, y, model, tuned_parameters)


# ## Gaussian Process Classifier

# In[23]:

from sklearn.gaussian_process import GaussianProcessClassifier
model = GaussianProcessClassifier
tuned_parameters = [{'n_restarts_optimizer': [1, 2, 3]}]
run_steps(X, y, model, tuned_parameters)


# ## Decision Tree Classifier

# In[24]:

from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier
tuned_parameters = [{'criterion': ['gini', 'entropy'], 
                     'splitter': ['best', 'random'], 
                     'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}]
run_steps(X, y, model, tuned_parameters)


# ## Random Forest Classifier

# In[25]:

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier
tuned_parameters = [{'n_estimators': [10, 20, 30],
                     'criterion': ['gini', 'entropy'], 
                     'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]}]
run_steps(X, y, model, tuned_parameters)


# ## Extra Trees Classifier

# In[26]:

from sklearn.ensemble import ExtraTreesClassifier
model = ExtraTreesClassifier
tuned_parameters = [{'n_estimators': [10, 20, 30],
                     'criterion': ['gini', 'entropy'], 
                     'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]}]
run_steps(X, y, model, tuned_parameters)


# ## Multi-Layer Perceptron Classifier

# In[27]:

from sklearn.neural_network import MLPClassifier
model = MLPClassifier
tuned_parameters = [{'hidden_layer_sizes': [(10,), (100,), (10,10), (100,100)],
                     'activation': ['identity', 'logistic', 'tanh', 'relu'], 
                     'alpha': [1e-3, 1e-4, 1e-5], 
                    'solver': ['lbfgs']}]
run_steps(X, y, model, tuned_parameters)


# ## Ada Boost Classifier

# In[31]:

from sklearn.ensemble import AdaBoostClassifier
model = AdaBoostClassifier
tuned_parameters = [{'n_estimators': [10, 20, 30, 40, 50, 60, 70, 100],
                     'learning_rate': [0.01, 0.1, 1, 10]}]
run_steps(X, y, model, tuned_parameters)


# ## Gaussian Naive Bayes Classifier
# There are no hyperparameters to optimize.

# In[35]:

from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
#model = model_f()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
model.fit(X_train, y_train)
print("Score on test using default settings: ", model.score(X_test, y_test))
scores = cross_val_score(model, X, y, cv = 5)
print("Scores for 5 fold CV: ", scores)
print("Mean score for 5 fold CV: ", np.mean(scores))

#tuned_parameters = [{'penalty': ['l1', 'l2'], 'C': [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000]}]

scores = ['precision', 'recall', 'accuracy', 'f1', 'roc_auc']
#scores = ['recall', 'accuracy', 'f1', 'roc_auc']

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    #clf = GridSearchCV(model, tuned_parameters, cv=5,
                   #scoring=score)
    #clf.fit(X_train, y_train)



    print("Best parameters set found on development set:")
    print()
    #print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    res = cross_val_score(model, X, y, cv = 5, scoring = score)
    means = np.mean(res)
    stds = np.std(res)
    #for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    print("%0.3f (+/-%0.03f)"
          % (mean, std * 2))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_test, model.predict(X_test)
    print(classification_report(y_true, y_pred))
    print()

#print("Best settings for ", scores[-1], ":", clf.best_params_)

cv = StratifiedKFold(n_splits=6)
classifier = model
#if (model_f == SVC):
#    classifier = model_f(**clf.best_params_, probability = True)
#else:
#    classifier = model_f(**clf.best_params_)
#X = X.as_matrix()
#y = y.as_matrix()

tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

i = 0
for train, test in cv.split(X, y):
    #print(X[train])
    probas_ = classifier.fit(X[train], y[train]).predict_proba(X[test])
    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    plt.plot(fpr, tpr, lw=1, alpha=0.3,
         label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

    i += 1
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
     label='Luck', alpha=.8)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
plt.plot(mean_fpr, mean_tpr, color='b',
     label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
     lw=2, alpha=.8)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
             label=r'$\pm$ 1 std. dev.')

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()


# ## Linear Discriminant Analysis
# No hyperparameters to optimize.

# In[37]:

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
model = LinearDiscriminantAnalysis()
#model = model_f()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
model.fit(X_train, y_train)
print("Score on test using default settings: ", model.score(X_test, y_test))
scores = cross_val_score(model, X, y, cv = 5)
print("Scores for 5 fold CV: ", scores)
print("Mean score for 5 fold CV: ", np.mean(scores))

#tuned_parameters = [{'penalty': ['l1', 'l2'], 'C': [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000]}]

scores = ['precision', 'recall', 'accuracy', 'f1', 'roc_auc']
#scores = ['recall', 'accuracy', 'f1', 'roc_auc']

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    #clf = GridSearchCV(model, tuned_parameters, cv=5,
                   #scoring=score)
    #clf.fit(X_train, y_train)



    print("Best parameters set found on development set:")
    print()
    #print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    res = cross_val_score(model, X, y, cv = 5, scoring = score)
    means = np.mean(res)
    stds = np.std(res)
    #for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    print("%0.3f (+/-%0.03f)"
          % (mean, std * 2))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_test, model.predict(X_test)
    print(classification_report(y_true, y_pred))
    print()

#print("Best settings for ", scores[-1], ":", clf.best_params_)

cv = StratifiedKFold(n_splits=6)
classifier = model
#if (model_f == SVC):
#    classifier = model_f(**clf.best_params_, probability = True)
#else:
#    classifier = model_f(**clf.best_params_)
#X = X.as_matrix()
#y = y.as_matrix()

tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

i = 0
for train, test in cv.split(X, y):
    #print(X[train])
    probas_ = classifier.fit(X[train], y[train]).predict_proba(X[test])
    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    plt.plot(fpr, tpr, lw=1, alpha=0.3,
         label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

    i += 1
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
     label='Luck', alpha=.8)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
plt.plot(mean_fpr, mean_tpr, color='b',
     label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
     lw=2, alpha=.8)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
             label=r'$\pm$ 1 std. dev.')

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()


# ## Quadratic Discriminant Analysis
# No hyperparameters to optimize.

# In[36]:

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
model = QuadraticDiscriminantAnalysis()
#model = model_f()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
model.fit(X_train, y_train)
print("Score on test using default settings: ", model.score(X_test, y_test))
scores = cross_val_score(model, X, y, cv = 5)
print("Scores for 5 fold CV: ", scores)
print("Mean score for 5 fold CV: ", np.mean(scores))

#tuned_parameters = [{'penalty': ['l1', 'l2'], 'C': [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000]}]

scores = ['precision', 'recall', 'accuracy', 'f1', 'roc_auc']
#scores = ['recall', 'accuracy', 'f1', 'roc_auc']

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    #clf = GridSearchCV(model, tuned_parameters, cv=5,
                   #scoring=score)
    #clf.fit(X_train, y_train)



    print("Best parameters set found on development set:")
    print()
    #print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    res = cross_val_score(model, X, y, cv = 5, scoring = score)
    means = np.mean(res)
    stds = np.std(res)
    #for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    print("%0.3f (+/-%0.03f)"
          % (mean, std * 2))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_test, model.predict(X_test)
    print(classification_report(y_true, y_pred))
    print()

#print("Best settings for ", scores[-1], ":", clf.best_params_)

cv = StratifiedKFold(n_splits=6)
classifier = model
#if (model_f == SVC):
#    classifier = model_f(**clf.best_params_, probability = True)
#else:
#    classifier = model_f(**clf.best_params_)
#X = X.as_matrix()
#y = y.as_matrix()

tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

i = 0
for train, test in cv.split(X, y):
    #print(X[train])
    probas_ = classifier.fit(X[train], y[train]).predict_proba(X[test])
    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    plt.plot(fpr, tpr, lw=1, alpha=0.3,
         label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

    i += 1
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
     label='Luck', alpha=.8)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
plt.plot(mean_fpr, mean_tpr, color='b',
     label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
     lw=2, alpha=.8)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
             label=r'$\pm$ 1 std. dev.')

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()


# ## Dummy Classifier

# In[38]:

from sklearn.dummy import DummyClassifier
model = DummyClassifier
tuned_parameters = [{'strategy': ['stratified', 'most_frequent', 'prior', 'uniform']}]
run_steps(X, y, model, tuned_parameters)


# In[668]:

test = pd.read_csv('test.csv')


# In[669]:

X = test[keep]
X = pd.get_dummies(X)
model.predict(X)


# In[ ]:



