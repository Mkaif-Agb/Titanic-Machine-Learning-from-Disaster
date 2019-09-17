import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,classification_report
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv("titanic_train.csv")


# Visualizing The Data
sns.pairplot(df, palette='coolwarm',)
plt.show()
sns.heatmap(df.isnull(),cbar=False,cmap='viridis',linewidths=1, linecolor='Black')
plt.show()
sns.set_style("whitegrid")
sns.countplot(x='Survived',data=df,hue='Sex')
sns.countplot(x='Survived',data=df,hue='Pclass')
sns.distplot(df['Age'].dropna(),bins=30,kde=False)
sns.distplot(df['Fare'],bins=30,kde=False)

# Cleaning The Data

plt.figure(figsize=(12,9))
sns.boxplot(x='Pclass',y='Age',data=df)

def imputeage(cols):
    Age = cols[0]
    Pclass = cols[1]
    if pd.isnull(Age):
        if Pclass==1:
            return 37
        elif Pclass==2:
            return 29
        else:
            return 24
    else:
        return Age

df['Age'] = df[['Age', 'Pclass']].apply(imputeage,axis=1) # All the Age are sorted
df.drop('Cabin',axis=1, inplace=True)
df.dropna(inplace=True)

# Now For Dummy Variables

sex = pd.get_dummies(df['Sex'],drop_first=True)
embarked = pd.get_dummies(df['Embarked'],drop_first=True)
df = pd.concat([df,sex,embarked],axis=1)

df.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)
df.drop('PassengerId',axis=1,inplace=True)

# Logistic Regression

y = df.iloc[:, 0:1].values
X = df.iloc[:, 1:].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

classifier = LogisticRegression()
classifier.fit(X_train, y_train)

y_pred_log = classifier.predict(X_test)

cm = confusion_matrix(y_test, y_pred_log)
cr = classification_report(y_test, y_pred_log)
print(cm)
print(cr)


from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)

y_pred_KNN = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix, classification_report
print(confusion_matrix(y_test, y_pred_KNN))
print(classification_report(y_test, y_pred_KNN))


# Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion='gini', max_depth=5, random_state=0) # gini, entropy,etc
classifier.fit(X_train, y_train)

y_pred_Decision = classifier.predict(X_test)
from sklearn.metrics import confusion_matrix, classification_report
print(confusion_matrix(y_test, y_pred_Decision))
print(classification_report(y_test, y_pred_Decision))

# Random Forest Classifier
'''
Random Forest has a special Parameter 'n_estimators' which can take in a certain number 
It decides The number of trees in the forest.
To get the Best possible value we create a loop and calculate a mean of predicted value and the real value
We then plot the range with respect to err_rate to get the best possible value
'''
from sklearn.ensemble import RandomForestClassifier
err_rate = []
for i in range(1, 5):
    classifier = RandomForestClassifier(n_estimators=i, criterion='gini', random_state=0)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    err_rate.append(np.mean(y_pred!=y_test))

plt.plot(range(1,5), err_rate, ls='-', color='r')
plt.title("N_Estimators")
plt.xlabel('Range')
plt.ylabel('Error Rate')
plt.tight_layout()
plt.show()

# On running the above code we can see the best number of trees in the forest are 40 so we will create an instance with-
# 2 tress and get the best possible value

classifier = RandomForestClassifier(n_estimators=2, criterion='gini', random_state=0)
classifier.fit(X_train, y_train)


y_pred_tree = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix, classification_report
print(confusion_matrix(y_test, y_pred_tree))
print(classification_report(y_test, y_pred_tree))

# Kernel Svm

from sklearn.svm import SVC
classifier = SVC(kernel='rbf', random_state=0) # We can try different Kernels to try and get best accuracy
classifier.fit(X_train, y_train)

y_pred_svm = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix, classification_report
print(confusion_matrix(y_test, y_pred_svm))
print(classification_report(y_test, y_pred_svm))

# Naive Bayes

from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)
y_pred_nb = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix, classification_report
print(confusion_matrix(y_test, y_pred_nb))
print(classification_report(y_test, y_pred_nb))



'''
After Running all the Models we can get the accuracy score from scikit-learn and choose which is the best model suited
for the Data

'''

from sklearn import metrics                               # Accuracy of all the classifiers respectively
print(metrics.accuracy_score(y_test, y_pred_log))         #0.8252
print(metrics.accuracy_score(y_test, y_pred_Decision))    #0.8389
print(metrics.accuracy_score(y_test, y_pred_tree))        #0.8089
print(metrics.accuracy_score(y_test, y_pred_KNN))         #0.8239
print(metrics.accuracy_score(y_test, y_pred_svm))         #0.8464
print(metrics.accuracy_score(y_test, y_pred_nb))          #0.8052

'''
Most of the Models performed equally with 82% accuracy average we can try and increase the accuracy by using better 
hyper-parameters which can be done by using grid search or we can try out a new Architecture called Neural Networks
Artificial Neural Network can be very good at classifying Models on 1 downfall of using a lot of computational power
'''