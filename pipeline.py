import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
import pickle

training_data = pd.read_csv("storepurchasedata.csv")
x = training_data.iloc[:, :-1].values
y = training_data.iloc[:, -1].values
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0, shuffle=True)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)

classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
y_prob = classifier.predict_proba(X_test)[:, 1]
# print(y_prob, y_test)

cm = confusion_matrix(y_test, y_pred)
#print(cm)

#new_prediction=classifier.predict_proba(sc.transform(np.array([[40, 50000]])))[:,1]
#print(new_prediction)

model_file="classifier.pickle"
pickle.dump(classifier, open(model_file, 'wb'))
scaler_file="sc.pickle"
pickle.dump(sc, open(scaler_file, 'wb'))
