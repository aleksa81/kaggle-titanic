from sklearn.neighbors import KNeighborsClassifier
import pandas as pd

from titanic import X_train, Y_train, X_test, Y_validate

knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)

Y_pred = list(map(lambda x: 1 if x >0.5 else 0, Y_pred))

submission = pd.DataFrame({
    "PassengerId": Y_validate,
    "Survived": Y_pred
})

submission.to_csv(
    "submission_knn.csv", 
    index=False
)