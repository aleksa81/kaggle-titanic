from sklearn.ensemble import RandomForestClassifier
import pandas as pd

from titanic import X_train, Y_train, X_test, Y_validate

random_forest = RandomForestClassifier(n_estimators=101)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)

Y_pred = list(map(lambda x: 1 if x >0.5 else 0, Y_pred))

submission = pd.DataFrame({
    "PassengerId": Y_validate,
    "Survived": Y_pred
})

submission.to_csv(
    "submission_rfc.csv", 
    index=False
)