from sklearn.tree import DecisionTreeClassifier
import pandas as pd

from titanic import X_train, Y_train, X_test, Y_validate

decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_pred = decision_tree.predict(X_test)

Y_pred = list(map(lambda x: 1 if x >0.5 else 0, Y_pred))

submission = pd.DataFrame({
    "PassengerId": Y_validate,
    "Survived": Y_pred
})

submission.to_csv(
    "submission_dt.csv", 
    index=False
)