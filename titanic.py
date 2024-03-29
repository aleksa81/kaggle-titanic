import pandas as pd
import numpy as np

train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

excluded_columns = ['Ticket', 'Cabin']

train_df = train_df.drop(excluded_columns, axis=1)
test_df = test_df.drop(excluded_columns, axis=1)

combine = [train_df, test_df]

#print(train_df.columns.values)

#print(train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False))
#print(train_df[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False))
#print(train_df[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False))
for dataset in combine:
    dataset['Title'] = dataset['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

#print(pd.crosstab(train_df['Title'], train_df['Sex']))

for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

#print(train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean())

title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

#print(train_df.head()['Title'])

# passengerId will be needed in test_df in order to make resulting file
train_df = train_df.drop(['Name', 'PassengerId'], axis=1)
test_df = test_df.drop(['Name'], axis=1)
combine = [train_df, test_df]

for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map({ 'female':1, 'male':0}).astype(int)

#check how many rows miss Age
#print(len(train_df[train_df['Age'].isnull()]))

guess_ages = np.zeros((2,3))
for dataset in combine:
    for i in range(0,2):
        for j in range(0,3):
            guess_df = dataset[(dataset['Sex'] == i) & (dataset['Pclass'] == j+1)]['Age'].dropna()
            age_guess = guess_df.median()
            guess_ages[i,j] = int(age_guess/0.5 + 0.5) * 0.5
    
    for i in range(0,2):
        for j in range(0,3):
            dataset.loc[(dataset['Age'].isnull()) & (dataset['Sex'] == i) & (dataset['Pclass'] == j+1), 'Age'] = guess_ages[i,j]
    
    dataset['Age'] = dataset['Age'].astype(int)

# confirm that Age generating step was successful
#print(len(train_df[train_df['Age'].isnull()]))

#train_df['AgeBand'] = pd.cut(train_df['Age'], 5)
#print(train_df[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True))

for dataset in combine:    
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age'] = 4

#print(train_df['Age'].unique())
combine = [train_df, test_df]
#print(train_df.head())

for dataset in combine:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

#for dataset in combine:
#    dataset['FamilySizeBand'] = pd.cut(dataset['FamilySize'], 5)
#print(train_df['FamilySizeBand'].unique())

for dataset in combine:
    dataset.loc[ dataset['FamilySize'] <= 2, 'FamilySize'] = 0
    dataset.loc[(dataset['FamilySize'] > 2) & (dataset['FamilySize'] <= 3), 'FamilySize'] = 1
    dataset.loc[(dataset['FamilySize'] > 3) & (dataset['FamilySize'] <= 5), 'FamilySize'] = 2
    dataset.loc[(dataset['FamilySize'] > 5) & (dataset['FamilySize'] <= 7), 'FamilySize'] = 3
    dataset.loc[ dataset['FamilySize'] > 7, 'FamilySize'] = 4

#print(train_df['FamilySize'].unique())
print(train_df[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='FamilySize', ascending=True))

train_df = train_df.drop(['Parch', 'SibSp'], axis=1)
test_df = test_df.drop(['Parch', 'SibSp'], axis=1)
combine = [train_df, test_df]

#print(train_df.head())

# if port is empty fill with most popular port
freq_port = train_df.Embarked.dropna().mode()[0]
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)
# encode ports
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

# test has one Fare null value, train is all good
test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)

for dataset in combine:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)

print(train_df[['Fare', 'Survived']].groupby(['Fare'], as_index=False).mean().sort_values(by='Fare', ascending=True))

combine = [train_df, test_df]
#print(len(train_df))

X_train = train_df.drop("Survived", axis=1)
Y_train = train_df["Survived"].astype(float)
X_test  = test_df.drop("PassengerId", axis=1).copy()
Y_validate = test_df["PassengerId"]



