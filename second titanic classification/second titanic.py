import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


data = pd.read_csv('/content/train.csv')


data = data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'Embarked'], axis=1)  # Drop irrelevant columns
data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})  # Encode gender as binary (0 for male, 1 for female)
data['Age'].fillna(data['Age'].median(), inplace=True)  # Fill missing ages with the median age
data = pd.get_dummies(data, columns=['Pclass'], prefix='Pclass', drop_first=True)  # One-hot encode passenger class


X = data.drop('Survived', axis=1)
y = data['Survived']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy * 100:.2f}%')
print(f'Confusion Matrix:\n{confusion}')
print(f'Classification Report:\n{classification_rep}')
