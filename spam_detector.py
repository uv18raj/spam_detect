import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('spam.csv', encoding='latin-1')
df = df[['v1', 'v2']]

df.columns = ['label','message']

df.head()

df['label'] = df['label'].map({'ham':0,'spam':1})

X = df['message']
y = df['label'] 

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42
)


from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)


from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
model.fit(X_train, y_train)
predictions = model.predict(X_test)


from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, predictions))


from sklearn.metrics import classification_report
print(classification_report(y_test, predictions))


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, predictions)
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='summer',
    xticklabels=['Ham','Spam'],
    yticklabels=['Ham','Spam']
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()