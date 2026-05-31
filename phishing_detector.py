from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

emails = [
    "Your account has been suspended click here",
    "Verify your bank account immediately",
    "Win a free iPhone now",
    "Meeting scheduled for tomorrow",
    "Project report attached",
    "Team meeting at 10 AM"
]

labels = [
    "Phishing",
    "Phishing",
    "Phishing",
    "Safe",
    "Safe",
    "Safe"
]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(emails)

X_train, X_test, y_train, y_test = train_test_split(
    X, labels, test_size=0.3, random_state=42
)

model = MultinomialNB()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, predictions))

new_email = ["Click here to verify your account"]
new_email_vector = vectorizer.transform(new_email)

result = model.predict(new_email_vector)

print("Prediction:", result[0])
