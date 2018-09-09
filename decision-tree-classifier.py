from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import accuracy_score

# Load dataset
data = load_breast_cancer()

# Organize our data
label_names = data['target_names']
labels = data['target']
feature_names = data['feature_names']
features = data['data']

# Split our data
train, test, train_labels, test_labels = train_test_split(features,
                                                          labels,
                                                          test_size=0.33,
                                                          random_state=42)

# Initialize our classifier
clf = tree.DecisionTreeClassifier()

# Train our classifier
clf = clf.fit(train, train_labels)

# Make predictions
preds = clf.predict(test)
print(preds)

# Evaluate accuracy
print(accuracy_score(test_labels, preds))
