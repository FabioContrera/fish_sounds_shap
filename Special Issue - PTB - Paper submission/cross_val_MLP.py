from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd

# Loading data
tabela = pd.read_excel('MFCCs.xlsx')

# Matrix
X = tabela.drop(columns=['class'])
y = tabela['class']

# Data splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=0.3)


sc_X = StandardScaler()
X_train_scaled = sc_X.fit_transform(X_train)
X_test_scaled = sc_X.transform(X_test)

# MLP Classifier
model = MLPClassifier(hidden_layer_sizes=(256, 128, 64, 32),
                      activation='relu',
                      solver='adam',
                      alpha=0.1,
                      batch_size=8,
                      learning_rate='constant',
                      max_iter=200,
                      random_state=1,
                      warm_start=True,
                      early_stopping=False)

# Cross-validation
cv_scores_accuracy = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
cv_scores_precision = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='precision_macro')
cv_scores_recall = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='recall_macro')
cv_scores_f1 = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='f1_macro')

# Print results
print("Cross-Validation Accuracy: %0.2f (+/- %0.2f)" % (cv_scores_accuracy.mean(), cv_scores_accuracy.std() * 2))
print("Cross-Validation Precision: %0.2f (+/- %0.2f)" % (cv_scores_precision.mean(), cv_scores_precision.std() * 2))
print("Cross-Validation Recall: %0.2f (+/- %0.2f)" % (cv_scores_recall.mean(), cv_scores_recall.std() * 2))
print("Cross-Validation F1-score: %0.2f (+/- %0.2f)" % (cv_scores_f1.mean(), cv_scores_f1.std() * 2))