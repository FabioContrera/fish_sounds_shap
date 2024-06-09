import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier

# Loading data
tabela = pd.read_excel('featureset_global2.xlsx')

tabela['random_var'] = np.random.random(len(tabela))

# Matrix
previsores = tabela.drop(columns=['class'])
classe = tabela['class']

# Data splitting
X_treinamento, X_teste, y_treinamento, y_teste = train_test_split(previsores,
                                                                  classe,
                                                                  test_size=0.3,
                                                                  random_state=42)

# Naive bayes
naive_bayes = GaussianNB()

accuracy_NB = cross_val_score(naive_bayes, previsores, classe, cv=5, scoring='accuracy')
precision_NB = cross_val_score(naive_bayes, previsores, classe, cv=5, scoring='precision_macro')
recall_NB = cross_val_score(naive_bayes, previsores, classe, cv=5, scoring='recall_macro')
f1_NB = cross_val_score(naive_bayes, previsores, classe, cv=5, scoring='f1_macro')

print("Naive Bayes Cross-Validation Accuracy: %0.2f (+/- %0.2f)" % (accuracy_NB.mean(), accuracy_NB.std() * 2))
print("Naive Bayes Cross-Validation Precision: %0.2f (+/- %0.2f)" % (precision_NB.mean(), precision_NB.std() * 2))
print("Naive Bayes Cross-Validation Recall: %0.2f (+/- %0.2f)" % (recall_NB.mean(), recall_NB.std() * 2))
print("Naive Bayes Cross-Validation f1: %0.2f (+/- %0.2f)" % (f1_NB.mean(), f1_NB.std() * 2))



# Random Forest
forest = ExtraTreesClassifier()

accuracy_RF = cross_val_score(forest, previsores, classe, cv=5, scoring='accuracy')
precision_RF = cross_val_score(forest, previsores, classe, cv=5, scoring='precision_macro')
recall_RF = cross_val_score(forest, previsores, classe, cv=5, scoring='recall_macro')
f1_RF = cross_val_score(forest, previsores, classe, cv=5, scoring='f1_macro')

print("Random Forest Cross-Validation Accuracy: %0.2f (+/- %0.2f)" % (accuracy_RF.mean(), accuracy_RF.std() * 2))
print("Random Forest Cross-Validation Precision: %0.2f (+/- %0.2f)" % (precision_RF.mean(), precision_RF.std() * 2))
print("Random Forest Cross-Validation Recall: %0.2f (+/- %0.2f)" % (recall_RF.mean(), recall_RF.std() * 2))
print("Random Forest Cross-Validation f1: %0.2f (+/- %0.2f)" % (f1_RF.mean(), f1_RF.std() * 2))


# Decision Tree
arvore = DecisionTreeClassifier()

accuracy_DT = cross_val_score(arvore, previsores, classe, cv=5, scoring='accuracy')
precision_DT = cross_val_score(arvore, previsores, classe, cv=5, scoring='precision_macro')
recall_DT = cross_val_score(arvore, previsores, classe, cv=5, scoring='recall_macro')
f1_DT = cross_val_score(arvore, previsores, classe, cv=5, scoring='f1_macro')

print("Decision Tree Cross-Validation Accuracy: %0.2f (+/- %0.2f)" % (accuracy_DT.mean(), accuracy_DT.std() * 2))
print("Decision Tree Cross-Validation Precision: %0.2f (+/- %0.2f)" % (precision_DT.mean(), precision_DT.std() * 2))
print("Decision Tree Cross-Validation Recall: %0.2f (+/- %0.2f)" % (recall_DT.mean(), recall_DT.std() * 2))
print("Decision Tree Cross-Validation f1: %0.2f (+/- %0.2f)" % (f1_DT.mean(), f1_DT.std() * 2))


