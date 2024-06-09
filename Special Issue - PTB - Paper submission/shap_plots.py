## Inspired by https://www.datacamp.com/tutorial/introduction-to-shap-values-machine-learning-interpretability

import shap
import pandas as pd
import numpy as np
import matplotlib as plt
shap.initjs()
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

tabela = pd.read_excel('only_timbre_aug.xlsx')

#tabela['random_var'] = np.random.random(len(tabela))

# Matrix
previsores = tabela.drop(columns=['class'])
classe     = tabela['class']

# Data splitting
X_treinamento, X_teste, y_treinamento, y_teste = train_test_split(previsores,
                                                                  classe,
                                                                  test_size = 0.3,
                                                                  random_state = 42)

# Model setting and training
clf = RandomForestClassifier()
clf.fit(X_treinamento, y_treinamento)

# Prediction using test data
y_pred = clf.predict(X_teste)

# Classification Report
print(classification_report(y_pred, y_teste))

#SHAP Explainer
explainer = shap.Explainer(clf)
shap_values = explainer.shap_values(X_teste)

#SHAP Summary Plot
class_names = ["Class 1", "Class 2", "Class 3", "Class 4"]
shap.summary_plot(shap_values, X_teste, class_names=class_names)


shap.summary_plot(shap_values[0], X_teste)
 

#Dependence Plot
#shap.dependence_plot("mfccs5", shap_values[2], X_teste,interaction_index="hi_freq")

#SHAP Force Plot
#shap.plots.force(explainer.expected_value[0], shap_values[0][0,:], X_teste.iloc[0, :], matplotlib = True)


