## Inspired by https://www.datacamp.com/tutorial/introduction-to-shap-values-machine-learning-interpretability

import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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

tabela = pd.read_excel('featureset3_aug.xlsx')

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

#########################################################

## Shap values array
num_classes = len(shap_values)
features = X_teste.columns

# Calculating average feature importance for each class
mean_shap_importance = np.mean([np.abs(shap_values[i]).mean(axis=0) for i in range(num_classes)], axis=0)

# Sorting features by average imortance 
sorted_indices = np.argsort(mean_shap_importance)[::-1]  

# Selecting the most important features
top_n = 20
top_features = features[sorted_indices[:top_n]]

# Updating X_teste and shap_values variables with only these features 
X_teste_top = X_teste[top_features]
shap_values_top = [shap_values[i][:, sorted_indices[:top_n]] for i in range(num_classes)]

# Number of features
num_features = len(top_features)

# Determine the angles for the polar plot
angles = np.linspace(0, 2 * np.pi, num_features, endpoint=False).tolist()
angles += angles[:1]  

# Polar plot for each class
fig, axs = plt.subplots(2, 2, figsize=(12, 12), subplot_kw=dict(polar=True))

colors = ['blue', 'green', 'red', 'purple']

axs = axs.ravel()

for i in range(num_classes):
    shap_importance_class = np.abs(shap_values_top[i]).mean(axis=0)
    
    shap_importance_class = np.concatenate((shap_importance_class, [shap_importance_class[0]]))
    
    axs[i].fill(angles, shap_importance_class, color=colors[i], alpha=0.25)
    axs[i].plot(angles, shap_importance_class, color=colors[i], linewidth=2)
    
    axs[i].set_xticks(angles[:-1])
    axs[i].set_xticklabels(top_features, fontsize=10)
    
    axs[i].set_title(f'Class {i+1}', size=15)

plt.tight_layout()
plt.show()
 

#Dependence Plot
#shap.dependence_plot("mfccs5", shap_values[2], X_teste,interaction_index="hi_freq")

#SHAP Force Plot
#shap.plots.force(explainer.expected_value[0], shap_values[0][0,:], X_teste.iloc[0, :], matplotlib = True)
