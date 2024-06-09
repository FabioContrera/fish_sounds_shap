import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

## Loading data

tabela = pd.read_excel('only_timbre.xlsx')

tabela['random_var'] = np.random.random(len(tabela))

# Matrix
previsores = tabela.drop(columns=['class'])
classe     = tabela['class']

# Data splitting
X_treinamento, X_teste, y_treinamento, y_teste = train_test_split(previsores,
                                                                  classe,
                                                                  test_size = 0.3,
                                                                  random_state = 42)

# Naive bayes -----------------------------------------------------------------------
# Setting and training the model (probabilities matrix)
naive_bayes = GaussianNB()
naive_bayes.fit(X_treinamento, y_treinamento)

# Previsision using test
previsoesNB = naive_bayes.predict(X_teste)

taxa_acertoNB = accuracy_score(y_teste, previsoesNB)
prc_NB = precision_score(y_teste, previsoesNB, average='macro')
rcl_NB = recall_score(y_teste, previsoesNB, average='macro')
f1_NB = f1_score(y_teste, previsoesNB, average='macro')

# Random forest ---------------------------------------------------------------------
# Extracting feature importance
forest = ExtraTreesClassifier()
forest.fit(X_treinamento, y_treinamento)
importancias = forest.feature_importances_

# Model setting, training, and prediction
floresta = RandomForestClassifier(n_estimators = 100)
floresta.fit(X_treinamento, y_treinamento)

floresta.estimators_

previsoesRF   = floresta.predict(X_teste)
confusao = confusion_matrix(y_teste, previsoesRF)
taxa_acertoRF = accuracy_score(y_teste, previsoesRF)

prc_RF = precision_score(y_teste, previsoesRF, average='macro')
rcl_RF = recall_score(y_teste, previsoesRF, average='macro')
f1_RF = f1_score(y_teste, previsoesRF, average='macro')

# Decision tree ---------------------------------------------------------------------
# Model setting and training
arvore = DecisionTreeClassifier()
arvore.fit(X_treinamento, y_treinamento)
previsoesDT = arvore.predict(X_teste)

taxa_acertoDT = accuracy_score(y_teste, previsoesDT)
prc_DT = precision_score(y_teste, previsoesDT, average='macro')
rcl_DT = recall_score(y_teste, previsoesDT, average='macro')
f1_DT = f1_score(y_teste, previsoesDT, average='macro')


#Printing the metrics
print('Accuracy NB:',100*taxa_acertoNB, ', Precision NB:',100*prc_NB, 
      ', Recall NB:', 100*rcl_NB, ', f1 NB:',100*f1_NB, ', Accuracy RF:',100*taxa_acertoRF,
      ', Precision RF:',100*prc_RF, ', Recall RF:',100*rcl_RF, ', f1 RF:',100*f1_RF, 
      ', Accuracy DT:',100*taxa_acertoDT, ', Precision DT:',100*prc_DT, 
      ', Recall DT:',100*rcl_DT, ', f1 DT:',100*f1_DT)


##-----------------------PLOTS--------------------------##

##FEATURE IMPORTANCE
#Random Forest
# the permutation based importance

feature_names = previsores.columns
perm_importance = permutation_importance(forest, X_teste, y_teste)

plt.figure(1)
sorted_idx = importancias.argsort()

plt.barh(feature_names[sorted_idx], importancias[sorted_idx])
plt.grid(False)
plt.title("Feature importance", fontsize='12')
plt.xlabel("Importances",fontsize='11')
plt.yticks(fontsize=8)
plt.xticks(fontsize=10)

plt.figure(2)
acc_NB = 100*taxa_acertoNB
precision_NB = 100*prc_NB
recall_NB = 100*rcl_NB
f1_score_NB = 100*f1_NB

acc_RF = 100*taxa_acertoRF
precision_RF = 100*prc_RF
recall_RF = 100*rcl_RF
f1_score_RF = 100*f1_RF

acc_DT = 100*taxa_acertoDT
precision_DT = 100*prc_DT
recall_DT = 100*rcl_DT
f1_score_DT = 100*f1_DT

#plot
barWidth = 0.15

bars1 = (acc_NB, acc_RF,acc_DT)
bars2 = (precision_NB, precision_RF,precision_DT)
bars3 = (recall_NB, recall_RF,recall_DT)
bars4 = (f1_score_NB, f1_score_RF, f1_score_DT)
 
r1 = np.arange(len(bars1))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]
r4 = [x + barWidth for x in r3]

plt.bar(r1, bars1, color='#5e60ce', width=barWidth, edgecolor='white', label='Accuracy')
plt.bar(r2, bars2, color='#4ea8de', width=barWidth, edgecolor='white', label='Precision')
plt.bar(r3, bars3, color='#56cfe1', width=barWidth, edgecolor='white', label='Recall')
plt.bar(r4, bars4, color='#80ffdb', width=barWidth, edgecolor='white', label='f1 score')
 
# Add xticks on the middle of the group bars
plt.xlabel('Algorithm', fontweight='bold')
plt.xticks([r + barWidth for r in range(len(bars1))], ['NB', 'RF', 'DT'])
plt.yticks(range(0,100,10)) 
# Create legend & Show graphic
#plt.legend(loc='upper right')
plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
plt.tight_layout()
plt.savefig('Performance_supervised_.jpg', dpi=300)
plt.show()