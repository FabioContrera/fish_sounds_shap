## Importando bibliotecas
from sklearn.neural_network import MLPClassifier 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score,recall_score, f1_score, confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

## Loading data
tabela = pd.read_excel('only_timbre.xlsx')

## Matrix
X = tabela.drop(columns=['class'])
y    = tabela['class']

X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=1, test_size=0.3)

sc_X = StandardScaler()
X_trainscaled=sc_X.fit_transform(X_train)
X_testscaled=sc_X.transform(X_test)


model = MLPClassifier(hidden_layer_sizes=(256,128,64,32), 
                      activation='relu', 
                      solver='adam', 
                      alpha=0.1, 
                      batch_size=8, 
                      learning_rate='constant', 
                      max_iter=200, 
                      random_state=1, 
                      warm_start=True, 
                      early_stopping=False)
model.fit(X_trainscaled, y_train)
y_pred = model.predict(X_testscaled)
accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)

#print(model.score(X_test, y_test))

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy*100}")

precision = precision_score(y_test, y_pred, average='macro')
print(f"Precision: {precision*100}")

recall = recall_score(y_test, y_pred, average='macro')
print(f"Recall: {recall*100}")

f1 = f1_score(y_test, y_pred, average='macro')
print(f"f1_score: {f1*100}")


plt.figure(1)
plt.plot(model.loss_curve_)
plt.title('Loss curve')
plt.savefig('Loss curve_MLP.jpg', dpi=300)
plt.show()



cm = confusion_matrix(y_test,y_pred)

plt.figure(2)
sns.heatmap(cm, cmap=sns.color_palette("Blues", as_cmap=True), annot=True, linewidth=.5)
plt.title('Confusion matrix')
plt.savefig('Confusion Matrix_MLP.jpg', dpi=300)
plt.show()

#plt.plot(accuracy)
#plt.show()

