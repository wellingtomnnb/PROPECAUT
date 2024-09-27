import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('cleveland.csv', header = None)

df.columns = ['age', 'sex', 'cp', 'trestbps', 'chol',
              'fbs', 'restecg', 'thalach', 'exang', 
              'oldpeak', 'slope', 'ca', 'thal', 'target']


################################## data preprocessing ####################################################

### Mapeia as classes transformando o problema em binário
df['target'] = df.target.map({0: 0, 1: 1, 2: 1, 3: 1, 4: 1})

### Verifica se há valores nulos
print(df.isnull().sum())

### Substitui os valores nulos pela média ##
# Inserir aqui o codigo

############################################

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

## Separa os dados em treinamento e teste
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

### Normaliza os dados ###############################
# Inserir aqui o codigo


######################################################

### Aplica PCA #######################################
# Deve ser utilizado somente no item 7 do exercício
# Inserir aqui o codigo


######################################################

#########################################   Naive Bayes  #############################################################

# Treinamento Modelo
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Teste modelo
y_pred = classifier.predict(X_test)

# Resultados
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix

cm_test = confusion_matrix(y_pred, y_test)
print(cm_test)

disp = plot_confusion_matrix(classifier, X_test, y_test, cmap=plt.cm.Blues)
disp.ax_.set_title('Naive Bayes - Test')
print(disp.confusion_matrix)
plt.show()

y_pred_train = classifier.predict(X_train)
cm_train = confusion_matrix(y_pred_train, y_train)
print(cm_train)

disp = plot_confusion_matrix(classifier, X_train, y_train, cmap=plt.cm.Blues)
disp.ax_.set_title('Naive Bayes - Train')
print(disp.confusion_matrix)
plt.show()

print()
print('Accuracy for training set for Naive Bayes = {}'.format((cm_train[0][0] + cm_train[1][1])/len(y_train)))
print('Accuracy for test set for Naive Bayes = {}'.format((cm_test[0][0] + cm_test[1][1])/len(y_test)))