import os
from PIL import Image
from io import BytesIO
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
warnings.filterwarnings('ignore')
plt.style.use('fivethirtyeight')
# TITANIC DATASET - https://www.kaggle.com/competitions/titanic
# NOTEBOOK - https://www.kaggle.com/code/alexisbcook/titanic-tutorial https://www.kaggle.com/code/adhamtarek147/titanic-survival-classification
# Titanic veri seti, yolcuların özelliklerini ve hayatta kalma durumlarını içerir.
# 15 Nisan 1912'de, Titanic adlı gemi, ilk seferinde bir buz dağına çarptı ve battı.
# Gemide 2224 yolcu ve mürettebat vardı.
# 1502 yolcu ve mürettebat öldü.
# Bu faciadan sadece 705 yolcu kurtuldu.
# Bu facia, tarihteki en ölümcül deniz kazalarından biri olarak kabul edilir.
# Bu veri seti, yolcuların özelliklerini (yaş, cinsiyet, sınıf vb.) ve hayatta kalma durumlarını içerir.

# Veri seti hakkında bilgilendirme
# survival: Survival	0 = No, 1 = Yes
# embarked	Port of Embarkation	C = Cherbourg, Q = Queenstown, S = Southampton

# pclass: A proxy for socio-economic status (SES) Ticket class	1 = 1st, 2 = 2nd, 3 = 3rd
# 1st = Upper
# 2nd = Middle
# 3rd = Lower

# age: Age is fractional if less than 1. If the age is estimated, is it in the form of xx.5
# sibsp: The dataset defines family relations in this way... # of siblings / spouses aboard the Titanic
# Sibling = brother, sister, stepbrother, stepsister

# Spouse = husband, wife (mistresses and fiancés were ignored)
# parch: The dataset defines family relations in this way... # of parents / children aboard the Titanic
# Parent = mother, father
# Child = daughter, son, stepdaughter, stepson
# Some children travelled only with a nanny, therefore parch=0 for them.

train_df = pd.read_csv('data\\titanic_train.csv')
test_df = pd.read_csv('data\\titanic_test.csv')

print('Train data shape: ', train_df.shape)
print(train_df.isnull().sum())
print(train_df.head())
print('------------------------------------')
print('Test data shape: ', test_df.shape)
print(test_df.isnull().sum())
print(test_df.head())
print('------------------------------------')

# Kurtulan kadınların oranı
women = train_df.loc[train_df.Sex == 'female']["Survived"]
rate_women = sum(women)/len(women)
print("% Kurtulan kadınların oranı: ", rate_women)
print('------------------------------------')
# Kurtulan erkeklerin oranı
men = train_df.loc[train_df.Sex == 'male']["Survived"]
rate_men = sum(men)/len(men)
print("% Kurtulan erkeklerin oranı: ", rate_men)
print('------------------------------------')

# Eksik veri analizi veri ön işleme
# 'Age', 'Fare', ve 'Embarked' sütunlarındaki eksik verileri dolduralım
train_df['Age'] = train_df['Age'].fillna(train_df['Age'].median())
train_df['Fare'] = train_df['Fare'].fillna(train_df['Fare'].median())
train_df['Embarked'] = train_df['Embarked'].fillna(train_df['Embarked'].mode()[0])

# Test veri setindeki eksik verileri aynı şekilde dolduralım
test_df['Age'] = test_df['Age'].fillna(test_df['Age'].median())
test_df['Fare'] = test_df['Fare'].fillna(test_df['Fare'].median())
test_df['Embarked'] = test_df['Embarked'].fillna(test_df['Embarked'].mode()[0])

# Eksik veri kontrolü
print(train_df.isnull().sum())
print(test_df.isnull().sum())

# Kategorik değişkenleri sayısal verilere dönüştürelim
train_df['Sex'] = LabelEncoder().fit_transform(train_df['Sex'])
train_df['Embarked'] = LabelEncoder().fit_transform(train_df['Embarked'])
test_df['Sex'] = LabelEncoder().fit_transform(test_df['Sex'])
test_df['Embarked'] = LabelEncoder().fit_transform(test_df['Embarked'])

# Gerekli sütunları seçelim
X_train = train_df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
y_train = train_df['Survived']
X_test = test_df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Eksik veri kontrolü
print(train_df.isnull().sum())
print(test_df.isnull().sum())

print('KNN MODELİ')
print('------------------------------------')
# Farklı parametrelerle KNN uygulama
k_values = [3, 5, 7]
error_rate = []
results = []
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    y_pred_train = knn.predict(X_train_scaled)
    accuracy = accuracy_score(y_train, y_pred_train)
    report = classification_report(y_train, y_pred_train, output_dict=True)
    results.append({
        'k': k,
        'accuracy': accuracy,
        'precision_0': report['0']['precision'],
        'recall_0': report['0']['recall'],
        'f1_score_0': report['0']['f1-score'],
        'precision_1': report['1']['precision'],
        'recall_1': report['1']['recall'],
        'f1_score_1': report['1']['f1-score'],
    })

    # Hata matrisini ve performans metriklerini gösterme
    cm = confusion_matrix(y_train, y_pred_train)
    print(f'K={k} (Train Set)')
    print(confusion_matrix(y_train, y_pred_train))
    print(classification_report(y_train, y_pred_train))
    print('------------------------------------')
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix for k={k} (Train - Test ayrı setler)')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    # Görsellerin kaydedilmesi
    # Saving the plot using savefig()
    plt.savefig(f'output\\test-train-output-confusion-matrix-k-{k}.png')
    print(f'output\\test-train-output-confusion-matrix-k-{k}.png - Başarılı bir şekilde kaydedildi.')
    plt.show()
    error_rate.append(np.mean(y_pred_train != y_train))

    # Test seti için tahminler
    y_pred_test = knn.predict(X_test_scaled)
    test_df[f'Prediction_k={k}'] = y_pred_test

# Tahminleri içeren test veri setini kaydet
test_df.to_csv('output\\tahmin-test-train-output.csv', index=False)
print("output\\tahmin-test-train-output.csv - Başarılı bir şekilde kaydedildi.")


plt.figure(figsize=(10, 6))
plt.plot(k_values, error_rate, color='blue', linestyle='dashed', marker='o', markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value (Train - Test ayrı setler)')
plt.xlabel('K')
print('En uygun K değeri: ', k_values[error_rate.index(min(error_rate))])
plt.legend(['Error Rate'])
plt.grid(True)
plt.savefig(f'output\\test-train-output-error-rate.png')
print('output\\test-train-output-error-rate.png - Başarılı bir şekilde kaydedildi.')
plt.show()


# Sonuçları görselleştirme
results_df = pd.DataFrame(results)
print(results_df)
print('------------------------------------')

# Doğruluk görselleştirmesi
plt.figure(figsize=(10, 6))
plt.plot(results_df['k'], results_df['accuracy'], marker='o', label='Accuracy')
plt.xlabel('k value')
plt.ylabel('Score')
plt.title('Accuracy for different k values (Train - Test ayrı setler)')
plt.legend()
plt.grid(True)
plt.savefig(f'output\\test-train-output-accuracy-k.png')
print('output\\test-train-output-accuracy-k.png - Başarılı bir şekilde kaydedildi.')
plt.show()

# Precision, Recall ve F1-Score görselleştirmesi
metrics = ['precision_0', 'recall_0', 'f1_score_0', 'precision_1', 'recall_1', 'f1_score_1']
plt.figure(figsize=(10, 6))
for metric in metrics:
    plt.plot(results_df['k'], results_df[metric], marker='o', label=metric)
plt.xlabel('k value')
plt.ylabel('Score')
plt.title('Precision, Recall and F1-Score for different k values (Train - Test ayrı setler)')
plt.legend()
plt.grid(True)
plt.savefig(f'output\\test-train-output-precision-recall-f1-score-k.png')
print('output\\test-train-output-precision-recall-f1-score-k.png - Başarılı bir şekilde kaydedildi.')
plt.show()

print('------------------------------------')
print('RANDOM FOREST MODELİ')
print('------------------------------------')
y = train_df["Survived"]
features = ["Pclass", "Sex", "SibSp", "Parch"]
X = pd.get_dummies(train_df[features])
X_test = pd.get_dummies(test_df[features])
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X, y)
predictions = model.predict(X_test)
output = pd.DataFrame({'PassengerId': test_df.PassengerId, 'Survived': predictions})
output.to_csv('output\\random-forest-test-train-output.csv', index=False)
print("output\\test-train-random-forest-output.csv - Başarılı bir şekilde kaydedildi.")
print('------------------------------------')



