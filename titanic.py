import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler


warnings.filterwarnings('ignore')
plt.style.use('fivethirtyeight')
sns.set(style='whitegrid')

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

# Veri setini okuma ve inceleme
df = pd.read_csv('data\\titanic_train.csv')
print('Veri seti boyutu (satır, sütun): ', df.shape)
print("Eksik veri sayısı: \n", df.isnull().sum())
print(df.head())

print('------------------------------------')
# Kurtulan kadınların oranı
women = df.loc[df.Sex == 'female']["Survived"]
rate_women = sum(women)/len(women)
print("Kurtulan kadınların oranı:")
print("Toplam kadın:", len(women))
print("Kurtulan kadın:", sum(women))
print("% Oran: ", rate_women)
print('------------------------------------')
# Kurtulan erkeklerin oranı
men = df.loc[df.Sex == 'male']["Survived"]
rate_men = sum(men)/len(men)
print("Kurtulan erkeklerin oranı:")
print("Toplam erkek:", len(men))
print("Kurtulan erkek:", sum(men))
print("% Oran:", rate_men)
print('------------------------------------')

# Eksik veri analizi ve veri ön işleme
# 'Age', 'Fare', ve 'Embarked' sütunlarındaki eksik verileri dolduralım
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Fare'] = df['Fare'].fillna(df['Fare'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# Eksik veri kontrolü
print("Eksik veri sayısı: \n", df.isnull().sum())

# Kategorik verileri sayısal verilere dönüştürme
df['Sex'] = LabelEncoder().fit_transform(df['Sex'])
df['Embarked'] = LabelEncoder().fit_transform(df['Embarked'])

# Veri setini özellikler ve etiketler olarak ayırma
X = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
y = df['Survived']

# Veri setini eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Verileri ölçeklendirme
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print('K EN YAKIN KOMŞU (KNN) MODELİ')
print('------------------------------------')
# Farklı K değerleri ile KNN Algoritması
k_values = [3, 5, 7]
error_rate = []
results = []
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k, metric='minkowski', p=2)
    knn.fit(X_train_scaled, y_train)
    y_pred_train = knn.predict(X_train_scaled)
    accuracy = accuracy_score(y_train, y_pred_train)
    report = classification_report(y_train, y_pred_train, output_dict=True)
    cm = confusion_matrix(y_train, y_pred_train)
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
    print(f'K={k} Eğitim Seti Performansı')
    print(confusion_matrix(y_train, y_pred_train))
    print(classification_report(y_train, y_pred_train))
    print(accuracy_score(y_train, y_pred_train))
    print('------------------------------------')
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Hata Matrisi K={k}')
    plt.xlabel('Tahmini')
    plt.ylabel('Gerçek')
    # Görsellerin kaydedilmesi
    # Saving the plot using savefig()
    plt.savefig(f'output\\output-confusion-matrix-k-{k}.png')
    print(f'output\\output-confusion-matrix-k={k}.png - Başarılı bir şekilde kaydedildi.')
    plt.show()
    # Hata oranını hesaplama
    error_rate.append(np.mean(y_pred_train != y_train))

    # Test seti için tahminler
    y_pred_test = knn.predict(X_test_scaled)
    df_test = pd.DataFrame({'Gerçek': y_test, 'Tahmini': y_pred_test})
    df_test[f'Prediction_k={k}'] = y_pred_test
    # Tahminleri içeren test veri setini kaydet
    df_test.to_csv(f'output\\tahmin-output-{k}.csv', index=False)
    print(f'output\\tahmin-output-{k}.csv - Başarılı bir şekilde kaydedildi.')
    print('------------------------------------')

# Tahminlerin analizi
print('Tahminlerin analizi')
print('------------------------------------')
for k in k_values:
    df_test = pd.read_csv(f'output\\tahmin-output-{k}.csv')
    print(f'K={k} Test Seti Performansı')
    print(confusion_matrix(df_test['Gerçek'], df_test[f'Prediction_k={k}']))
    print(classification_report(df_test['Gerçek'], df_test[f'Prediction_k={k}']))
    print('------------------------------------')

    # Görselleştirmeler
    plt.figure(figsize=(6, 4))
    sns.heatmap(confusion_matrix(df_test['Gerçek'], df_test[f'Prediction_k={k}']), annot=True, fmt='d', cmap='Blues')
    plt.title(f'Hata Matrisi Test Seti K={k}')
    plt.xlabel('Tahmini')
    plt.ylabel('Gerçek')
    plt.savefig(f'output\\output-confusion-matrix-k-{k}-test.png')
    print(f'output\\output-confusion-matrix-k={k}-test.png - Başarılı bir şekilde kaydedildi.')
    plt.show()


# Hata oranını görselleştirme
plt.figure(figsize=(10, 6))
plt.plot(k_values, error_rate, color='blue', linestyle='dashed', marker='o', markerfacecolor='red', markersize=10)
plt.title('Hata Oranı vs. K Değeri')
plt.xlabel('K')
print('En uygun K değeri: ', k_values[error_rate.index(min(error_rate))])
plt.legend(['Hata Oranı', 'K Değeri'])
plt.grid(True)
plt.savefig(f'output\\output-error-rate.png')
print('output\\output-error-rate.png - Başarılı bir şekilde kaydedildi.')
plt.show()

# Sonuçları görselleştirme
results_df = pd.DataFrame(results)
print(results_df)
print('------------------------------------')

# Doğruluk görselleştirmesi
plt.figure(figsize=(10, 6))
plt.plot(results_df['k'], results_df['accuracy'], marker='o', label='Accuracy')
plt.xlabel('k değeri')
plt.ylabel('Skor')
plt.title('Farklı k değerleri için doğruluk')
plt.legend()
plt.grid(True)
plt.savefig(f'output\\output-accuracy.png')
print('output\\output-accuracy.png - Başarılı bir şekilde kaydedildi.')
plt.show()

# Precision, Recall ve F1-Score görselleştirmesi
metrics = ['precision_0', 'recall_0', 'f1_score_0', 'precision_1', 'recall_1', 'f1_score_1']
plt.figure(figsize=(10, 6))
for metric in metrics:
    plt.plot(results_df['k'], results_df[metric], marker='o', label=metric)
plt.xlabel('k değeri')
plt.ylabel('Skor')
plt.title('Farklı k değerleri için Hassasiyet, Duyarlılık ve F1-Skoru')
plt.legend()
plt.grid(True)
plt.savefig(f'output\\output-precision-recall-f1-score.png')
print('output\\output-precision-recall-f1-score.png - Başarılı bir şekilde kaydedildi.')
plt.show()

print('------------------------------------')
print('RASTGELE ORMAN MODELİ')
y = df["Survived"]
features = ["Pclass", "Sex", "SibSp", "Parch"]
X = pd.get_dummies(df[features])
X_test = pd.get_dummies(df[features])
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X, y)
predictions = model.predict(X_test)
output = pd.DataFrame({'PassengerId': df.PassengerId, 'Survived': predictions})
output.to_csv('output\\output-random.csv', index=False)
print("output\\random-forest-output.csv - Başarılı bir şekilde kaydedildi.")
print('------------------------------------')
# Öznitelik önem derecelerini görselleştirme
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]
plt.figure(figsize=(10, 6))
plt.title("Öznitelik Önem Dereceleri")
plt.bar(range(X.shape[1]), importances[indices], align="center")
plt.xticks(range(X.shape[1]), X.columns[indices], rotation=90)
plt.xlim([-1, X.shape[1]])
plt.savefig(f'output\\output-feature-importances.png')
print('output\\output-feature-importances.png - Başarılı bir şekilde kaydedildi.')
plt.show()
print('------------------------------------')
# Rasgele orman modeli performans metrikleri
print("F1 Score: ", f1_score(y, predictions))
print("Accuracy: ", accuracy_score(y, predictions))
print("Confusion Matrix: \n", confusion_matrix(y, predictions))
print("Classification Report: \n", classification_report(y, predictions))