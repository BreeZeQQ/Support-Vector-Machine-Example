import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

###Dataseti import ettik
winedata = pd.read_csv("wine-all.csv")

###Class Sütununu droplar kalan dataları X e eşitler.
X = winedata.drop('Class', axis=1)

###Datasetin Class bilgileri burada tutulur.
y = winedata['Class']

###Bu kısımda import ettiğimiz tüm verileri train ve test olarak ayırdık ve
###X_train, X_test, y_train, y_test değişkenlerine eşitledik. Train_Size = 0.80 Test_Size = 0.20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)

###Train işlemi için Support Vector Classifier kütüphanesini kullanıdık.
svclassifier = SVC(kernel='linear')

###Train datalarımız ile algorimamızı eğittik
svclassifier.fit(X_train, y_train)


###Test için ayırdığımız verilerle test yaptık ve sonucları y_pred adlı değişkene aktardık.
y_pred = svclassifier.predict(X_test)

###Test verilerine hangi class dediğini burdan görebiliriz istersek.
#print(y_pred)

###Test verileri ile tahmin verilerimizi confusion matrixe döken kod.
print("\n<<<<< CONFUSION MATRIX >>>>> \n")
print(confusion_matrix(y_test,y_pred))

print("\n<<<<< DETAYLI RAPOR >>>>>\n")
print(classification_report(y_test,y_pred))
