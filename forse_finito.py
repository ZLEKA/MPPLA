import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from fbprophet import Prophet

from sklearn.ensemble import RandomForestRegressor
#
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import ConfusionMatrixDisplay
import time

import sklearn.metrics as metrics

#
from sklearn.metrics import classification_report 
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score 

#




from sklearn.svm import SVC
from sklearn.metrics import mean_absolute_error

#

from sklearn.svm import SVR 




#
from sklearn.metrics import mean_squared_error

#numero random per alberi
SEED=42 

data_set = pd.read_csv("avocado.csv")

data_set.head()
print(data_set.head())
data=data_set.copy()
#estraggo il numero di colonne e righe

row, columns = data.shape

data.corr()

liste = []

#ripilisco le date dal - 
for date in data.Date:
    liste.append(date.split("-"))
    
#split month and day adding to lists
month = []
day = []
for i in range(len(liste)):
    month.append(liste[i][1])
    day.append(liste[i][2])
    
# adding to dataset
data["month"] = month
data["day"] = day
data.drop(["Date"],axis=1,inplace=True)

#convert objects to int
data.month = data.month.values.astype(int)
data.day = data.day.values.astype(int)

# drop unnecessary features
data.drop(["Unnamed: 0"],axis=1,inplace=True)
#converte la colonna type in volre booleano
data["month"].unique()


#y
y = data["AveragePrice"].copy()

# x
x=data
x.drop(["AveragePrice"],axis=1,inplace=True)

x = pd.get_dummies(data = x, columns = ["type", "region", "month", "year"], prefix = ["type", "region", "month", "year"], drop_first = True)

#funzione scalare
sc = StandardScaler()
# Scale the data to be between -1 and 1
x = sc.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=SEED, shuffle = True)
model_scores = {"train" : [],
                "test" : [],
                "mae" : [],
                "mse" : [],
                "rmse" : []}

def get_results(clf):
    clf.fit(x_train, y_train)
    train_score = clf.score(x_train, y_train)
    model_scores["train"].append(train_score)
    pred = clf.predict(x_test)
    test_score = clf.score(x_test, y_test)
    model_scores["test"].append(test_score)
    #stimo l'errore medio del valore assoluto sul totale dei valori
    mae = mean_absolute_error(pred, y_test)
    model_scores["mae"].append(mae)
    #stimo l'errore medio in rapporto al numero dei valori
    mse = mean_squared_error(pred, y_test)
    model_scores["mse"].append(mse)
    rmse = np.sqrt(mse)
    model_scores["rmse"].append(rmse)
    print("train score: {0:.4f}\nTest score: {1:.4f}\nMAE: {2:.4f}\nMSE: {3:.4f}\nRMSE: {4:.4f}".format(train_score, test_score, mae, mse, rmse))
    plt.figure(figsize = (13, 6))
    plt.subplot(1, 2, 1)
    plt.title("y_true x y_pred", size = 14)
    plt.scatter(y_test, pred, color = "mediumseagreen")
    plt.xlabel("y_true")
    plt.ylabel("y_pred")
    plt.subplot(1, 2, 2)
    plt.title("Residuals (y_true - y_pred)", size = 14)
    sns.histplot((y_test-pred), kde = True, color = "green")
    plt.show()

  

    
      
#Random Forset Regressor
rfr = RandomForestRegressor(random_state = SEED)
print("Random Forest\n\n")
get_results(rfr)

#Support Vector Regressor
svr = SVR(kernel='linear')

print("Support Vector Machine (SVM)\n\n")
#get_results(svr)




data_set.groupby('type').groups

PREDICTION_TYPE = 'conventional'
df = data_set[data_set.type == PREDICTION_TYPE]
df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')

#previsione dei prezzi su un singolo stato-----------------------

regions = df.groupby(df.region)
#print("Total regions :", len(regions))
#print("-------------")

def printg(dp):
    
    m = Prophet()
    m.fit(dp)
    future = m.make_future_dataframe(periods=365)
    forecast = m.predict(future)
    print("------Stima andamento prezzi nel 2019\n")
    m.plot(forecast)
    time.sleep(5.0)
    print('Possibile variazione di trend nel 2019\n')
    m.plot_components(forecast) 
    time.sleep(5.0)

#previsione prezzo stato
PREDICTING_FOR = "TotalUS"
date_price = regions.get_group(PREDICTING_FOR)[['Date', 'AveragePrice']].reset_index(drop=True)
print("------Prezzo medio grafo--------------")
date_price.plot(x='Date', y='AveragePrice', kind="line")

date_price = date_price.rename(columns={'Date':'ds', 'AveragePrice':'y'})
print('Previsione dei prezzi sullo stato  {0:.4f}\n',format(PREDICTING_FOR))
printg(date_price)
time.sleep(5.0)


#---------------Confusion matrix and ROC -------------------
data.drop(["region"],axis=1,inplace=True)
data["type"] = pd.get_dummies(data.type,drop_first=True)
# Y
y = data[["type"]][:]
# X
x = data
x.drop(["type"],axis=1,inplace=True)



# Scale the data to be between -1 and 1
sc = StandardScaler()
X = sc.fit_transform(x)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=SEED)

#SUpport Vector Classifier
svm=SVC(kernel='linear',probability=True)
svm.fit(X_train, y_train.values.ravel())

y_pred_train = svm.predict(X_train)
y_pred_test = svm.predict(X_test)

#Accuracy score
print('Accuracy score for test data using SVM :', accuracy_score(y_test,y_pred_test))

print('**************************************')

np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
titles_options = [
    ("Confusion matrix, without normalization", None),
    ("Normalized confusion matrix", "true"),
]
for title, normalize in titles_options:
    disp = ConfusionMatrixDisplay.from_estimator(
        svm,
        X_test,
        y_test,
        display_labels=['organic','Conventional'],
        cmap=plt.cm.Blues,
        normalize=normalize,
    )
    disp.ax_.set_title(title)

    print(title)
    print(disp.confusion_matrix)

plt.show()
time.sleep(5.0)
print('**************************************')

#AUC ROC Curve

svm=svm.fit(X_train,y_train.values.ravel())
probs=svm.predict_proba(X_test)
preds=probs[:,1]

fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
roc_auc = metrics.auc(fpr, tpr)

plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.3f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()





























































