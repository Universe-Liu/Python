import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from xgboost.sklearn import XGBClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score
from xgboost import plot_importance
from sklearn.model_selection import GridSearchCV

data=pd.read_csv('train.csv')
data=data.fillna(-1)
print(pd.Series(data['RainTomorrow']).value_counts())

numrical_features=[x for x in data.columns if data[x].dtype == np.float]
category_features=[x for x in data.columns if data[x].dtype != np.float and x!='RainTomorrow']
'''
sns.pairplot(data=data[['Rainfall','Evaporation','Sunshine']+['RainTomorrow']],diag_kind='hist',hue='RainTomorrow')
plt.show()
for col in data[numrical_features].columns:
    if col!='RainTomorrow':
        sns.boxplot(x='RainTomorrow',y=col,saturation=0.5,palette='pastel',data=data)
        plt.title(col)
        plt.show()
'''

def get_mapfunction(x):
    mapp = dict(zip(x.unique().tolist(),
         range(len(x.unique().tolist()))))
    def mapfunction(y):
        if y in mapp:
            return mapp[y]
        else:
            return -1
    return mapfunction
for i in category_features:
    data[i] = data[i].apply(get_mapfunction(data[i]))

data_target_part=data['RainTomorrow']
data_features_part=data[[x for x in data.columns if x!='RainTomorrow']]
x_train,x_test,y_train,y_test=train_test_split(data_features_part,data_target_part,test_size=0.2,random_state=2020)
clf = XGBClassifier()
clf.fit(x_train,y_train)

def predictdata_analysis(x_train, x_test, y_train, y_test,clf):
    train_predict=clf.predict(x_train)
    test_predict=clf.predict(x_test)
    #print("w & b:", clf.coef_, clf.intercept_)
    print('The accuracy of the Logistic Regression is:',metrics.accuracy_score(y_train,train_predict))
    print('The accuracy of the Logistic Regression is:',metrics.accuracy_score(y_test,test_predict))

    ## 查看混淆矩阵 (预测值和真实值的各类情况统计矩阵)
    confusion_matrix_result = metrics.confusion_matrix(test_predict,y_test)
    print('The confusion matrix result:\n',confusion_matrix_result)

    # 利用热力图对于结果进行可视化
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix_result, annot=True, cmap='Blues')
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    #plt.show()
    
#predictdata_analysis(x_train,x_test,y_train,y_test,clf)
#sns.barplot(y=data_features_part.columns,x=clf.feature_importances_)
#plt.show()

def estimate(model,data):
    ax1=plot_importance(model,importance_type="gain")
    ax1.set_title('gain')
    ax2=plot_importance(model, importance_type="weight")
    ax2.set_title('weight')
    ax3 = plot_importance(model, importance_type="cover")
    ax3.set_title('cover')
    plt.show()

def classes(data,label,test):
    model=XGBClassifier()
    model.fit(data,label)
    ans=model.predict(test)
    #estimate(model, data)
    return ans

ans=classes(x_train,y_train,x_test)
pre=accuracy_score(y_test, ans)
print('acc=',accuracy_score(y_test,ans))
learning_rate=[0.1,0.3,0.6]
subsample=[0.8,0.9]
colsample_bytree=[0.6,0.8]
max_depth=[3,5,8]
parameters = { 'learning_rate': learning_rate,
              'subsample': subsample,
              'colsample_bytree':colsample_bytree,
              'max_depth': max_depth}
model = XGBClassifier(n_estimators = 50)
clf = GridSearchCV(model, parameters, cv=3, scoring='accuracy',verbose=1,n_jobs=-1)
clf = clf.fit(x_train, y_train)
print(clf.best_params_)