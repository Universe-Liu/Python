import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from lightgbm.sklearn import LGBMClassifier
from sklearn import metrics

df=pd.read_csv('high_diamond_ranked_10min.csv')
#print(df.info())
drop_cols=['gameId','blueWins']
x=df.drop(drop_cols,axis=1)
y=df.blueWins
#print(x.describe())
drop_cols = ['redFirstBlood','redKills','redDeaths','redGoldDiff','redExperienceDiff', 'blueCSPerMin','blueGoldPerMin','redCSPerMin','redGoldPerMin']
x.drop(drop_cols, axis=1, inplace=True)
'''
data=x
data_std=(data-data.mean())/data.std()
data=pd.concat([y,data_std.iloc[:,0:9]],axis=1)
data=pd.melt(data,id_vars='blueWins',var_name='Features',value_name='Values')
fig,ax=plt.subplots(1,2,figsize=(15,5))
sns.violinplot(x='Features',y='Values',hue='blueWins',data=data,split=True,inner='quart',ax=ax[0],palette='Blues')
fig.autofmt_xdate(rotation=45)

data=x
data_std=(data-data.mean())/data.std()
data=pd.concat([y,data_std.iloc[:,9:18]],axis=1)
data=pd.melt(data,id_vars='blueWins',var_name='Features',value_name='Values')
sns.violinplot(x='Features',y='Values',hue='blueWins',data=data,split=True,inner='quart',ax=ax[1],palette='Blues')
fig.autofmt_xdate(rotation=45)
plt.show()

plt.figure(figsize=(18,14))
sns.heatmap(round(x.corr(),2),cmap='Blues',annot=True)
plt.show()

drop_cols=['redAvgLevel','blueAvgLevel']
x.drop(drop_cols,axis=1,inplace=True)
sns.set(style='whitegrid', palette='muted')

data = x[['blueWardsPlaced','blueWardsDestroyed','wardsPlacedDiff','wardsDestroyedDiff']].sample(1000)
data_std = (data - data.mean()) / data.std()
data = pd.concat([y, data_std], axis=1)
data = pd.melt(data, id_vars='blueWins', var_name='Features', value_name='Values')
'''
#plt.figure(figsize=(10,6))
#sns.swarmplot(x='Features', y='Values', hue='blueWins', data=data)
#plt.xticks(rotation=45)
#plt.show()
x['wardsPlacedDiff'] = x['blueWardsPlaced'] - x['redWardsPlaced']
x['wardsDestroyedDiff'] = x['blueWardsDestroyed'] - x['redWardsDestroyed']
drop_cols=['blueWardsPlaced','blueWardsDestroyed','wardsPlacedDiff','wardsDestroyedDiff','redWardsPlaced','redWardsDestroyed']
x.drop(drop_cols,axis=1,inplace=True)
x['killsDiff'] = x['blueKills'] - x['blueDeaths']
x['assistsDiff'] = x['blueAssists'] - x['redAssists']
#x[['blueKills','blueDeaths','blueAssists','killsDiff','assistsDiff','redAssists']].hist(figsize=(12,10), bins=20)
#plt.show()
x['dragonsDiff'] = x['blueDragons'] - x['redDragons']
x['heraldsDiff'] = x['blueHeralds'] - x['redHeralds']
x['eliteDiff'] = x['blueEliteMonsters'] - x['redEliteMonsters']
data = pd.concat([y, x], axis=1)
eliteGroup = data.groupby(['eliteDiff'])['blueWins'].mean()
dragonGroup = data.groupby(['dragonsDiff'])['blueWins'].mean()
heraldGroup = data.groupby(['heraldsDiff'])['blueWins'].mean()
'''
fig, ax = plt.subplots(1,3, figsize=(15,4))
eliteGroup.plot(kind='bar', ax=ax[0])
dragonGroup.plot(kind='bar', ax=ax[1])
heraldGroup.plot(kind='bar', ax=ax[2])
print(eliteGroup)
print(dragonGroup)
print(heraldGroup)
plt.show()
'''
x['towerDiff'] = x['blueTowersDestroyed'] - x['redTowersDestroyed']
data = pd.concat([y, x], axis=1)

towerGroup = data.groupby(['towerDiff'])['blueWins']
print(towerGroup.count())
print(towerGroup.mean())
'''
fig, ax = plt.subplots(1,2,figsize=(15,5))

towerGroup.mean().plot(kind='line', ax=ax[0])
ax[0].set_title('Proportion of Blue Wins')
ax[0].set_ylabel('Proportion')

towerGroup.count().plot(kind='line', ax=ax[1])
ax[1].set_title('Count of Towers Destroyed')
ax[1].set_ylabel('Count')
'''
data_target_part=y
data_features_part=x
x_train,x_test,y_train,y_test=train_test_split(data_features_part,data_target_part,test_size=0.2,random_state=2020)
clf=LGBMClassifier()
clf.fit(x_train,y_train)
train_predict=clf.predict(x_train)
test_predict=clf.predict(x_test)
## 利用accuracy（准确度）【预测正确的样本数目占总预测样本数目的比例】评估模型效果
print('The accuracy of the Logistic Regression is:',metrics.accuracy_score(y_train,train_predict))
print('The accuracy of the Logistic Regression is:',metrics.accuracy_score(y_test,test_predict))

## 查看混淆矩阵 (预测值和真实值的各类情况统计矩阵)
confusion_matrix_result = metrics.confusion_matrix(test_predict,y_test)
print('The confusion matrix result:\n',confusion_matrix_result)
'''
# 利用热力图对于结果进行可视化
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix_result, annot=True, cmap='Blues')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.show()
'''
sns.barplot(y=data_features_part.columns,x=clf.feature_importances_)
from sklearn.metrics import accuracy_score
from lightgbm import plot_importance

def estimate(model,data):

    #sns.barplot(data.columns,model.feature_importances_)
    ax1=plot_importance(model,importance_type="gain")
    ax1.set_title('gain')
    ax2=plot_importance(model, importance_type="split")
    ax2.set_title('split')
    plt.show()
def classes(data,label,test):
    model=LGBMClassifier()
    model.fit(data,label)
    ans=model.predict(test)
    estimate(model, data)
    return ans
 
ans=classes(x_train,y_train,x_test)
pre=accuracy_score(y_test, ans)
print('acc=',accuracy_score(y_test,ans))
from sklearn.model_selection import GridSearchCV

## 定义参数取值范围
learning_rate = [0.1, 0.3, 0.6]
feature_fraction = [0.5, 0.8, 1]
num_leaves = [16, 32, 64]
max_depth = [-1,3,5,8]

parameters = { 'learning_rate': learning_rate,
              'feature_fraction':feature_fraction,
              'num_leaves': num_leaves,
              'max_depth': max_depth}
model = LGBMClassifier(n_estimators = 50)

## 进行网格搜索
#clf = GridSearchCV(model, parameters, cv=3, scoring='accuracy',verbose=3, n_jobs=-1)
#clf = clf.fit(x_train, y_train)
#print("best:   ",clf.best_params_)
clf = LGBMClassifier(feature_fraction = 0.8,
                    learning_rate = 0.1,
                    max_depth= 3,
                    num_leaves = 16)
# 在训练集上训练LightGBM模型
clf.fit(x_train, y_train)

train_predict = clf.predict(x_train)
test_predict = clf.predict(x_test)

## 利用accuracy（准确度）【预测正确的样本数目占总预测样本数目的比例】评估模型效果
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
plt.show()