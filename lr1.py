import time
import timeit

from sklearn.datasets import load_boston
import pandas as pd
import sklearn.datasets as datasets
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import DistanceMetric





#
pd.set_option('display.max_columns', None)
# #11111111111111111111111111111111111111
from scipy import stats
# # #1.1
# V = load_boston(return_X_y=True)
# boston = load_boston()
# x = V[0]
# y = V[1]
# boston_df = pd.read_csv(r"C:\Users\123\PycharmProjects\pythonProject3\boston.csv")
# print(boston_df.head())
# print('Crime - уровень преступности,ZN -доля жилой земли,25sq,Indus-доля бизнесов не предназначенных для продажи,Chas -переменная чарльса,NOX-коэфицент азота,RM-среднее количество комнта в помещении,'
#       '\nAge -доля квартир занятая владельцами с 1940,DIS-взвешенные расстояния до пяти бостонских центров занятости,Rad-индекс доступности к радиальным магистралям,'
#       '\nTax-ставка налога на недвижимость с полной стоимостью за 10 000 долларов США,PIRATTO-соотношение учащихся и учителей в разбивке по городам,B-1000(Bk - 0,63)^2, где Bk - доля чернокожих в разбивке по городам,LSTAT-более низкий статус населения,MEDV-Средняя стоимость домов, занятых владельцами, в 1000 долларов')
# print(boston_df.isnull().sum())
# print(x.shape, y.shape)
# print(boston.keys())
# print(boston.DESCR)
# #1.2
# pd.DataFrame(x).head()
# pd.DataFrame(x).info()
# #1.3
# plt.hist(y)
# plt.grid()
# plt.title('y histogram')
# plt.show()
# #1.4
# plt.hist(x[:,1])
# plt.grid()
# plt.title('x histogram')
# plt.show()
# plt.hist(x[:,2])
# plt.grid()
# plt.title('x histogram')
# plt.show()
# plt.hist(x[:,3])
# plt.grid()
# plt.title('x histogram')
# plt.show()
# plt.hist(x[:,4])
# plt.grid()
# plt.title('x histogram')
# plt.show()
# plt.hist(x[:,5])
# plt.grid()
# plt.title('x histogram')
# plt.show()
# plt.hist(x[:,6])
# plt.grid()
# plt.title('x histogram')
# plt.show()
# plt.hist(x[:,7])
# plt.grid()
# plt.title('x histogram')
# plt.show()
# plt.hist(x[:,8])
# plt.grid()
# plt.title('x histogram')
# plt.show()
# plt.hist(x[:,9])
# plt.grid()
# plt.title('x histogram')
# plt.show()
# plt.hist(x[:,10])
# plt.grid()
# plt.title('x histogram')
# plt.show()
# plt.hist(x[:,11])
# plt.grid()
# plt.title('x histogram')
# plt.show()
# plt.hist(x[:,12])
# plt.grid()
# plt.title('x histogram')
# plt.show()
#
# # 1.6 Грёбанный модуль , искал наугад полгода
# stat, p = stats.shapiro(x[:,1])
# print(p ,'оценка нормальности 1')
# stat, p = stats.shapiro(x[:,2])
# print(p ,'оценка нормальности 2')
# stat, p = stats.shapiro(x[:,3])
# print(p ,'оценка нормальности 3')
# stat, p = stats.shapiro(x[:,4])
# print(p ,'оценка нормальности 4')
# stat, p = stats.shapiro(x[:,5])
# print(p ,'оценка нормальности 5')
# stat, p = stats.shapiro(x[:,6])
# print(p ,'оценка нормальности 6')
# stat, p = stats.shapiro(x[:,7])
# print(p ,'оценка нормальности 7')
# stat, p = stats.shapiro(x[:,8])
# print(p ,'оценка нормальности 8')
# stat, p = stats.shapiro(x[:,9])
# print(p ,'оценка нормальности 9')
# stat, p = stats.shapiro(x[:,10])
# print(p ,'оценка нормальности 10')
# stat, p = stats.shapiro(x[:,11])
# print(p ,'оценка нормальности 11')
# stat, p = stats.shapiro(x[:,12])
# print(p ,'оценка нормальности 12')
#
# # 1.7.
# print(pd.DataFrame(y).describe())
# print(pd.DataFrame(x).describe())
#1.8. Набор интересный




# # #222222222222222222222222222222222222222222222222
# # # В связи с документацией https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_blobs.html , число информативных признаков выставленно 3
# X, y = datasets.make_classification(n_samples = 1000, n_redundant=3, n_classes=3,
#                                     n_informative=3, n_clusters_per_class=2)

#получение информации
# plt.scatter(X[:,0],X[:,1],c = y)
# plt.grid()
# plt.xlabel('x0')
# plt.ylabel('x1')
# plt.show()
# stat, p = stats.shapiro(y) #оценка нормальности
# print(p)
# pd.DataFrame(X).head()
# pd.DataFrame(X).info()
# plt.hist(X[1,:])
# plt.grid()
# plt.title('x1 histogram')
# plt.show()
# plt.hist(X[2,:])
# plt.grid()
# plt.title('x2 histogram')
# plt.show()
# plt.hist(X[:,1])
# plt.grid()
# plt.title('x3 histogram')
# plt.show()
# plt.hist(X[:,2])
# plt.grid()
# plt.title('x4 histogram')
# plt.show()
# plt.hist(X[:,3])
# plt.grid()
# plt.title('x5 histogram')
# plt.show()

#3333333333333333333333333


V3 = datasets.load_breast_cancer()
X,y = datasets.make_classification(n_samples = 1000, n_features=10, n_redundant=2, n_informative=2,random_state=10, n_clusters_per_class=2)
X1,y1 = datasets.make_blobs(n_samples=1000, centers=2, n_features=5, random_state=10)
print( V3.keys())
df = pd.DataFrame(V3.data, columns = V3.feature_names)

# Add the target columns, and fill it with the target data
df["target"] = V3.target

# Show the dataframe
print(df)


plt.scatter(X1[:,0],X1[:,1],c = y)
plt.grid()
plt.xlabel('x0')
plt.ylabel('x1')
plt.show()
stat, p1 = stats.shapiro(y1) #оценка нормальности
print(p1)
pd.DataFrame(X1).head()
pd.DataFrame(X1).info()
print(pd.DataFrame(y1).describe())
print(pd.DataFrame(X1).describe())

plt.scatter(X[:,0],X[:,1],c = y)
plt.grid()
plt.xlabel('x0')
plt.ylabel('x1')
plt.show()
stat, p = stats.shapiro(y) #оценка нормальности
print(p)
pd.DataFrame(X).head()
pd.DataFrame(X).info()
print(pd.DataFrame(y).describe())
print(pd.DataFrame(X).describe())
plt.hist(X[:, 1])
plt.grid()
plt.title('y histogram')
plt.show()
#На основе признака размера опухоли рака мы можем наблюдать на гистограмме нормальное расспределение
#с вероятностью изменений в средний часте размер опухолей от 3 до 250 мм количество столбцов 10

print(df['target'].sum())