# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error


# %%
month_data=pd.read_csv(r"F:\シュウイチ\program\hackthon\train_data (1).csv")

X=month_data
train_data=pd.read_csv(r"F:\シュウイチ\program\hackthon\test_data.csv")
Y=train_data
data=pd.read_csv(r"F:\シュウイチ\program\hackthon\pcr_case_daily.csv")
data = data.fillna(0)
X_train,X_test,Y_train,Y_test =train_test_split(X,Y,test_size=0.2,shuffle=False)
# %%
clf =MLPRegressor(hidden_layer_sizes=(2,100), 
                       solver='adam', alpha=0.0001,learning_rate='constant', 
                      learning_rate_init=0.001)

clf.fit(X_train,Y_train)
clf.score(X_test,Y_test)
y_pred=clf.predict(X_test)

# %%
print('平均絶対誤差:',mean_absolute_error(Y_test,y_pred))
print('平均平方二乗誤差:',np.sqrt(mean_squared_error(Y_test,y_pred)))
# %%
def display_plot(pred):
    plt.plot(data["日付"],Y["death_total"],color="orange",label="test")
    plt.plot(data["日付"].iloc[-len(pred):],pred,color="blue",label="predict")
    plt.legend(loc="upper left")
    plt.xlabel("date")
    plt.ylabel("death")
    ax=plt.gca()
    ax.axes.xaxis.set_ticks([])
    ax.axes.yaxis.set_ticks([])
    plt.grid(True)
    plt.show()
# %%
display_plot(y_pred)
# %%
#print('current loss computed with the loss function: ',clf.loss_)
#print('coefs: ', clf.coefs_)
#print('intercepts: ',clf.intercepts_)
#print(' number of iterations the solver: ', clf.n_iter_)
#print('num of layers: ', clf.n_layers_)
#print('Num of o/p: ', clf.n_outputs_)
