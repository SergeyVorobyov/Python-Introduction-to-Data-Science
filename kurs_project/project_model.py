
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor


# #### предобработка данных

# In[3]:


def my_preproc(frame):
    # число комнат
    # если число комнат = 0, делаем равным 1
    # если число комнат > 6, делаем равным 6
    frame.loc[ (frame[frame['Rooms'] == 0]).index, 'Rooms'] = 1
    frame.loc[ (frame[frame['Rooms'] > 6]).index, 'Rooms'] = 6
    
    # выбросы жилой площади
    # если жилая площадь больше 200, то приравниваем к общей - 1
    frame.loc[(frame[frame['LifeSquare'] > 200]).index, 'LifeSquare'] =     frame.loc[(frame[frame['LifeSquare'] > 200]).index, 'Square'] - 1    
    
    # общая площадь 
    # если площадь меньше жилой площади, приравниваем к жилой + 1
    frame.loc[(frame[frame['Square'] < frame['LifeSquare']]).index, 'Square'] =     frame.loc[(frame[frame['Square'] < frame['LifeSquare']]).index, 'LifeSquare'] + 1    
    
    # площадь кухни
    # если площадь кухни больше общей площади, приравниваем к общей - 1 
    frame.loc[(frame[frame['KitchenSquare'] > frame['Square']]).index, 'KitchenSquare'] =     frame.loc[(frame[frame['KitchenSquare'] > frame['Square']]).index, 'Square'] - 1
    
    # жилая площадь с NaN
    # считаем среднюю жилую площадь на 1 комнату
    # заполняем NaN исходя из числа комнат
    mean_lifesqr_per_room = (frame['LifeSquare']/frame['Rooms']).median()
    # если полученная жилая площадь больше общей, то берем общую - 1
    def null_lifesqr_preproc(row):
        sqr =  row['Rooms']* mean_lifesqr_per_room
        if sqr > row['Square'] :
            return row['Square'] - 1
        else:
            return sqr
    frame.loc[frame[frame['LifeSquare'].isnull()].index, 'LifeSquare'] =     frame.loc[frame[frame['LifeSquare'].isnull()].index,:].apply(null_lifesqr_preproc, axis = 1)
    
    # год потройки
    # если год больше 2020, то присваиваем 2020
    frame.loc[ (frame[frame['HouseYear'] > 2020]).index, 'HouseYear'] = 2020
    
    # Healthcare_1 слишком много незаполненных значений
    frame = frame.drop('Healthcare_1', axis = 1)
    
    frame['DistrictId'] = frame['DistrictId'].astype(str)
    frame['Shops_1'] = frame['Shops_1'].astype(str)
    frame['Helthcare_2'] = frame['Helthcare_2'].astype(str)
    frame['Social_3'] = frame['Social_3'].astype(str)
    
    # разные значения в train и test в признаке DistrictId
    frame = frame.drop('DistrictId', axis = 1)
    
    # часть полей содержат категориальные признаки (по типу данных или по смыслу)
    cols_for_dummy = list()
    #cols_for_dummy.append('DistrictId')
    # разные значения в train и test в признаке DistrictId
    cols_for_dummy.append('Ecology_2')
    cols_for_dummy.append('Ecology_3')
    cols_for_dummy.append('Social_3')
    cols_for_dummy.append('Helthcare_2')
    cols_for_dummy.append('Shops_1')
    cols_for_dummy.append('Shops_2')
    
    # делаем для отобранных признаков one hot encoding
    df_cat = frame[cols_for_dummy]
    df_cat = pd.get_dummies(df_cat, drop_first = True)
    frame = frame.drop(cols_for_dummy, axis = 1)
    frame = pd.concat([frame, df_cat], axis = 1)
    
    return frame


# #### загружаем обучающую выборку

# In[4]:


data = pd.read_csv('train.csv')
df = data.copy()


# In[5]:


df = my_preproc(df)


# In[6]:


X = df.set_index('Id', inplace = True)
X = df.drop('Price', axis = 1)
y = df['Price']


# In[7]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)


# #### случайный лес через подбор параметров

# In[8]:


pipe = Pipeline([('scaler', StandardScaler()), ('rf',RandomForestRegressor())])


# In[9]:


paramgrid = {'rf__n_estimators':[200, 250, 300, 350,  400, 450, 500, 550, 600],
            'rf__max_depth': range(1, 31)
            }


# In[ ]:


cv = GridSearchCV(pipe, param_grid= paramgrid, scoring = 'r2', cv = 5)


# In[ ]:


cv.fit(X_train, y_train)


# In[ ]:


# подобранные параметры
cv.best_params_


# In[ ]:


# R2 на train
cv.best_score_


# In[ ]:


#rmse на train
mean_squared_error(y_train, cv.predict(X_train))**0.5


# In[ ]:


# R2 на test
cv.score(X_test, y_test)


# In[ ]:


#rmse на train
mean_squared_error(y_test, cv.predict(X_test))**0.5


# #### прогноз на тестовой выборке

# In[ ]:


df_test = pd.read_csv('test.csv')


# In[ ]:


df_test = my_preproc(df_test)


# In[ ]:


test_id = df_test['Id']


# In[ ]:


test_id.shape


# In[ ]:


df_test = df_test.drop('Id', axis = 1)


# In[ ]:


X_train.shape


# In[ ]:


X_test.shape


# In[ ]:


df_test.shape


# In[ ]:


set1 = set(X_train.columns)


# In[ ]:


set2 = set(df_test.columns)


# In[ ]:


set2 - set1


# In[ ]:


set1 - set2


# In[ ]:


y_pred = cv.predict(df_test)


# In[ ]:


y_pred.shape


# In[ ]:


result = pd.DataFrame(data = y_pred, index = test_id.values, columns= ['Price'])


# In[ ]:


result.reset_index(inplace= True)


# In[ ]:


result.columns = ['Id','Price']


# In[ ]:


result.to_csv('SVorobyov_predictions1.csv', index = False)


# In[ ]:




