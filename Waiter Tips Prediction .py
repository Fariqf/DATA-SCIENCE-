#!/usr/bin/env python
# coding: utf-8

# <!DOCTYPE html>
# <html>
# <head>
#     <style>
#         h1 {
#             color: red;
#         }
#     </style>
# </head>
# <body>
#     <h1>Waiter Tips Prediction</h1>
# </body>
# </html>
# 

# In[1]:


import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


# In[3]:


data = pd.read_csv(r"Cwaiters.csv")
print(data.head())


# <h1>Waiter Tips Analysis</h1>

# In[4]:


figure = px.scatter(data_frame = data, x="total_bill",
                    y="tip", size="size", color= "day", trendline="ols")
figure.show()


# In[5]:


figure = px.scatter(data_frame = data, x="total_bill",
                    y="tip", size="size", color= "sex", trendline="ols")
figure.show()


# In[6]:


figure = px.scatter(data_frame = data, x="total_bill",
                    y="tip", size="size", color= "time", trendline="ols")
figure.show()


# In[7]:


figure = px.pie(data, 
             values='tip', 
             names='day',hole = 0.5)
figure.show()


# In[8]:


figure = px.pie(data, 
             values='tip', 
             names='sex',hole = 0.5)
figure.show()


# In[10]:


figure = px.pie(data, 
             values='tip', 
             names='smoker',hole = 0.5)
figure.show()


# In[11]:


figure = px.pie(data, 
             values='tip', 
             names='time',hole = 0.5)
figure.show()


# <h1>Waiter Tips Prediction Model</h1>

# In[12]:


data["sex"] = data["sex"].map({"Female": 0, "Male": 1})
data["smoker"] = data["smoker"].map({"No": 0, "Yes": 1})
data["day"] = data["day"].map({"Thur": 0, "Fri": 1, "Sat": 2, "Sun": 3})
data["time"] = data["time"].map({"Lunch": 0, "Dinner": 1})
data.head()


# In[34]:


x=data.drop("tip",axis=1)
y=data['tip']


# In[44]:


#splitting the data into x_test,x_train,y_test,y_train
from sklearn.model_selection import train_test_split
xtrain,ytrain,xtest,ytest = train_test_split(x, y, 
                                                test_size=0.2, 
                                                random_state=42)


# In[ ]:





# In[50]:


#importing linearregression from scikit learn
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(xtrain,xtest)


# In[51]:


# features = [[total_bill, "sex", "smoker", "day", "time", "size"]]
features = np.array([[24.50, 1, 0, 0, 1, 4]])
model.predict(features)


# In[52]:


y_pred=model.predict(ytrain)


# In[57]:


#score of the model
from sklearn.metrics import mean_squared_error

mse = mean_squared_error(ytest, y_pred)
print("Mean Squared Error:", mse*100,"%")


# In[ ]:





# In[ ]:




