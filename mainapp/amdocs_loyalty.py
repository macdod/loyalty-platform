import pandas as pd
import numpy as np
import random
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
import plotly.offline as po
import plotly.graph_objs as go

def upload_():
  # files.upload()
  cust_original=pd.read_csv("mainapp/static/mainapp/data/df1.csv")
  cust_original=cust_original.drop(columns=['Unnamed: 0'])
  # files.upload()
  incentive=pd.read_csv("mainapp/static/mainapp/data/R.csv")
  incentive=incentive.drop(columns=['Unnamed: 0'])
  return cust_original,incentive

def Cust_data(cust_original):
  X=cust_original[['age','gender','profile','tenure','TV','movies','Vouchers','calling','Total_charges','Billing']]
  Y=cust_original['Loyalty']
  return X,Y

def train_Loyalty(X,Y):
  reg = MLPRegressor(hidden_layer_sizes=(5,),
  activation='relu',
  solver='adam',
  learning_rate='adaptive',
  max_iter=1000,
  learning_rate_init=0.01,
  alpha=0.01).fit(X, Y)
  return reg

def Loyalty(X,reg):
  return reg.predict(X).sum()

def Incentive_data(incentive):
  X1=incentive[['TV','movies','Vouchers','calling','Total_charges','Billing','shopping_voucher','food_voucher','movie_voucher','travel_voucher','billing_voucher','extra_data']]
  Y1=incentive['TV_']
  Y2=incentive['movies_']
  Y3=incentive['Vouchers_']
  Y4=incentive['calling_']
  Y5=incentive['Total_charges_']
  Y6=incentive['Billing_']
  return X1,Y1,Y2,Y3,Y4,Y5,Y6

def Incentive_train(X1,Y1,Y2,Y3,Y4,Y5,Y6):
  reg1 = MLPRegressor(hidden_layer_sizes=(5,),
  activation='relu',
  solver='adam',
  learning_rate='adaptive',
  max_iter=1000,
  learning_rate_init=0.01,
  alpha=0.01).fit(X1, Y1)
  reg2 = MLPRegressor(hidden_layer_sizes=(5,),
  activation='relu',
  solver='adam',
  learning_rate='adaptive',
  max_iter=1000,
  learning_rate_init=0.01,
  alpha=0.01).fit(X1, Y2)
  reg3 = MLPRegressor(hidden_layer_sizes=(5,),
  activation='relu',
  solver='adam',
  learning_rate='adaptive',
  max_iter=1000,
  learning_rate_init=0.01,
  alpha=0.01).fit(X1, Y3)
  reg4 = MLPRegressor(hidden_layer_sizes=(5,),
  activation='relu',
  solver='adam',
  learning_rate='adaptive',
  max_iter=1000,
  learning_rate_init=0.01,
  alpha=0.01).fit(X1, Y4)
  reg5 = MLPRegressor(hidden_layer_sizes=(5,),
  activation='relu',
  solver='adam',
  learning_rate='adaptive',
  max_iter=1000,
  learning_rate_init=0.01,
  alpha=0.01).fit(X1, Y5)
  reg6 = MLPRegressor(hidden_layer_sizes=(5,),
  activation='relu',
  solver='adam',
  learning_rate='adaptive',
  max_iter=1000,
  learning_rate_init=0.01,
  alpha=0.01).fit(X1, Y6)
  return reg1,reg2,reg3,reg4,reg5,reg6

def after_incentive(customer,reg1,reg2,reg3,reg4,reg5,reg6,a):
  cust=customer[['TV','movies','Vouchers','calling','Total_charges','Billing']]
  cust['shopping_voucher']=a[0]
  cust['food_voucher']=a[1]
  cust['movie_voucher']=a[2]
  cust['travel_voucher']=a[3]
  cust['billing_voucher']=a[4]
  cust['extra_data']=a[5]
  y1=reg1.predict(np.array(cust))
  y2=reg2.predict(np.array(cust))
  y3=reg3.predict(np.array(cust))
  y4=reg4.predict(np.array(cust))
  y5=reg5.predict(np.array(cust))
  y6=reg6.predict(np.array(cust))
  Cust_final=customer[['age','gender','profile','tenure']]
  Cust_final['TV']=y1
  Cust_final['movies']=y2
  Cust_final['Vouchers']=y3
  Cust_final['calling']=y4
  Cust_final['Total_charges']=y5
  Cust_final['Billing']=y6
  return Cust_final

def ideal_customer_in_population(population,reg):
  
  sample = pd.DataFrame(columns=['age','gender','profile','tenure','TV','movies','Vouchers','calling','Total_charges','Billing','Loyalty'])
  for i in range(0,100):
    sample.loc[i]=population.loc[i]
  sample.sort_values(["Loyalty"], axis=0, 
    ascending=False, inplace=True)
  sample.reset_index(drop=True, inplace=True)
  result = pd.DataFrame(columns=['age','gender','profile','tenure','TV','movies','Vouchers','calling','Total_charges','Billing','Loyalty'])
  z=0
  v=0
  print("valala")
  while v<=5:
    for i in range(0,100,2):
      user1=np.array(sample.loc[i])
      user2=np.array(sample.loc[i+1])
      user1[3],user2[3]= user2[3],user1[3]
      user1[5],user2[5]= user2[5],user1[5]
      user1[7],user2[7]= user2[7],user1[7]
      result.loc[z]=[user1[0],user1[1],user1[2],user1[3],user1[4],user1[5],user1[6],user1[7],user1[8],user1[9],user1[10]]
      z=z+1
      result.loc[z]=[user2[0],user2[1],user2[2],user2[3],user2[4],user2[5],user2[6],user2[7],user2[8],user2[9],user2[10]]
      z=z+1

    result1 = pd.DataFrame(columns=['age','gender','profile','tenure','TV','movies','Vouchers','calling','Total_charges','Billing','Loyalty'])
    m=0
    for i in range(0,100):
      u=np.array(result.loc[i])
      k=random.choice([3,4,5,6,7,8,9])
      if k==3:
        u[k]=random.randrange(6,24)
      elif k==4:
        u[k]=random.randrange(0,48)
      elif k==5:
        u[k]=random.randrange(0,48)
      elif k==6:
        u[k]=random.randrange(0,100)
      elif k==7:
        u[k]=random.randrange(0,48)
      elif k==8:
        u[k]=random.randrange(2000,7000)
      elif k==9:
        u[k]=random.randrange(0,1)
      result1.loc[m]=[u[0],u[1],u[2],u[3],u[4],u[5],u[6],u[7],u[8],u[9],u[10]]
      m=m+1
  
    X=result1[['age','gender','profile','tenure','TV','movies','Vouchers','calling','Total_charges','Billing']]
    Y=result1['Loyalty']
    df1=pd.DataFrame(reg.predict(np.array(X)))  
    for i in range(0,100):
      m=df1.loc[i]
      n=result1.loc[i]
      n[10]=m[0]
      result1.loc[i]=n
    result1=result1[['age','gender','profile','tenure','TV','movies','Vouchers','calling','Total_charges','Billing','Loyalty']]

    result1.sort_values(["Loyalty"], axis=0, 
    ascending=False, inplace=True)
    result1.reset_index(drop=True, inplace=True)
    r1 = pd.DataFrame(columns=['age','gender','profile','tenure','TV','movies','Vouchers','calling','Total_charges','Billing','Loyalty'])
    c=1
    i=1
    k=0
    x=sample['Loyalty'].max()
    u=result1.loc[0]
    r1.loc[0]=u
    for j in range(1,99):
      u=result1.loc[j]
      if u[10]>=x  and c<=100:
        r1.loc[i]=result1.loc[j]
        i=i+1
        c=c+1   
    if c<100:
      while c<100:
        r1.loc[i]=sample.loc[k]
        k=k+1
        i=i+1
        c=c+1
  
    sample=r1
    v=v+1
  r1.sort_values(["Loyalty"], axis=0, 
  ascending=False, inplace=True)
  r1.reset_index(drop=True, inplace=True)
  return r1.loc[0];

cust_original,incentive=upload_()
X,Y = Cust_data(cust_original)
reg=train_Loyalty(X,Y)
X1,Y1,Y2,Y3,Y4,Y5,Y6 = Incentive_data(incentive)
reg1,reg2,reg3,reg4,reg5,reg6=Incentive_train(X1,Y1,Y2,Y3,Y4,Y5,Y6)

def f(s):
  display(text)
  text.on_submit(ideal)
  
def plot_graph(ideal,avg,population):
  #po.init_notebook_mode(connected=True)
  a=np.array(avg)
  a[8]=np.log(avg[8])
  b=np.array(ideal)
  b[8]=np.log(ideal[8])
  xlist=list(population.columns)
  trace_avg=go.Bar(
      x=xlist,
      y=a,
      name='Average customer',
      marker=dict(
          color='rgb(240,155,55)',
          line=dict(
              color='rgb(240,155,55)',
              width=5,
          ),
          opacity=0.5
      )
  )
  trace_ideal=go.Bar(
      x=xlist,
      y=b,
      name='Ideal customer',
      marker=dict(
          color='rgb(55,155,240)',
          line=dict(
              color='rgb(55,155,240)',
              width=5,
          ),
          opacity=0.7
      )
  )
  data=[trace_avg,trace_ideal]
  layout=dict(
      xaxis=dict(
          visible=True,
          rangeslider=dict(
              visible=True,
            thickness=0.09
              ),
          tickcolor="#fff",
          tickfont=dict(color="#fff")
          ),
      yaxis=dict(
          visible=True,
          tickfont=dict(color="#fff")
          ),
      plot_bgcolor='rgb(0, 0, 0,0.1)',
      paper_bgcolor='rgb(0, 0, 0,0.1)'
  )
  fig=dict(data=data,layout=layout)
  po.plot(fig,filename='mainapp/templates/mainapp/plots/plot1.html',auto_open=False)


def Bar_graph(x,y) :
  trace=dict(
    type='bar',
      x=list(x),
      y=list(y),
      marker=dict(
          color='rgb(25,255,25)',
          line=dict(
              color='rgb(25,255,25)',
              width=5,
          ),
          opacity=0.5
      )
  )
  data=[trace]
  layout=dict(
      xaxis=dict(
          visible=True,
          rangeslider=dict(
              visible=True,
            thickness=0.09
              ),
          tickcolor="#fff",
          tickfont=dict(color="#fff")
          ),
      yaxis=dict(
          visible=True,
          tickfont=dict(color="#fff")
          ),
      plot_bgcolor='rgb(0, 0, 0,0.1)',
      paper_bgcolor='rgb(0, 0, 0,0.1)'
  )
  fig=dict(data=data,layout=layout)
  po.plot(fig,auto_open=False,filename='mainapp/templates/mainapp/plots/plot2.html')
  return 'plot2.html'


def ideal(s):
  print (s)
  a=int(s)
  population=cust_original.loc[cust_original['profile'] == a]
  population.reset_index(drop=True, inplace=True)
  ideal=ideal_customer_in_population(population,reg)
  avg=population.mean(axis = 0) 
  bins = np.linspace(-10, 10, 100)
  #pyplot.hist(ideal, bins, alpha=0.5, label='ideal customer')
  #pyplot.hist(avg, bins, alpha=0.5, label='Average customer')
  #pyplot.legend(loc='upper right')
  #pyplot.show()
  plot_graph(ideal,avg,population)

def f1(s):
  display(text)
  text.on_submit(LOYALTY) 
  
def LOYALTY(s):
  Incent=s.split(" ")
  a=int(Incent[0])
  for i in range (1,7):
    Incent[i-1]=int(Incent[i])
  population_before=cust_original[['age','gender','profile','tenure','TV','movies','Vouchers','calling','Total_charges','Billing']].loc[cust_original['profile'] == a]
  population_after_incentive=after_incentive(population_before,reg1,reg2,reg3,reg4,reg5,reg6,Incent)
  a=Loyalty(population_before,reg)/len(population_before)
  b=Loyalty(population_after_incentive,reg)/len(population_before)
  Bar_graph(['a','b'],[a,b])
  x = np.arange(2)
  money = [a,b]