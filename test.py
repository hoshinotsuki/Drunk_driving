from __future__ import print_function

from IPython import display
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 30
pd.options.display.float_format = '{:.1f}'.format

dataframe = pd.read_csv("D:\datas\dd.csv", sep=",")
display.display(dataframe.describe())
display.display(dataframe)

plt.figure()
plt.title("Alcohol concentration in the blood")
plt.ylabel("Alcohol concentration")
plt.xlabel("Time")
plt.scatter(dataframe["time"],dataframe["content"])

#使用非线性最小二乘法拟合
from scipy.optimize import curve_fit

def func(x,a,b,c):
    return a*(np.exp(-b*x)-np.exp(-c *x))
#用指数形式来拟合
x = np.array(dataframe["time"])
y = np.array(dataframe["content"])
popt, pcov = curve_fit(func, x,y)
a=popt[0]
b=popt[1]
c=popt[2]
yvals=func(x,a,b,c)
plot2=plt.plot(x, yvals, 'r',label='2 bottles of beer / short time')
print (a,b,c)

def exponent_residuals(a,b,c, y, x):
    '''計算指數模型的誤差并返回。'''
    return y - a*(np.exp(-b*x)-np.exp(-c *x))
exp_err_ufunc = np.frompyfunc(lambda y, x: exponent_residuals(a,b,c, y, x), 2, 1)
exp_errors = exp_err_ufunc(y, x)
exp_rms = np.sqrt(np.sum(np.square(exp_errors)))
print(' 模型的均方根误差:',  exp_rms)

yvals=func(x,177.941963,b,c)
plot2=plt.plot(x, yvals, 'y',label='3 bottles of beer / short time')


k1 = c
k2 = b
k0 = 69984 / 2
v0 = 433.33
A3 = 435.2926
B3 = 44.3105
xT = k0 / k1 * (1 - np.exp(-k1 * 2))
def func3(t):
    cT = A3*(1-np.exp(-k2*t))-B3*(np.exp(-k2*t)-np.exp(-k1*t))
    ct = k1*xT/(k1-k2)/v0*(np.exp(-k2*(t-2))-np.exp(-k1*(t-2)))+cT*np.exp(-k2*(t-2))
    return  ct
x = np.array(range(2,21))
yvals2= func3(x)
plt.plot(x, yvals2, 'black',label='3 bottles of beer / 2 hours')

x = np.array(range(0,21))
y1=[20 for i in x]
plt.plot(x,y1,'g',label='standard')

plt.legend(loc=0)
plt.show()

