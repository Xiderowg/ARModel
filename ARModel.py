#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2017-05-05 21:31:16
# @Author  : 林子珩 (zhlin@hhu.edu.cn)
# @Link    : https://edlinus.cn
# @Version : 0.1
import os
import numpy as np
import matplotlib.pyplot as plt

'''
计算Cvx
k:模比系数矩阵
'''
def cvx(k):
    cv=0
    n=len(k);sum=0
    tmp=k-1
    tmp=np.power(tmp,2)
    for num in tmp:
        sum+=num
    cv=np.sqrt(sum/(n-1))
    return cv
'''
计算Csx
k:模比系数矩阵
cv:离势系数
'''
def csx(k,cv):
    cs=0
    n=len(k);sum=0
    tmp=k-1;
    tmp=np.power(tmp,3)
    for num in tmp:
        sum+=num
    cs=sum/(np.power(cv,3)*(n-3))
    return cs
'''
计算相关系数
x:系列
ss:下标号
'''
def rho(x,ss):
    xt=x[ss:]
    xl=x[:len(x)-ss]
    cor=np.corrcoef(xt,xl)
    return cor
'''
计算h回归系数
x:系列
n:阶数
'''
def fhi(x,n):
    r=np.zeros([n])
    f=np.zeros([n,n])
    for i in range(1,n+1):
        r[i-1]=rho(x,i)[0,1]
    f[0,0]=r[0]#初值
    for j in range(1,n):
        cr=cross(f,r,j)
        f[j,j]=(r[j]-cr[0])/(1-cr[1])
        for k in range(0,j):
            f[j,k]=f[j-1,k]-f[j,j]*f[j-1,j-k-1]
    return f,r
'''
交叉乘
m1:系列1
m2:系列2
n:位置
'''
def cross(m1,m2,n):
    sum_1=0;sum_2=0
    for i in range(0,n):
        sum_1+=m1[n-1,i]*m2[n-i-1]
        sum_2+=m1[n-1,i]*m2[i]
    return sum_1,sum_2
'''
计算回归系数和相关系数的积
f:回归系数矩阵
r:相关系数矩阵
n:矩阵长度
'''
def fhirho(f,r,n):
    sum=0
    for i in range(0,n):
        sum+=f[n-1,i]*r[i]
    return sum
'''
计算最小值及下标
m:系列
'''
def min(m):
    i=0;mnum=m[i]
    for x in range(1,len(m)):
        if m[x]<mnum:
            i=x
            mnum=m[x]
    return mnum,i
'''
计算数值在数组中的位置
num:数字
list:数组
'''
def pos(num,list):
    ub = 0;db=0
    for i in range(0,len(list)-1):
        ubn=list[i];dbn=list[i+1]
        if num>=ubn and num<=dbn :
            ub=i;db=i+1
    return ub,db
'''
生成PIII型分布的随机误差
u:0-1均匀分布的随机数
csr:随机误差的离势系数
plist:φ值表矩阵
'''
def p3num(u,csr,dr,plist):
    u=u*100
    xp=pos(csr,plist[:,0])
    yp=pos(u,plist[0,:])
    if xp[0]==xp[1] or yp[0]==yp[1]:
        return 0
    cst1=plist[xp[0],yp[0]]+plist[xp[1],yp[0]]*(u-plist[0,yp[0]])/(plist[0,yp[1]]-plist[0,yp[0]])
    cst2=plist[xp[0],yp[1]]+plist[xp[1],yp[1]]*(u-plist[0,yp[1]])/(plist[0,yp[1]]-plist[0,yp[0]])
    p3n=cst1+(cst2-cst1)*(csr-plist[xp[0],0])/(plist[xp[1],0]-plist[xp[0],0])
    p3n=dr*p3n
    return p3n
filepath=os.path.dirname(os.path.realpath(__file__))
x=np.loadtxt(filepath+"\\x.dat")#数据导入
n=len(x) #数据数目

pmax=int(n/5) #最大阶数上界
pmin=int(n/10) #最大阶数下界
fpe=np.zeros([pmax-pmin+1]) #模型误差矩阵
xm=np.mean(x) #平均值
y=x-xm #中心化系列
k=x/xm #模比系数
cv=cvx(k) #离势系数
cs=csx(k,cv) #偏态系数
dx=np.std(x) #标准差
'''
flag=0
for p in range(pmin,pmax+1):
    (f,r)=fhi(x,p) #回归系数及相关系数矩阵
    dk=np.power(dx,2)*(1-fhirho(f,r,len(r))) #均方差
    fperr=dk*(1+3*p/n) #λ=3时的模型误差
    fpe[flag]=fperr
    flag+=1
p=min(fpe)[1]+pmin
'''
p=pmax
#TODO
(f,r)=fhi(x,p)
cof=f[len(r)-1,:] #回归系数
b=xm #回归方程截距
str1=""
for c in range(0,len(cof)):
    b+=-cof[c]*xm
    if cof[c]>=0:
        str1+='+'+str(cof[c]) + "x|t-" + str(c+1)
    else:
        str1+=str(cof[c]) + "x|t-" + str(c+1)
str0="回归方程为：y="
str0+=str(b) + str1 + "+rt"
print(str0)

plist=np.loadtxt(filepath+"\\φ.dat")
r_2=0 #R^2
for i in range(0,p):
    r_2+=f[p-1,i]*r[i]
dr=dx*np.sqrt(1-r_2) #随机变量的标准差
csr=cs*(1-np.power(r_2,1.5))/np.power(1-r_2,1.5) #随机变量的偏态系数

#fp = input("请输入预报长度:\n") #预报长度
fp=10
yf=np.zeros([fp])
xin=x[len(x)-len(cof):]
for i in range(0,fp):
    u=np.random.rand()
    p3n=p3num(u,csr,dr,plist)
    #print(p3n)
    yf[i]+=b+p3n
    for j in range(0,len(cof)):
        yf[i]+=xin[j]*cof[len(cof)-j-1]
    for k in range(0,len(cof)-1):
        xin[k]=xin[k+1]
    xin[len(cof)-1]=yf[i]

np.savetxt(filepath+"\\y.dat",yf,fmt='%s')
print("预报结果已保存在"+filepath+"\\y.dat")
plt.figure(figsize=(8,4))
xl=range(1,len(yf)+1)
plt.plot(xl,yf,label="$Forecast Stage(m)$",color="red",linewidth=2)
plt.ylabel("Stage(m)")
plt.xlabel("Time(day)")
plt.legend()
plt.show()