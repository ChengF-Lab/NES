import numpy as np
import pandas as pd
import os  


a=[] 

def file_name(file_dir):   
    for root, dirs, files in os.walk(file_dir):   
        for i in files:
            a.append(i)  
            
            
file_name('patient_feature\in_data')
aa=[]
for i in a:
    aa.append('in_data/'+i)
    bb=[]
for i in a:
    bb.append('out_data/1_10_'+i)
    
    
    
for m,n in zip(aa,bb):
    data = pd.read_csv(m,sep=',',header=None)
    #data=data.drop(132,axis=1)
    data=data.dropna(axis=0, how='any')
    data=data.sort_values(by=3)
    data['c']=1
    q=[3.0,4.0,6.0,8.0,11.0,14.0,17.0,21.0,25.0]
    w=[2,3,4,5,6,7,8,9,10]
    #l=[2,3,4,5,6,7,8,9,10]
    for i,j in zip(q,w):
        data.loc[data[3]>i,'c']=j
    df = data.groupby([data[0],data['c']]).sum()
    df1=df.drop([1,2,3],axis=1)
    #print(df1)
    df1.to_csv('group.txt',header=None)
    df11=pd.read_csv("group.txt",sep=',',header=None)
    LL=[]
    for i in range(0,len(list(set(df11[0])))):
        for j in range(0,10):
            LL.append(list(set(df11[0]))[i])
    LL1=[]
    for i in range(0,len(list(set(df11[0])))):
        for j in range(1,11):
            LL1.append(j)
    dd=np.array(LL).T
    dd1=pd.DataFrame(dd)
    dd1[1]=LL1
    df0=pd.merge(dd1,df11,on = [0,1],how='left')
    df0=df0.fillna(0)
    p1=df0.iloc[0::10,:]
    p2=df0.iloc[1::10,:]
    p3=df0.iloc[2::10,:]
    p4=df0.iloc[3::10,:]
    p5=df0.iloc[4::10,:]
    p6=df0.iloc[5::10,:]
    p7=df0.iloc[6::10,:]
    p8=df0.iloc[7::10,:]
    p9=df0.iloc[8::10,:]
    p10=df0.iloc[9::10,:]
    q=np.hstack((p1,p2,p3,p4,p5,p6,p7,p8,p9,p10))
    zz=pd.DataFrame(q)
    print(zz)
    for i in range(129,1170,130):
        zz=zz.drop(i,axis=1)
    for i in range(130,1171,130):
        zz=zz.drop(i,axis=1)
    zz=zz.drop(1,axis=1)
    #[1,32,33,64,65,96,97,128,129]
    zz.to_csv(n,header=None,index=0)