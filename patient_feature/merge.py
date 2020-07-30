import numpy as np
import pandas as pd
import os  


ttt = pd.read_csv("gene_emb.txt",sep=',',header=None)
a=[]
def file_name(file_dir):   
    for root, dirs, files in os.walk(file_dir):   
        for i in files:
            a.append(i)  
            
            
file_name('dataset\Mutation_Individual')
b=[]
for i in a:
    b.append('in_data/'+i)

for i,j,k in zip(a,range(1,16),b):
    data2=data.merge(ttt,left_on=data.iloc[:,1],right_on=ttt.iloc[:,0],how='left')
    data2=data2.drop(['key_0','0_y'],axis=1)
    data2=data2[data2['2_x']==j]
    data2.to_csv(k,index=0,header=None)