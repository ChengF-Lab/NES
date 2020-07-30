import numpy as np
import pandas as pd
import sys


a=[] 
def file_name(file_dir):   
    for root, dirs, files in os.walk(file_dir):   
        for i in files:
            a.append(i) 
file_name('patient_feature\in_data')
aa=[]
for i in a:
    aa.append('in_data/'+i)

ttt = pd.read_csv("gene_neighbor.txt",sep=',',header=None)

for i in aa:
    z = pd.read_csv(j,sep=',',header=None)
    data2=z.merge(ttt,left_on=z[1],right_on=ttt[0],how='left')
    data2[3]=data2['1_y']
    data2=data2.drop(['key_0','0_y','1_y'],axis=1)
    data2.to_csv(i,index=0,header=None)