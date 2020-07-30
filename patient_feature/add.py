import os  
a=[] 
def file_name(file_dir):   
    for root, dirs, files in os.walk(file_dir):   
        for i in files:
            a.append(i)  
file_name('..\processing\out_data')

import pandas as pd
L=[10,11, 12, 13,15, 16, 17, 18,19,1,2, 3, 4,7,8]
df=pd.DataFrame()

for i,j in zip(a,L):
    data = pd.read_csv('../processing/out_data)/'+i,sep=',',header=None)
    z=pd.DataFrame(data)
    z['c']=j
    df=pd.concat([df,z], axis=0, ignore_index=True)
    
df.to_csv('15patient_feature.txt',index=0,header=None)