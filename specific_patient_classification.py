import lightgbm as lgb  
import pandas as pd  
import numpy as np  
from sklearn.metrics import roc_auc_score  
from sklearn.model_selection import train_test_split
from sklearn import model_selection
from sklearn.metrics import accuracy_score 
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn import metrics
from sklearn.utils import shuffle


data = pd.read_csv("dataset/TCGA_Clinical/CESC/nationwidechildrens.org_clinical_patient_cesc.txt",sep='\t')
df = pd.read_csv("processing/out_data/12345_3_CESC.txt",sep=',',header=None)

df1 = pd.read_csv("tf_idf.txt",sep=',',header=None)
df2=df1[df1[2]==3]
df3=df2.groupby(df2[0]).count()
a=[]
for i in df3.index:
    str=i
    a.append(str[0:12])
a1=pd.DataFrame(a)
df3.index=a
t=df3.merge(data,left_on=df3.index,right_on=data["bcr_patient_barcode"],how='left')
t=t[['key_0',1,'clinical_stage']]


data=data[['bcr_patient_barcode','clinical_stage']]
a=[]
for i in df[0]:
    str=i
    a.append(str[0:12])
a1=pd.DataFrame(a)
df[0]=a
t=df.merge(data,left_on=df[0],right_on=data["bcr_patient_barcode"],how='left')


t.loc[t['clinical_stage'] == 'Stage IA','clinical_stage'] = 'Stage I'
t.loc[t['clinical_stage'] == 'Stage IA1','clinical_stage'] = 'Stage I'
t.loc[t['clinical_stage'] == 'Stage IA2','clinical_stage'] = 'Stage I'
t.loc[t['clinical_stage'] == 'Stage IB','clinical_stage'] = 'Stage I'
t.loc[t['clinical_stage'] == 'Stage IB1','clinical_stage'] = 'Stage I'
t.loc[t['clinical_stage'] == 'Stage IB2','clinical_stage'] = 'Stage I'
t.loc[t['clinical_stage'] == 'Stage IIA','clinical_stage'] = 'Stage II'
t.loc[t['clinical_stage'] == 'Stage IIA1','clinical_stage'] = 'Stage II'
t.loc[t['clinical_stage'] == 'Stage IIA2','clinical_stage'] = 'Stage II'
t.loc[t['clinical_stage'] == 'Stage IIB','clinical_stage'] = 'Stage II'
t.loc[t['clinical_stage'] == 'Stage IIB1','clinical_stage'] = 'Stage II'
t.loc[t['clinical_stage'] == 'Stage IIB2','clinical_stage'] = 'Stage II'
t.loc[t['clinical_stage'] == 'Stage IIIA','clinical_stage'] = 'Stage III'
t.loc[t['clinical_stage'] == 'Stage IIIB','clinical_stage'] = 'Stage III'
t.loc[t['clinical_stage'] == 'Stage IIIC','clinical_stage'] = 'Stage III'
t.loc[t['clinical_stage'] == 'Stage IIIC1','clinical_stage'] = 'Stage III'
t.loc[t['clinical_stage'] == 'Stage IIIC2','clinical_stage'] = 'Stage III'
t.loc[t['clinical_stage'] == 'Stage IVA','clinical_stage'] = 'Stage IV'
t.loc[t['clinical_stage'] == 'Stage IVB','clinical_stage'] = 'Stage IV'

class_mapping = {'Stage I':1, 'Stage II':2, 'Stage III':3, 'Stage IV':4}
t['clinical_stage'] = t['clinical_stage'].map(class_mapping)
t=t[t['clinical_stage']<=4]

t['clinical_stage'] = t['clinical_stage'].astype('int')

dataset1=t.sort_values(by='clinical_stage' , ascending=True,axis=0)
dataset1=dataset1.reset_index()
y = label_binarize(dataset1['clinical_stage'], classes=[1,2,3,4])
u=pd.DataFrame(y)
u.columns=[x for x in range(1500,1504)]
y1=dataset1.join(u)
del y1['index']



zz=[]
for i in range(0,5):
    z=[]
    y1 = shuffle(y1)
    for i in range(0,4):
        array= y1.values   
        X= array[:,2:1282]       
        Y= array[:,1284+i]         
        validation_size= 0.1   
        seed= 10     
        train_x, valid_x, train_y, valid_y = train_test_split(X, Y, test_size=0.2, random_state=0)   
        train = lgb.Dataset(train_x, train_y)
        valid = lgb.Dataset(valid_x, valid_y, reference=train)
        parameters = {
                      'num_leaves': [25,30,35, 40,45,50,55,60,65],
                      'max_depth': [-1, 1, 3,5,7,10],
                      'learning_rate': [0.01,0.03,0.05,0.06,0.07,0.1,0.5]
                      'feature_fraction': [0.6, 0.7,0.8, 0.95],
                      'bagging_fraction': [0.6, 0.7,0.8, 0.95],
                      'bagging_freq': [2, 4, 5, 6,7, 8],
                      'lambda_l1': [0, 0.4, 0.6,0.7],
                      'lambda_l2': [0, 20,30, 40],
                      'cat_smooth': [1, 10, 15, 20, 35]
        }
        gbm = lgb.LGBMClassifier(boosting_type='gbdt',
                                 objective = 'binary',
                                 metric = 'auc',
                                 verbose = 0,
                                 learning_rate = 0.01,
                                 num_leaves = 35,
                                 feature_fraction=0.8,
                                 bagging_fraction= 0.9,
                                 bagging_freq= 8,
                                 lambda_l1= 0.6,
                                 lambda_l2= 0)
     
        gsearch = GridSearchCV(gbm, param_grid=parameters, scoring='roc_auc', cv=3)
        gsearch.fit(train_x, train_y)
        
        p=gsearch.predict_proba(valid_x) 
        fpr,tpr,threshold = roc_curve(valid_y.astype('int'),p[:,1].ravel())
        print(j)

        print("Best score: %0.2f" % gsearch.best_score_)
        print("Best parameters set:")
        best_parameters = gsearch.best_estimator_.get_params()
        for param_name in sorted(parameters.keys()):
            print("\t%s: %r" % (param_name, best_parameters[param_name]))
        print("......................................")
        auc=gsearch.best_score_
        z.append(round(auc,3))
    zz.append(z)


