import pandas as pd  
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
from lifelines.statistics import logrank_test
from lifelines import KaplanMeierFitter
from sklearn import metrics
from sklearn.metrics import pairwise_distances
from lifelines.statistics import pairwise_logrank_test
from lifelines.statistics import multivariate_logrank_test
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
from lifelines.statistics import logrank_test
from sklearn import metrics
from sklearn.metrics import pairwise_distances


def EuclideanDistances(A, B):
    BT = B.transpose()
    vecProd = np.dot(A,BT)
    SqA =  A**2
    sumSqA = np.matrix(np.sum(SqA, axis=1))
    sumSqAEx = np.tile(sumSqA.transpose(), (1, vecProd.shape[1]))
    SqB = B**2
    sumSqB = np.sum(SqB, axis=1)
    sumSqBEx = np.tile(sumSqB, (vecProd.shape[0], 1))
    SqED = sumSqBEx + sumSqAEx - 2*vecProd
    SqED[SqED<0]=0.0
    ED = np.sqrt(SqED)
    return ED

dataset4 = pd.read_csv("processing/out_data/12345_3_CESC.txt",header=None,sep=',')
df = pd.read_csv("dataset/TCGA_Clinical/CESC/nationwidechildrens.org_clinical_patient_cesc.txt",sep='\t')
df=df[["bcr_patient_barcode","gender","vital_status","last_contact_days_to","death_days_to"]]

a=[]
for i in dataset4[0]:
    str=i
    a.append(str[0:12])
a1=pd.DataFrame(a)
dataset4[1282]=a1

t=df.merge(dataset4,left_on=df["bcr_patient_barcode"],right_on=dataset4[1282],how='left')
class_mapping = {'Dead':0, 'Alive':1}
t["vital_status"] = t["vital_status"].map(class_mapping)
del t["key_0"]
del t[1282]
t=t.dropna(axis=0,how='any')




for i in range(8,80):
    for j in range(2,10):
        X=t.iloc[:,6:1286]
        db = DBSCAN(eps=i, min_samples=j).fit(X)
        labels = db.labels_
        if len(set(list(labels)))==2:
            result = Counter(labels)
            print(i,j,len(set(list(labels))),result)


X=t.iloc[:,6:1286]
db = DBSCAN(eps=75, min_samples=10).fit(X)
labels = db.labels_ 
t['cluster_db'] = labels  
t=t.sort_values('cluster_db')
data_zs=t.iloc[:,6:1286]
tsne=TSNE()
tsne.fit_transform(data_zs)  
#a=tsne.fit_transform(data_zs) 
tsne=pd.DataFrame(tsne.embedding_,index=data_zs.index)


d=cosine_similarity(t1.values, t1.values)

 
Euclidean_dis=EuclideanDistances(tsne.values,tsne.values)
d1=pd.DataFrame(Euclidean_dis)
d1=-d1


f, ax = plt.subplots(figsize = (5, 4))
#cmap = sns.cubehelix_palette(start = 1, rot = 3, gamma=0.8, as_cmap = True)
sns.heatmap(d1, cmap = 'Blues', ax = ax, vmin=-10, vmax=0,cbar = False)
ax.set_title('CESC',fontsize=15)
ax.set_ylim([len(d1), 0])
plt.xticks([])
plt.yticks([])
plt.show()


data_zs=t.iloc[:,6:156]
tsne=TSNE()
tsne.fit_transform(data_zs)  
tsne=pd.DataFrame(tsne.embedding_,index=data_zs.index)
d1=tsne[t['cluster_db']==0]
plt.plot(d1[0],d1[1],'g*',markersize=3,label='Subtype I')
d2=tsne[t['cluster_db']==-1]
plt.plot(d2[0],d2[1],'r*',markersize=3,label='Subtype II')
plt.legend(loc='best',frameon=False)
plt.title("CESC",fontsize=15)
plt.xticks([])
plt.yticks([])
plt.xlim(-25,25)
plt.ylim(-25,25)
plt.show()


dataset55=t[["vital_status","last_contact_days_to","death_days_to","cluster_db"]].sort_values('cluster_db')
t4=dataset55[dataset55['cluster_db']==0]
t4.loc[t4.last_contact_days_to=='[Not Available]','last_contact_days_to']=0
t4.loc[t4.last_contact_days_to=='[Not Applicable]','last_contact_days_to']=0
t4.loc[t4.death_days_to=='[Not Available]','death_days_to']=0
t4.loc[t4.death_days_to=='[Not Applicable]','death_days_to']=0
aaa33=t4.astype(int)
aaa33['time']=aaa33['last_contact_days_to']+aaa33['death_days_to']


T2 = aaa33['time']
E2 = aaa33['vital_status']
T0=T2
E0=E2
kmf00 = KaplanMeierFitter()
kmf00.fit(T2.astype('int'), event_observed=E2.astype('int')) # more succiently, kmf.fit(T,E)
kmf00.plot()
plt.show()

t5=dataset55[dataset55['cluster_db']==-1]
t5.loc[t5.last_contact_days_to=='[Not Available]','last_contact_days_to']=0
t5.loc[t5.last_contact_days_to=='[Not Applicable]','last_contact_days_to']=0
t5.loc[t5.death_days_to=='[Not Available]','death_days_to']=0
t5.loc[t5.death_days_to=='[Not Applicable]','death_days_to']=0
aaa33=t5.astype(int)
aaa33['time']=aaa33['last_contact_days_to']+aaa33['death_days_to']


T2 = aaa33['time']
E2 = aaa33['vital_status']
T1=T2
E1=E2
kmf11 = KaplanMeierFitter()
kmf11.fit(T2.astype('int'), event_observed=E2.astype('int')) # more succiently, kmf.fit(T,E)
kmf11.plot()
plt.show()



result1=logrank_test(T0, T1, event_observed_A=E0, event_observed_B=E1)
print(result1.p_value)
h1=kmf00.survival_function_
h2=kmf11.survival_function_
x1=h1.index
y1=h1
x2=h2.index
y2=h2
plt.plot(x1,y1,color='green', label='Subtype I',linewidth=2,marker='+')
plt.plot(x2,y2,color='red',label='Subtype II',linewidth=2,marker='+')
plt.legend(loc='best',frameon=False)
plt.title("CESC Survival Analysis")
plt.xlabel('Time(days)')
plt.ylabel('Survival Probability')
plt.text(1900, 0.5, r'$p=0.01$',size=13)
plt.tick_params(labelsize=8)
plt.savefig('CESC_survival.pdf',bbox_inches='tight',dpi=400)
plt.show()


print(metrics.silhouette_score(X, labels, metric='euclidean'))
print(metrics.calinski_harabasz_score(X, labels))