import lightgbm as lgb  
import pandas as pd  
import numpy as np  
from sklearn.metrics import roc_auc_score  
import sklearn    
import matplotlib.pyplot as plt     
from sklearn import model_selection
from sklearn.metrics import accuracy_score 
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
from sklearn.metrics import roc_curve
from sklearn.metrics import auc 
from sklearn.model_selection import KFold



def readdate(name):
    dataset = pd.read_csv("15patient_feature.txt",sep=',',header=None)
    return dataset




if __name__ == "__main__":

	dataset=readdate('15patient(gene_neighbor_1280)_sum.txt')
    plt.figure(figsize=(12,15))
    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    for i,j,x,y in zip(range(1,3),L,c1,c2):
          
        X=y1.iloc[:,1:1281].values
        Y=y1.iloc[:,1281+i].values

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
        
        plt.subplot(5,3,i)
        plt.plot(fpr, tpr, color='darkorange',lw=2,label='AUC = {0:0.2f}' ''.format(gsearch.best_score_),linewidth=2.5 )
        plt.plot([0, 1], [0, 1], 'k--',color='black',linewidth=1)
        plt.legend(loc="lower right",frameon=False)
        plt.text(0.8, 0.4, j,size=15)
        plt.tick_params(labelsize=8)
    plt.tight_layout()
    plt.show()














