import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
from sklearn import preprocessing


def readdate(name):
    df = pd.read_csv('gene_mutation.txt', sep=',', header=None)
    return df


if __name__ == "__main__":

	df=readdate('gene_mutation.txt')
	del df[0]
	t2 = df.groupby([1,2]).size().unstack(level = -1, fill_value = 0)
	z=t2.iloc[0:28,]

	scaler_value = sklearn.preprocessing.StandardScaler()
	train_values = scaler_value.fit_transform(z)
	zz=pd.DataFrame(train_values)

	f, ax = plt.subplots(figsize = (15, 12))
	sns.heatmap(zz, cmap = 'RdBu_r', linewidths = 0.05, ax = ax, vmin=-1, vmax=1)
	ax.set_title('Gene mutation distribution',fontsize=27)
	ax.set_ylabel('gene',fontsize=25)
	ax.set_xticklabels(['BLCA','BRCA','CESC','COAD','HNSC','KIRC','LIHC','LUAD','LUSC','OV','READ','SKCM','STAD','THCA','UCEC']) 
	ax.set_ylim(28, 0)
	plt.yticks([])
	plt.xticks(rotation=45)
	cax = plt.gcf().axes[-1]
	cax.tick_params(labelsize=17)
	plt.tick_params(labelsize=18)
	plt.show()

