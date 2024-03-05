import numpy as np
import os
from sklearn import  linear_model,metrics
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc as sklearn_auc
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
import scipy
from scipy import stats
from scipy.stats import chisquare,kstest,pearsonr,spearmanr
#######  keras #########
import keras
from keras.layers import Input, Conv1D, Conv2D, Activation, MaxPool2D, Flatten, Dense, Dropout, BatchNormalization, ZeroPadding2D,Concatenate
from keras.models import Sequential
from keras.models import load_model
from keras.models import Model
from keras.constraints import max_norm
####### matplotlib ######## 
import matplotlib
matplotlib.use('Agg') 
import matplotlib.cbook as cbook
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot,savefig

def Confusion_Matrix(true_label,pre_label):
	TP=float(list(true_label+pre_label).count(2))
	FN=float(list(true_label-pre_label).count(1))
	FP=float(list(true_label-pre_label).count(-1))
	TN=float(list(true_label+pre_label).count(0))
	try:
		acc=(TP+TN)/(TP+FN+FP+TN)
	except:
		acc=0
	try:
		pre=(TP)/(TP+FP)
	except:
		pre=0
	try:
		tpr=(TP)/(TP+FN)
	except:
		tpr=0
	try:
		tnr=(TN)/(TN+FP)
	except:
		tnr=0
	try:
		F1=(2*pre*tpr)/(pre+tpr)
	except:
		F1=0
	return TP,FN,FP,TN,acc,pre,tpr,tnr,F1


def model_check(true_label,pre_label,pre_value):
	TP,FN,FP,TN,acc,pre,tpr,tnr,F1=Confusion_Matrix(true_label,pre_label)
	fpr_list, tpr_list, _ = metrics.roc_curve(true_label,pre_value,pos_label=1)	
	roc_auc=metrics.auc(fpr_list, tpr_list)	
	lr_precision, lr_recall, _ = precision_recall_curve(true_label,pre_value,pos_label=1)	
	PR_auc=sklearn_auc(lr_recall, lr_precision)
	return TP,FN,FP,TN,acc,pre,tpr,tnr,F1,roc_auc,PR_auc	


def data_load_model(file_name,X_train,y_train):
	f1=open(file_name,'r')
	m1=f1.readlines()
	f1.close()
	for i in range(1,len(m1)):
		p1=m1[i].strip().split('\t')
		tp=[]
		for j in range(1,len(p1)-1):
			tp.append(float(p1[j]))
		X_train.append(np.array(tp))
		y_train.append(float(p1[-1]))
	return X_train,y_train

def socre_label(pre_score):
	pre_label=[]
	for tp in pre_score:
		if tp[0] > tp[1]:
			pre_label.append(0)
		else:
			pre_label.append(1)
	return pre_label



floder=os.getcwd()
file1='%s/train/5_fold_1.txt'%floder
file2='%s/train/5_fold_2.txt'%floder
file3='%s/train/5_fold_3.txt'%floder
file4='%s/train/5_fold_4.txt'%floder
file5='%s/train/5_fold_5.txt'%floder
file6='%s/train/vaildation_data.txt'%floder

data_group=[[file1,file2,file3,file4,file5]]
data_name=['file1','file2','file3','file4','file5']

X_train=[]
y_train=[]
for j in range(0,len(data_group[0])):
	X_train,y_train=data_load_model(data_group[0][j],X_train,y_train)

X_val=[]
y_val=[]
X_val,y_val=data_load_model(file6,X_val,y_val)
X_train=np.array(X_train)
y_train=np.array(y_train)
X_val=np.array(X_val)
y_val=np.array(y_val)
input_all = Input(shape=(len(X_train[0])))
Dense1 = Dense(32,kernel_initializer=keras.initializers.he_uniform(seed=None),activation='relu')(input_all)
Dropout1 = Dropout(0.25)(Dense1)
Batch1 = BatchNormalization()(Dropout1)
Dense22 = Dense(64,kernel_initializer=keras.initializers.he_uniform(seed=None),activation='relu')(Batch1)
Dropout22 = Dropout(0.25)(Dense22)
Batch22 = BatchNormalization()(Dropout22)
Dense32 = Dense(32,kernel_initializer=keras.initializers.he_uniform(seed=None),activation='relu')(Batch22)
Dropout32 = Dropout(0.25)(Dense32)
Batch32 = BatchNormalization()(Dropout32)
Dense2 = Dense(16,kernel_initializer=keras.initializers.he_uniform(seed=None),activation='relu')(Batch32)
Dropout2 = Dropout(0.25)(Dense2)
Batch2 = BatchNormalization()(Dropout2)
output = Dense(1, activation='linear')(Batch2)
model_linear = Model(inputs=input_all, outputs=output)
adam=keras.optimizers.Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.001)
model_linear.compile(loss='mse', metrics=['mse'], optimizer=adam)
all_epoch=32
batch_size=64
for epoch in range(all_epoch):
	model_linear.fit(X_train, y_train, batch_size=batch_size,epochs=256,validation_data=(X_val, y_val),verbose=0)
mp = "%s/final_float_model.h5"%(floder)
model_linear.save(mp)
	

