import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
from sklearn.metrics import roc_curve,roc_auc_score
from sklearn.model_selection import KFold
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
#Reading data
adult_chart=pd.read_csv('adult_icu')
baby_chart=pd.read_csv('n_icu')

#Q2
train_ad=adult_chart[adult_chart['train']==1]
test_ad=adult_chart[adult_chart['train']==0]
train_ad_label=np.asarray(train_ad['mort_icu'])
test_ad_label=np.asarray(test_ad['mort_icu'])
ad_feature_names=adult_chart.columns[[3]+list(range(7,63))]
train_ad_feat=train_ad[ad_feature_names].values
test_ad_feat=test_ad[ad_feature_names].values
#rescale feature matrix to range[0,1]
min_max_scaler = preprocessing.MinMaxScaler()
std_train_ad_feat=min_max_scaler.fit_transform(train_ad_feat)
std_test_ad_feat=min_max_scaler.fit_transform(test_ad_feat)
#Do 5-fold cross validation for regularizor parameter selection, metric=average AUC
auc=[]
hyper=(100,10,1,0.1,0.01)
for C in hyper:
    clf= LogisticRegression(C=C, penalty='l2')
    cv=KFold(n_splits=5, shuffle=True)
    scores = cross_val_score(clf, std_train_ad_feat,train_ad_label,cv=cv,scoring='roc_auc')
    auc.append(scores.mean())
#Use best C and complete training set to train logistic regression model
best_c=hyper[np.argmax(auc)]
clf_1= LogisticRegression(C=best_c, penalty='l2')
clf_1.fit(std_train_ad_feat,train_ad_label)
test_score=clf_1.predict_proba(std_test_ad_feat)[:,1]
fpr_1, tpr_1, _ = roc_curve(test_ad_label, test_score)
AUC_1=roc_auc_score(test_ad_label, test_score)
best_param=clf_1.coef_[0]
plt.plot(fpr_1, tpr_1, label='Adault_ICU_struct(AUC = %0.3f)' % (AUC_1))
rank_pred=np.argsort(best_param)
low_predictors=ad_feature_names[rank_pred[0:5]]
top_predictors=ad_feature_names[rank_pred[-5:len(ad_feature_names)]]
print (top_predictors,low_predictors)

#Q3a
train_n=baby_chart[baby_chart['train']==1]
test_n=baby_chart[baby_chart['train']==0]
train_n_label=np.asarray(train_n['mort_icu'])
test_n_label=np.asarray(test_n['mort_icu'])
n_feature_names=baby_chart.columns[[3]+list(range(7,45))]
train_n_feat=train_n[n_feature_names].values
test_n_feat=test_n[n_feature_names].values
#rescale feature matrix to range[0,1]
min_max_scaler = preprocessing.MinMaxScaler()
std_train_n_feat=min_max_scaler.fit_transform(train_n_feat)
std_test_n_feat=min_max_scaler.fit_transform(test_n_feat)   
#set features that not shown up in N-ICU into zero and fit the A-ICU model
overlap_index=np.where(np.asarray([x in n_feature_names for x in ad_feature_names])==True)[0]
padded_test_n_feat=np.zeros((len(std_test_n_feat),len(ad_feature_names)))
padded_test_n_feat[:,overlap_index]=std_test_n_feat
#Test Q2 model on N-ICU data
test_score=clf_1.predict_proba(padded_test_n_feat)[:,1]
fpr_2, tpr_2, _ = roc_curve(test_n_label, test_score)
AUC_2=roc_auc_score(test_n_label, test_score)

#Q3b
#Do 5-fold cross validation for N-ICU regularizor parameter selection, metric=average AUC
auc=[]
hyper=(100,10,1,0.1,0.01)
for C in hyper:
    clf= LogisticRegression(C=C, penalty='l2')
    cv=KFold(n_splits=5, shuffle=True)
    scores = cross_val_score(clf, std_train_n_feat,train_n_label,cv=cv,scoring='roc_auc')
    #print(scores)
    auc.append(scores.mean())
best_c=hyper[np.argmax(auc)]
clf_3= LogisticRegression(C=best_c, penalty='l2')
clf_3.fit(std_train_n_feat,train_n_label)
test_score=clf_3.predict_proba(std_test_n_feat)[:,1]
fpr_3, tpr_3, _ = roc_curve(test_n_label, test_score)
AUC_3=roc_auc_score(test_n_label, test_score)
best_param=clf_3.coef_[0]
rank_pred=np.argsort(best_param)
low_predictors_n=n_feature_names[rank_pred[0:5]]
top_predictors_n=n_feature_names[rank_pred[-5:len(n_feature_names)]]
print (top_predictors_n,low_predictors_n)

#Q4
#Converting text into bag of words vectors
Notes=pd.read_csv('adult_notes')
train_nt_label=np.asarray(Notes[Notes['train']==1]['mort_icu'])
test_nt_label=np.asarray(Notes[Notes['train']==0]['mort_icu'])
vt= CountVectorizer(max_features=1000)
X=vt.fit_transform(Notes['chartext']).toarray()
train_nt_feat=X[np.asarray(Notes['train']==1),:]
test_nt_feat=X[np.asarray(Notes['train']==0),:]
#Do 5-fold cross validation for bag of words model regularizor parameter selection, metric=average AUC
auc=[]
hyper=(100,10,1,0.1,0.01)
for C in hyper:
    clf= LogisticRegression(C=C, penalty='l1')
    cv=KFold(n_splits=5, shuffle=True)
    scores = cross_val_score(clf,train_nt_feat,train_nt_label,cv=cv,scoring='roc_auc')
    #print(scores)
    auc.append(scores.mean())
best_c=hyper[np.argmax(auc)]
clf_4= LogisticRegression(C=best_c, penalty='l1')
clf_4.fit(train_nt_feat,train_nt_label)
test_score=clf_4.predict_proba(test_nt_feat)[:,1]
fpr_4, tpr_4, _ = roc_curve(test_nt_label, test_score)
AUC_4=roc_auc_score(test_nt_label, test_score)
best_param=clf_4.coef_[0]
rank_pred=np.argsort(best_param)
vocab=np.asarray(vt.get_feature_names())
low_predictors_no=vocab[rank_pred[0:5]]
top_predictors_no=vocab[rank_pred[-5:len(vocab)]]
print (top_predictors_no,low_predictors_no)

#Q5 merge 2 datasets using icuid
Merged_chart=pd.merge(adult_chart, Notes,on=['icustay_id','mort_icu'])
train_mg_label=np.asarray(Merged_chart[Merged_chart['train_y']==1]['mort_icu'])
test_mg_label=np.asarray(Merged_chart[Merged_chart['train_y']==0]['mort_icu'])
vt= CountVectorizer(max_features=1000)
X1=vt.fit_transform(Merged_chart['chartext']).toarray()
X2=min_max_scaler.fit_transform(Merged_chart[ad_feature_names].values)
X2=Merged_chart[ad_feature_names].values
Feat=np.append(X2,X1,axis=1)
train_mg_feat=Feat[np.asarray(Merged_chart['train_y']==1),:]
test_mg_feat=Feat[np.asarray(Merged_chart['train_y']==0),:]
#Do 5-fold cross validation for bag of words model regularizor parameter selection, metric=average AUC
auc=[]
hyper=(100,10,1,0.1,0.01,0.001)
for C in hyper:
    clf= LogisticRegression(C=C, penalty='l1')
    cv=KFold(n_splits=5, shuffle=True)
    scores = cross_val_score(clf,train_mg_feat,train_mg_label,cv=cv,scoring='roc_auc')
    #print(scores)
    auc.append(scores.mean())
best_c=hyper[np.argmax(auc)]
clf_5= LogisticRegression(C=best_c, penalty='l1')
clf_5.fit(train_mg_feat,train_mg_label)
test_score=clf_5.predict_proba(test_mg_feat)[:,1]
fpr_5, tpr_5, _ = roc_curve(test_mg_label, test_score)
AUC_5=roc_auc_score(test_mg_label, test_score)
best_param=clf_5.coef_[0]
rank_pred=np.argsort(best_param)
all_feat_names=np.append(np.asarray(ad_feature_names),vocab,axis=0)
low_predictors_a=all_feat_names[rank_pred[0:10]]
top_predictors_a=all_feat_names[rank_pred[-10:len(all_feat_names)]]
print (top_predictors_a,low_predictors_a)

#Draw ROC plot
fig=plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_1, tpr_1, label='Adult_ICU_struct(AUC = %0.3f)' % (AUC_1))
plt.plot(fpr_2, tpr_2, label='Adult_on_N_ICU_struct(AUC = %0.3f)' % (AUC_2))
plt.plot(fpr_3, tpr_3, label='N_ICU_struct(AUC = %0.3f)' % (AUC_3))
plt.plot(fpr_4, tpr_4, label='Adult_ICU_unstruct(AUC = %0.3f)' % (AUC_4))
plt.plot(fpr_5, tpr_5, label='Adult_ICU_combine(AUC = %0.3f)' % (AUC_5))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('Test ROC curve of different tasks')
plt.legend(loc='best',prop={'size':8})   
fig.savefig('P5-test-auc.png')
