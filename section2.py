import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegressionCV, LassoCV 
from sklearn import linear_model
from sklearn.model_selection import cross_val_score,KFold
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder,LabelEncoder,MinMaxScaler
from sklearn.metrics import roc_curve,roc_auc_score
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt

twins_chart=pd.read_csv('twins')
single_chart=pd.read_csv('singletons')

#Q1
#Selecting good twins
under=twins_chart['dbirwt']<2700
good=np.logical_xor(np.asarray(under[0:len(under):2]),np.asarray(under[1:len(under):2]))
good_index=np.where(good==True)[0]
#calculating ATE
sum_ite=0
for x in good_index:
    if twins_chart['dbirwt'].iloc[2*x]<2700:
        ite=twins_chart['mort'].iloc[2*x]-twins_chart['mort'].iloc[2*x+1]
    else:
        ite=twins_chart['mort'].iloc[2*x+1]-twins_chart['mort'].iloc[2*x]
    sum_ite=sum_ite+ite
ATE_1=float(sum_ite)/float(len(good_index))
print ("the ATE is %f" % ATE_1)

#Q2
label=single_chart['dbirwt']
#convert categorical variable to one-hot
enc = OneHotEncoder()
le = LabelEncoder()
X=np.transpose(np.asarray([le.fit_transform(single_chart['mrace'].values),le.fit_transform(single_chart['dmeduc'].values),le.fit_transform(single_chart['frace'].values),le.fit_transform(single_chart['dfeduc'].values)]))
cat=enc.fit_transform(X).toarray()
feat_name=single_chart.columns[[1,3]+list(range(6,26))+[27,28]]
treat=single_chart['tobacco'].values
feat=np.append(single_chart[feat_name].values,cat,axis=1)
min_max_scaler =MinMaxScaler()
std_feat=min_max_scaler.fit_transform(feat)
reg_input=np.append(treat.reshape((len(treat),1)),std_feat,axis=1)
#fit LASSO with cross validation
model = LassoCV(cv=10).fit(reg_input, label)
print ("the Treatment weight is %f" % model.coef_[0])

#Q3
#Fit L1 logistic regression
lr=LogisticRegressionCV(Cs=[10,1,0.1,0.05,0.01],cv=10,penalty='l1',scoring='roc_auc',solver='liblinear')
model2=lr.fit(std_feat,treat)
prob=lr.predict_proba(std_feat)
plt.style.use('seaborn-white')
fpr, tpr, _ = roc_curve(treat, prob[:,1])
AUC=roc_auc_score(treat,prob[:,1])
fig=plt.figure(figsize=(5,4))
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr, label='AUC = %0.3f' % (AUC))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve for Inverse label prediction')
plt.legend(loc='best') 
plt.show()
#Compute ATE
sum1=0
sum0=0
prop_score1=[]
prop_score0=[]
for i in range(len(treat)):
    if treat[i]:
        sum1=sum1+(label[i])/prob[i,1]
        prop_score1.append(prob[i,1])
    else:
        sum0=sum0+(label[i])/prob[i,0]
        prop_score0.append(prob[i,0])
ATE_3=(sum1-sum0)/len(treat)
print ("the ATE is %f" % ATE_3)

#plot propensity score histogram
sns.set(color_codes=True)
sns.set(rc={"figure.figsize": (12, 8),'axes.labelsize': 30, 'font.size': 30, 'legend.fontsize': 30, 'axes.titlesize': 30})
ax=sns.distplot(np.asarray(prop_score1),hist_kws={"normed":0,"label":"smoke yes"})
ax=sns.distplot(np.asarray(prop_score0),hist_kws={"normed":0,"label":"smoke no"})
plt.legend(loc='upper left')
fig = ax.get_figure()
fig.savefig('score_dist.png')

