#!/usr/bin/env python
# coding: utf-8

# In[142]:


import numpy as np #arrays, slower exe
import pandas as pd # data cleaning and analysis
import matplotlib.pyplot as plt #plots
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
#cv - preprocessing


# In[143]:


import os

path = os.listdir('C:/Users/RITHANYA/OneDrive/Desktop/AML PROJECT/CNN/Training/')
classes = {'no_tumor':0, 'pituitary_tumor':1} #dictionary declaration


# In[144]:


get_ipython().system('pip install opencv-python')
import cv2
cv2.__version__
Y = []
X = []
for cls in classes:
    pth = 'C:/Users/RITHANYA/OneDrive/Desktop/AML PROJECT/CNN/Training/'+cls
    for j in os.listdir(pth):
        img = cv2.imread(pth+'/'+j, 0)
        img = cv2.resize(img, (200,200))
        X.append(img)
        Y.append(classes[cls])


# In[145]:


X = np.array(X)
Y = np.array(Y)

X_updated = X.reshape(len(X), -1) #flatten the size


# In[146]:


np.unique(Y)


# In[147]:


pd.Series(Y).value_counts()


# In[148]:


X.shape, X_updated.shape


# In[149]:


plt.imshow(X[0], cmap='gray')


# In[150]:


X_updated = X.reshape(len(X), -1)
X_updated.shape


# In[151]:


xtrain, xtest, ytrain, ytest = train_test_split(X_updated, Y, random_state=10,
                                               test_size=.20)


# In[152]:


xtrain.shape, xtest.shape


# In[153]:


print(xtrain.max(), xtrain.min())
print(xtest.max(), xtest.min())
xtrain = xtrain/255
xtest = xtest/255
print(xtrain.max(), xtrain.min())
print(xtest.max(), xtest.min())


# In[154]:


from sklearn.decomposition import PCA


# In[155]:


print(xtrain.shape, xtest.shape)

pca = PCA(.98)
# pca_train = pca.fit_transform(xtrain)
# pca_test = pca.transform(xtest)
pca_train = xtrain
pca_test = xtest


# In[156]:


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


# In[157]:


import warnings
warnings.filterwarnings('ignore')

lg = LogisticRegression(C=0.1) #parameter
lg.fit(xtrain, ytrain)


# In[158]:


sv = SVC()
sv.fit(xtrain, ytrain)


# In[159]:


print("Training Score:", lg.score(xtrain, ytrain))
print("Testing Score:", lg.score(xtest, ytest))


# In[160]:


print("Training Score:", sv.score(xtrain, ytrain))
print("Testing Score:", sv.score(xtest, ytest))


# In[161]:


pred = sv.predict(xtest)


# In[162]:


misclassified=np.where(ytest!=pred)
misclassified


# In[163]:


print("Total Misclassified Samples: ",len(misclassified[0]))
print(pred[51],ytest[51])


# In[164]:


dec = {0:'No Tumor', 1:'Positive Tumor'}


# In[165]:


plt.figure(figsize=(12,8))
p = os.listdir('C:/Users/RITHANYA/OneDrive/Desktop/AML PROJECT/CNN/Testing/')
c=1
for i in os.listdir('C:/Users/RITHANYA/OneDrive/Desktop/AML PROJECT/CNN/Testing/no_tumor/')[:9]:
    plt.subplot(3,3,c)
    
    img = cv2.imread('C:/Users/RITHANYA/OneDrive/Desktop/AML PROJECT/CNN/Testing/no_tumor/'+i,0)
    img1 = cv2.resize(img, (200,200))
    img1 = img1.reshape(1,-1)/255
    p = sv.predict(img1)
    plt.title(dec[p[0]])
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    c+=1


# In[166]:


plt.figure(figsize=(12,8))
p = os.listdir('C:/Users/RITHANYA/OneDrive/Desktop/AML PROJECT/CNN/Testing/')
c=1
for i in os.listdir('C:/Users/RITHANYA/OneDrive/Desktop/AML PROJECT/CNN/Testing/pituitary_tumor/')[:16]:
    plt.subplot(4,4,c)
    
    img = cv2.imread('C:/Users/RITHANYA/OneDrive/Desktop/AML PROJECT/CNN/Testing/pituitary_tumor/'+i,0)
    img1 = cv2.resize(img, (200,200))
    img1 = img1.reshape(1,-1)/255
    p = sv.predict(img1)
    plt.title(dec[p[0]])
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    c+=1


# In[ ]:





# In[ ]:





# In[167]:


from sklearn import metrics


# In[168]:


confusion_matrix = metrics.confusion_matrix(ytest, pred)


# In[169]:


cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])


# In[170]:


import matplotlib.pyplot as plt


# In[171]:


cm_display.plot()
plt.show()


# In[172]:


Accuracy = metrics.accuracy_score(ytest, pred)


# In[173]:


Precision = metrics.precision_score(ytest, pred)


# In[174]:


Sensitivity_recall = metrics.recall_score(ytest, pred)


# In[175]:


F1_score = metrics.f1_score(ytest, pred)


# In[176]:


print({"Accuracy":Accuracy,"Precision":Precision,"Sensitivity_recall":Sensitivity_recall,
       "F1_score":F1_score})


# In[177]:


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import metrics


# In[178]:


random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(xtrain, ytrain)
# Predict Output
rf_predicted = random_forest.predict(xtest)

random_forest_score = round(random_forest.score(xtrain, ytrain) * 100, 2)
random_forest_score_test = round(random_forest.score(xtest, ytest) * 100, 2)
print('Random Forest Score: \n', random_forest_score)
print('Random Forest Test Score: \n', random_forest_score_test)
print('Accuracy: \n', accuracy_score(ytest,rf_predicted))
print(confusion_matrix(ytest,rf_predicted))
print(classification_report(ytest,rf_predicted))
confusion_matrix = metrics.confusion_matrix(ytest,rf_predicted)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])
cm_display.plot()
plt.show()


# In[184]:


logreg = LogisticRegression()
import numpy as np
# Train the model using the training sets and check score
logreg.fit(xtrain, ytrain)

# Predict Output
log_predicted= logreg.predict(xtest)

logreg_score = round(logreg.score(xtrain, ytrain) * 100, 2)
logreg_score_test = round(logreg.score(xtest, ytest) * 100, 2)

# Equation coefficient and Intercept
print('Logistic Regression Training Score: \n', logreg_score)
print('Logistic Regression Test Score: \n', logreg_score_test)

print(classification_report(ytest,log_predicted))
confusion_matrix = metrics.confusion_matrix(ytest,log_predicted )
print(confusion_matrix)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])
cm_display.plot()
plt.show()


# In[186]:


gaussian = GaussianNB()
gaussian.fit(xtrain, ytrain)
# Predict Output
gauss_predicted = gaussian.predict(xtest)

gauss_score = round(gaussian.score(xtrain, ytrain) * 100, 2);
gauss_score_test = round(gaussian.score(xtest, ytest) * 100, 2);
print('Gaussian Score: \n', gauss_score);
print('Gaussian Test Score: \n', gauss_score_test);
print('Accuracy: \n', round(accuracy_score(ytest, gauss_predicted)*100,2))

print(classification_report(ytest,gauss_predicted))
confusion_matrix = metrics.confusion_matrix(ytest,gauss_predicted )
print(confusion_matrix)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])
cm_display.plot()
plt.show()


# In[102]:


models = pd.DataFrame({
    'Model': [ 'Logistic Regression', 'Gaussian Naive Bayes','Random Forest','SVM Model'],
    'Score': [ logreg_score, gauss_score, random_forest_score,sv.score(xtrain, ytrain)*100],
    'Test Score': [ logreg_score_test, gauss_score_test, random_forest_score_test,sv.score(xtest, ytest)*100]})
models.sort_values(by='Test Score', ascending=False)


# In[ ]:





# In[ ]:




