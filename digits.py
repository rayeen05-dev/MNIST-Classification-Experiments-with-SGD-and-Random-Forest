from sklearn.datasets import load_digits 
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier 
from sklearn.model_selection import cross_val_score , cross_val_predict
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import precision_score , recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve 
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

digits = load_digits()

#exploring the data and its target classification 
x , y = digits["data"] , digits["target"] 

num = x[600]
"""num_image = num.reshape(8,8)
plt.imshow(num_image, cmap = matplotlib.cm.binary,
interpolation="nearest")
plt.axis("off")
print(y[1500])
plt.show()"""

#spliting the dataset 

# not a good aproach of spliting the data 
"""x_train, x_test, y_train, y_test = x[:1200], x[1200:], y[:1200], y[1200:]
shuffle_index = np.random.permutation(1200)
x_train,y_train = x_train[shuffle_index] , y_train[shuffle_index] """

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.3,
    stratify=y,
    random_state=42
)
#BINARY CLASSIFICATION 
#training and testing only on number 8 to simplify 

y_train_8 = (y_train == 8 )
y_test_8 = (y_test == 8 )

sgd_clf = SGDClassifier(random_state=42) 
sgd_clf.fit(x_train,y_train_8) #only fiting on the 8 digit data 
#evaluating the classifier  
"""print(cross_val_score(sgd_clf,x_train , y_train_8 , cv=5 , scoring="accuracy"))"""

#using confusion matrix to see the rate of when the model confused between 8 and other digits 
y_train_predict = cross_val_predict(sgd_clf,x_train,y_train_8,cv=3)

"""print(confusion_matrix(y_train_8,y_train_predict))"""


#a perfect classifier would have only true-positives and true-negatives (top right and bottom left = 0 )

#calculating prediction and recall scores 

"""print(precision_score(y_train_8 , y_train_predict)) 
print(recall_score(y_train_8,y_train_predict))"""

#despite having a high accuracy the recall and precision scores seems awfull => not yet a good model 

#using the f1-score that combines precison nad recal into a single value 

"""print(f1_score(y_train_8,y_train_predict))"""

#manually changing the threshold to see how the model reacts when having a very high one or a very low one 
num1 = x[253]
y_score = sgd_clf.decision_function(num1.reshape(1,-1)) 
"""print(y_scores)
print(y[253])"""
threshold =  -10.0 # with very low threshold even an 8 the model couldnt recognize it so basically its always false 
y_num_pred = (y_score > threshold) 
"""print(y_num_pred)"""

#we have to variey between thresholds to see which one suits owr case 

y_scores = cross_val_predict(sgd_clf,x_train,y_train_8,cv=3,method="decision_function")
precisions , recalls , thresholds = precision_recall_curve(y_train_8 , y_scores)
"""def plot_precesion_recall_vs_threshold(precisions , recalls , thresholds) : 
    plt.plot(thresholds,precisions[:-1],"b--",label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    plt.xlabel("Threshold")
    plt.legend(loc="center left")
    plt.ylim([0, 1])
plot_precesion_recall_vs_threshold(precisions,recalls,thresholds)
plt.show()"""

#another method to find the suitable threshold ROC curve 

fpr,tpr,thresholds  = roc_curve(y_train_8 , y_scores)
def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
"""plot_roc_curve(fpr, tpr)
plt.show()"""

#ROC AUC a way to compare classifiers to mesure the area under the curve of the roc 
"""print(roc_auc_score(y_train_8,y_scores))"""

#comparing randomforest classi to SGD classi 

forest_clf = RandomForestClassifier(random_state=42)
"""y_proba_forest = cross_val_predict(forest_clf,x_train,y_train_8,cv=3,method="predict_proba")
y_scores_forest = y_proba_forest[:,1] 
fpr_forest , tpr_forest , thresholds_forest = roc_curve(y_train_8 , y_scores_forest) 
plt.plot(fpr, tpr, "b:", label="SGD")
plot_roc_curve(fpr_forest, tpr_forest, "Random Forest")
plt.legend(loc="lower right")
plt.show()
print(roc_auc_score(y_train_8 , y_scores_forest))"""
#=>random forest with a better pr curve and a better roc_auc_score 0.9896 : randomforest ; 0.966 : SGD
#“Random Forest captures digit patterns better than a linear model for this task.”

#MULTICLASS CLASSIFICATION 
#going with OvA methode 

sgd_clf.fit(x_train,y_train) #going with the whole data 
#what happened here is that scikit-learn trained 10 binary classifieres got their decision scores for the image 
#and selected the class with the heighest score 
"""print(y[50])
print(sgd_clf.predict([x[50]]))"""

#randomforest 

forest_clf.fit(x_train,y_train) # randomforest can directly claasify instances into multiple classes 
"""print(forest_clf.predict([x[200]]))
print(forest_clf.predict_proba([x[200]])) # array of proba assigned to each class 
print(y[200])"""

#ANALYZING ERRORS 
scaler = StandardScaler()
x_train_scaled =  scaler.fit_transform(x_train.astype(np.float64))
y_train_pred = cross_val_predict(sgd_clf,x_train_scaled,y_train,cv=3 ) #making predictions 

conf_mx = confusion_matrix(y_train,y_train_pred) # confusion matrix 
"""plt.matshow(conf_mx,cmap=plt.cm.gray)
plt.show()"""

row_sums = conf_mx.sum(axis=1,keepdims=True )
norm_conf_mx = conf_mx / row_sums # deviding each value in the confusion matrix by the number of images 
#so we can compare error rates per image

np.fill_diagonal(norm_conf_mx , 0 )
plt.matshow(norm_conf_mx, cmap=plt.cm.gray)
plt.show()