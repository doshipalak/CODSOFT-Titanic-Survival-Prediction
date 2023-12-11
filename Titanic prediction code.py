import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
df=pd.read_csv("/kaggle/input/titanic-survival-prediction/tested.csv")
df.head(2)
df.shape
df.duplicated().sum()
df.info()
df.isnull().sum()

#Filling Missing Values

for i in df.select_dtypes(include="float64"):
    df[i].fillna(df[i].median(),inplace=True)

# DATA CLEANING
df["Ticket"].unique()
l=[]
for i in df["Ticket"]:
    ticket=i.split(" ")
    if len(ticket)==1:
        l.append(ticket[0])
    elif len(ticket)==2:
        l.append(ticket[1])
    elif len(ticket)==3:
        l.append(ticket[2])
df["Ticket"]=l
df["Ticket"]=df["Ticket"].astype("int32")

df["Ticket"].unique()

# DATA PREPROCESSING

df["Embarked"].replace(["S","C","Q"],["Southampton","Cherbourg","Queenstown"],inplace=True)
conditions = [
    ((df["Age"]>0) & (df["Age"]<13.0)),
    (df["Age"] >=13.0) & (df["Age"] <18.0),
    (df["Age"]>=18.0) & (df["Age"]<32.0),
    (df["Age"]>=32.0)& (df["Age"]<50.0),
    (df["Age"]>=50.0)]
values = ["Child","Teenage","Young","Mid-Age","Senior"]
df['Age Group'] = np.select(conditions, values)

df["Family"]=df["SibSp"]+df["Parch"]
df.head(2)
#Data Visualisation
plt.figure(figsize=(12,10))
plt.subplot(2,2,1)
plt.pie(x=df["Sex"].value_counts().values,labels=df["Sex"].value_counts().index,autopct="%0.2F%%")
plt.pie([1],radius=0.5,colors="white")
plt.title("Passenger Aboard on Ship Based on Gender",fontsize=10)
plt.legend(loc=1)

plt.subplot(2,2,2)
plt.pie(x=df["Pclass"].value_counts().values,labels=df["Pclass"].value_counts().index,autopct="%0.2F%%")
plt.pie([1],radius=0.5,colors="white")
plt.title("Distribution of Passenger Aboard Classwise",fontsize=10)
plt.legend(loc=1)

plt.subplot(2,2,3)
plt.pie(x=df["Survived"].value_counts().values,labels=df["Survived"].value_counts().index,autopct="%0.2F%%")
plt.pie([1],radius=0.5,colors="white")
plt.title("Distribution tage of Passenger life Status",fontsize=10)
plt.legend(loc=1)

plt.subplot(2,2,4)
plt.pie(x=df["Age Group"].value_counts().values,labels=df["Age Group"].value_counts().index,autopct="%0.2F%%")
plt.pie([1],radius=0.5,colors="white")
plt.title("Distribution of Passenger Aboard Age Wise",fontsize=10)
plt.legend(loc=1)
plt.show()

plt.figure(figsize=(15,12))
plt.subplot(2,2,1)
sns.countplot(x=df["Sex"],data=df,width=0.5,hue=df["Survived"])
plt.xlabel("Gender Of Passenger",fontsize=10)
plt.ylabel("Count",fontsize=10)
plt.yticks([i for i in range(0,300,50)])
plt.title("Passenger Survival based on Gender",fontsize=12)

plt.subplot(2,2,2)
sns.countplot(x=df["Age Group"],data=df,width=0.5,hue=df["Survived"])
plt.xlabel("Age Group Of Passengers",fontsize=10)
plt.ylabel("Count",fontsize=10)
plt.yticks([i for i in range(0,300,50)])
plt.title("Passenger Survived Based on Age Group")

plt.subplot(2,2,3)
sns.countplot(x=df["Pclass"],data=df,width=0.5,hue=df["Survived"])
plt.xlabel("Passenger Class",fontsize=10)
plt.ylabel("Count",fontsize=10)
plt.title("Passenger Survived According to Passenger class",fontsize=15)
plt.yticks([i for i in range(0,300,50)])

plt.subplot(2,2,4)
sns.countplot(x=df["Embarked"],data=df,width=0.5,hue=df["Survived"])
plt.xlabel("Town Of Passenger",fontsize=10)
plt.ylabel("Count",fontsize=10)
plt.yticks([i for i in range(0,300,50)])
plt.title("Passenger Survived Based on Town")
plt.show()

sns.histplot(x=df["Age"],data=df,kde=True,color="darkblue")
plt.xticks([i for i in range(0,80,5)])
plt.xlabel("Passenger Age",fontsize=10)
plt.title("Distribution of Passenger Age",fontsize=12)
plt.show()

plt.figure(figsize=(10,5))
sns.histplot(x=df["Fare"],data=df,color="darkblue")
plt.xlabel("Fare",fontsize=10)
plt.title("Distribution of Fare",fontsize=12)
plt.show()

plt.figure(figsize=(18,12))
plt.subplot(2,2,1)
sns.countplot(x=df["Pclass"],data=df,width=0.5)
plt.xlabel("Passenger Class",fontsize=10)
plt.ylabel("Count",fontsize=10)
plt.yticks([i for i in range(0,300,50)])
plt.title("Person Aboard According to Passenger class",fontsize=15)

plt.subplot(2,2,2)
sns.countplot(x=df["Pclass"],data=df,width=0.5,hue="Sex")
plt.xlabel("Passenger Class",fontsize=10)
plt.ylabel("Count",fontsize=10)
plt.yticks([i for i in range(0,300,50)])
plt.title("Person According to Passenger class",fontsize=15)

plt.subplot(2,2,3)
sns.countplot(x=df["Pclass"],data=df,width=0.5,hue=df["Survived"])
plt.xlabel("Passenger Class",fontsize=10)
plt.ylabel("Count",fontsize=10)
plt.title("Survival of People According to Passenger class",fontsize=15)
plt.yticks([i for i in range(0,300,50)])

plt.subplot(2,2,4)
sns.countplot(x=df["Pclass"],data=df,width=0.5,hue=df["Embarked"])
plt.xlabel("Passenegr Class",fontsize=10)
plt.ylabel("Count",fontsize=10)
plt.yticks([i for i in range(0,300,50)])
plt.title("Embarked Town According to Passenger class",fontsize=15)
plt.show()

df.drop(["PassengerId","Name","Cabin","Embarked"],axis=1,inplace=True)
df.head(2)

# Label Encoding
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
le.fit(df["Sex"])
df["Sex"]=le.transform(df["Sex"])
le.fit(df["Age Group"])
df["Age Group"]=le.transform(df["Age Group"])

#input-output data
x=df[["Pclass","Sex","Age","SibSp","Parch","Ticket","Fare","Age Group","Family"]]
y=df["Survived"]

y.value_counts()

#Output data balancing
from imblearn.over_sampling import RandomOverSampler
ro=RandomOverSampler()
inp,out=ro.fit_resample(x,y)

out.value_counts()

# DATA SPLIITING
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.metrics import f1_score,recall_score,precision_score,accuracy_score,confusion_matrix
a_train,a_test,b_train,b_test=train_test_split(inp,out,test_size=0.20,random_state=42)

# DECISION TREE CLASSIFIER
from sklearn.tree import DecisionTreeClassifier
dc=DecisionTreeClassifier(criterion="gini",splitter="best")
dc.fit(a_train,b_train)
train=dc.score(a_train,b_train)*100
test=dc.score(a_test,b_test)*100
f1=f1_score(b_test,dc.predict(a_test))*100
recall=recall_score(b_test,dc.predict(a_test))*100
precision=precision_score(b_test,dc.predict(a_test))*100
conf_matrix=confusion_matrix(b_test,dc.predict(a_test))
cv=cross_val_score(DecisionTreeClassifier(criterion="gini",splitter="best"),inp,out,cv=2)*100

print("Decision Tree Classifier: ")
plt.figure(figsize=(8,6))
sns.heatmap(conf_matrix,annot=True,fmt="d",cmap="Blues",xticklabels=["Not Survived","Survived"],yticklabels=["Not Survived","Survived"])
plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.show()
print("Traing Accuracy: ",train)
print("Testing Accuracy: ",test)
print("F1-Score: ",f1)
print("Recall Score: ",recall)
print("Precision Score: ",precision)
print("Cross validation Score: ",min(cv)," & ",max(cv))

# RANDOM FOREST CLASSIFIER
from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier(criterion="gini", n_estimators=100)
rfc.fit(a_train,b_train)
train=rfc.score(a_train,b_train)*100
test=rfc.score(a_test,b_test)*100
f1=f1_score(b_test,rfc.predict(a_test))*100
recall=recall_score(b_test,rfc.predict(a_test))*100
precision=precision_score(b_test,rfc.predict(a_test))*100
conf_matrix=confusion_matrix(b_test,rfc.predict(a_test))
cv=cross_val_score(RandomForestClassifier(criterion="gini", n_estimators=100),inp,out,cv=2)*100

print("Random Forest Classifier: ")
plt.figure(figsize=(8,6))
sns.heatmap(conf_matrix,annot=True,fmt="d",cmap="Blues",xticklabels=["Not Survived","Survived"],yticklabels=["Not Survived","Survived"])
plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.show()
print("Traing Accuracy: ",train)
print("Testing Accuracy: ",test)
print("F1-Score: ",f1)
print("Recall Score: ",recall)
print("Precision Score: ",precision)
print("Cross validation Score: ",min(cv)," & ",max(cv))

# SUPPORT VECTOR CLASSIFIER
from sklearn.svm import SVC
svc=SVC(kernel="rbf")
svc.fit(a_train,b_train)
train=svc.score(a_train,b_train)*100
test=svc.score(a_test,b_test)*100
f1=f1_score(b_test,svc.predict(a_test))*100
recall=recall_score(b_test,svc.predict(a_test))*100
precision=precision_score(b_test,svc.predict(a_test))*100
conf_matrix=confusion_matrix(b_test,svc.predict(a_test))
cv=cross_val_score(SVC(kernel="rbf"),inp,out,cv=2)*100

print("Support Vector Classifier: ")
plt.figure(figsize=(8,6))
sns.heatmap(conf_matrix,annot=True,fmt="d",cmap="Blues",xticklabels=["Not Survived","Survived"],yticklabels=["Not Survived","Survived"])
plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.show()
print("Traing Accuracy: ",train)
print("Testing Accuracy: ",test)
print("F1-Score: ",f1)
print("Recall Score: ",recall)
print("Precision Score: ",precision)
print("Cross validation Score: ",min(cv)," & ",max(cv))

# K NEIGHBOUR CLASSIFIER
from sklearn.neighbors import KNeighborsClassifier
knc=KNeighborsClassifier(n_neighbors=5)
knc.fit(a_train,b_train)
train=knc.score(a_train,b_train)*100
test=knc.score(a_test,b_test)*100

f1=f1_score(b_test,knc.predict(a_test))*100
recall=recall_score(b_test,knc.predict(a_test))*100
precision=precision_score(b_test,knc.predict(a_test))*100
conf_matrix=confusion_matrix(b_test,knc.predict(a_test))
cv=cross_val_score(KNeighborsClassifier(n_neighbors=5),inp,out,cv=2)*100

print("K Neighbour Classifier: ")
plt.figure(figsize=(8,6))
sns.heatmap(conf_matrix,annot=True,fmt="d",cmap="Blues",xticklabels=["Not Survived","Survived"],yticklabels=["Not Survived","Survived"])
plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.show()
print("Traing Accuracy: ",train)
print("Testing Accuracy: ",test)
print("F1-Score: ",f1)
print("Recall Score: ",recall)
print("Precision Score: ",precision)
print("Cross validation Score: ",min(cv)," & ",max(cv))
x_predict =rfc.predict(a_test)
predicted_df = {'predicted_values': x_predict,'original_values': b_test}
print(pd.DataFrame(predicted_df).head(10))
print('')
print('Here O = not survived and 1 = survived')

# FINAL PREDICTION

Pclass=int(input("Enter the passenger Class 1/2/3 : "))
Sex=input("Enter the Gender: ")
Age=float(input("Enter Passenger Age: "))
SibSp=int(input("Enter the Number of Sibling or Spouse: "))
Parch=int(input("Enter the Number of Parent or child: "))
Ticket=int(input("Enter the Ticket Number: "))
Fare=float(input("Enter the Ticket fare: "))
Age_Group=input("Enter the Age Group Child/Teenage/Young/Mid-Age/Senior : ")
Family=int(input("Enter the Number of Family Member on board: "))
user_data=[Pclass,Sex,Age,SibSp,Parch,Ticket,Fare,Age_Group,Family]
feature=le.fit_transform(user_data)
prediction=dc.predict([feature])
if prediction==0:
    print("We are extremely Sorry but Passenger Not Survived")
elif prediction==1:
    print("Congratulation Passenger Survived")
