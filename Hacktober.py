import numpy as numpy
import pandas as pd 
import matplotlib.pyplot as plt

def userFilePath():
    filepath=input("Enter the location of the file: ")
    data=pd.read_csv(filepath,low_memory=True)
    return data

#Ask them features and target if model is Supervised
#Input the name of the features and variable
#to put comma after each feature
def selectFeatures():
    x = input("Enter the names of features : ")
    features=x.split(',')
    target = input("Enter the target between '' : ")
    return features, target

def replaceNull(data, features):
    #to count the number of null values in each column
    count_null=data[data.columns].isna().sum()
    count_null=dict(count_null)
    rows=len(data.index)
    for i in features:
        k=count_null[i]
        if k!=0:
            if k>(rows/2):
                data.drop([i], axis = 1)
            else:
                print('How do u want to replace the null values for feature: ',i)
                print('1. with 0\n2.with mean\n3.with median\n4.with mode')
                ch=int(input())
                if ch==1:
                    data[i] = data[i].fillna(0)
                elif ch==2:
                    mean=data[i].mean()
                    data[i] = data[i].fillna(mean)
                elif ch==3:
                    med=data[i].median()
                    data[i] = data[i].fillna(med)
                elif ch==2:
                    mode=data[i].mode()
                    data[i] = data[i].fillna(mode)
    return data

#Function to load the features into the X and y respectively and return it
def featget(data, features, target):
    X = data.loc[:,features]
    print(target)
    y = data.loc[:,target]     
    return (X,y)

#Pre-processing

#OneHotEncoding
#Function that returns X after performing oneHotEncoding
# columnNumber --> takes the columnNumber for which one hot encoding is to be done
def oneHotEncoding(X, columnNumber):
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder
    ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [columnNumber])], remainder='passthrough')
    X = np.array(ct.fit_transform(X))
    return X

#LabelEncoder
# X--> takes only a single column
def labelEncodingColumn(X):
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    X = le.fit_transform(X)
    return X

#traintestsplit
def split_dataset(X, y, testSize):
    from sklearn.model_selection import train_test_split
    # testSize--> defines the size of the test from the dataset(takes decimal less than 1)
    # randomState --> takes integer input
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = testSize)
    return (X_train, X_test, y_train, y_test)

#Standardization
def standard(X_train,X_test):
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    return (X_train, X_test)

#Normalisation
def normalize(X_train,X_test):
    from sklearn.preprocessing import Normalizer
    scaler = Normalizer().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    return (X_train, X_test)

#Methods to predict the values using ML values

#Logistic Regression
def logReg(X_train,X_test, y_train):
    from sklearn.linear_model import LogisticRegression
    logr = LogisticRegression()
    logr.fit(X_train,y_train)
    y_pred = logr.predict(X_test)
    return y_pred

#SV Regression
def svcModel(X_train,X_test, y_train):
    from sklearn.svm import SVC
    svc = SVC(kernel='linear')
    svc.fit(X_train, y_train)
    y_pred = svc.predict(X_test)
    return y_pred

#KNN
def knnModel(X_train,X_test,y_train,n):
    from sklearn import neighbors
    knn = neighbors.KNeighborsClassifier(n_neighbors=n, metric = 'minkowski', p=2)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    return y_pred

#To find the accuracy of your model.
def accuracy(y_pred,y_test):
    from sklearn.metrics import r2_score
    return r2_score(y_pred, y_test)

filepath = input('Enter location of the file: ')
df = pd.read_csv(filepath, low_memory=True)
#replacing null values
features, target = selectFeatures()
replaceNull(df, features)

X,y = featget(df,features, target)

print("Do you want to perform One Hot Encoding?[y/n]")
if(input() == 'y'):
    X = oneHotEncoding(X, input('Enter the Column number : '))

print('Do you want to perform Label Encoding?[y/n]')
if(input() == 'y'):
    col_num = input('Enter the column name : ')
    X[col_num] = labelEncodingColumn(X[col_num])
    
standardizationYN = input("Do you want to perform Standardization ?[y/n]")
normalizationYN = input("Do you want to perform Normalization ?[y/n]")

#after this we have to call the train test split and other prediction models
test_size = 0.1
while(test_size <= 0.45):
    X_train,X_test,y_train,y_test = split_dataset(X, y, test_size, 0)
    if(standardizationYN == 'y'):
        X_train, X_test = standard(X_train,X_test)
    
    if(normalizationYN == 'y'):
        X_train, X_test = normalize(X_train,X_test)
    test_size+=0.05
    #Now when the accuracy calculation function is done 
    #call all the ML model inside them and then print their respective accuracy


print("1. for Regression: ")
print("2. for Classification: ")
c=int(input("Enter: "))
if c==1:
    y_pred1=linearReg(X_train,X_test, y_train)
    p=accuracy(y_pred1,y_test)
    y_pred2=svcModel(X_train,X_test, y_train)
    q=accuracy(y_pred2,y_test)
    acc={'Linear Regression':p,'Support Vector Regression':q}
    table = pd.DataFrame(acc, columns = ['Linear Regression','Support Vector Regression'], index=['Accuracy'])
    print(table)
    print("1. for Linear Regression: ")
    print(" 2. for Support Vector Regression: ")
    result=int(input("Enter: "))
    if result==1:
        print("Prediction for Linear Regression model: ")
        output = pd.DataFrame({target: y_pred1})
        print(output,'\n')
    else:
        print("\nPrediction for Support Vector Regression model: ")
        output = pd.DataFrame({target: y_pred2})
        print(output,'\n')
elif c==2:  
    y_pred1=logReg(X_train,X_test, y_train)
    p=accuracy(y_pred1,y_test)
    n=int(input("\nEnter the value for K"))
    y_pred2=knnModel(X_train,X_test, y_train, n)
    q=accuracy(y_pred2,y_test)
    acc={'Logistic Regression':p,'K-nearest neighbours':q}
    table = pd.DataFrame(acc, columns = ['Logistic Regression','K-nearest neighbours'], index=['Accuracy'])
    print(table)
    print("\n1. for Logistic Regression: ")
    print("\n2. for K-nearest neighbours: ")
    result=int(input("Enter: "))
    if p>q:
        print("Prediction for Logistic Regression model: ")
        output = pd.DataFrame({target: y_pred1})
        print(output,'\n')
    else:
        print("\nPrediction for K-nearest neighbours model: ")
        output = pd.DataFrame({target: y_pred2})
        print(output,'\n')
