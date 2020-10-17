# MlQuery

## Functions
* **UserFilePath() :** Takes the file path from the user
* **selectFeatures() :** Takes the features and the target column from the user

### Preprocessing
* **replaceNUL() :** This function either drops the complete column if more than 50% of the data is missing or it would replace it withe either 0, mean, mode or median.
* **featGet(data, features, target) :** returns values of X and y according to the choosen features and target columns.
* **oneHotEncoding(X, columnNumber) :** This function performs one Hot encoding, given columnNumebr.
* **split_dataset(X, y, testSize) :** This function return X_train, X_test, y_train, y_test according
                                      to the given testSize
* **standard(X_train, X_test) :** This function Standardizes X_train, X_test and return them.
* **normalize(X_train, X_test) :** This function Normalizes X_train, X_test and returns them.
### Training Models
* **logReg(X_train, X_test, y_train) :** performs logistice regression and return y_pred.
* **svcModel(X_train, X_test, y_train) :** performs SVC and returns y_pred.
* **knnModel(X_train, X_test, y_train, n) :** performs knn, where n = n_neighbors and return y_pred.
### Accuracy score
* **accuracy(y_pred, y_test) :** performs accuracy prediction and returns the accuracy score.
 ## Sample Input
Enter location of the file: E:/ML PROJECTS/Arya's SInking titanic/SinkingTitanic/train.csv **Takes the file path**

Enter the names of features : Pclass,Sex,Age **Takes the features for X**

Enter the target between '' : Survived **Takes the column for y**

How do u want to replace the null values for feature:  Age **Finds the column withe null value and ask if you want to perform mean, median or mode with missing data**

1. with 0

2.with mean

3.with median

4.with mode

2 **chose option 2**

Do you want to perform One Hot Encoding?[y/n] **option to perform One Hot Encoding**
n

Do you want to perform Label Encoding?[y/n] **Option to perform Label Encoding**
n

Do you want to perform Standardization ?[y/n]y **Option to perform Standardization**

Do you want to perform Normalization ?[y/n]n **Option to perform Normalization**

**option to choose from regression or classification**

Enter 1. for Regression: 

Enter 2. for Classification: 

2

Enter the value for K7

                Logistic Regression  K-nearest neighbours

    Accuracy              0.12037              0.015856
          
          Logistic Regression  K-nearest neighbours

    Accuracy             0.088649              0.109043
          
          Logistic Regression  K-nearest neighbours

    Accuracy             0.169979              0.265189
          
          Logistic Regression  K-nearest neighbours

    Accuracy             0.112783               0.22028
          
          Logistic Regression  K-nearest neighbours

    Accuracy             0.151037              0.029713
          
          Logistic Regression  K-nearest neighbours

    Accuracy             0.060074             -0.051049
          
          Logistic Regression  K-nearest neighbours

    Accuracy             0.060917              0.107806
          
          Logistic Regression  K-nearest neighbours

    Accuracy              0.03062              0.040513
