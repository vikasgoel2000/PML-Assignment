---
title: "Practical Machine Learning- Assignment"
---

#### Executive Summary:
The goal of this exercise is to build a prediction model to automatically predict the how people perform some predetermined types of exercises. The model is built with the help of  a **training data set** , cross validated with a **testing data set** and finally applied once to a 3rd **test data set**. The model built has a very good **accuracy of 99%** and was successfully applied to the test data set.

#### I. load libraries and data
  
```{r,warning=FALSE,message=FALSE,options(cache = TRUE)}
library(caret)
library(ggplot2)
library(randomForest)
library(e1071)
set.seed(2637)
setwd("C:/Users/Vikas-goyal/Downloads")
training <- read.csv("pml-training.csv", na.strings=c("", "NA", "NULL"))
testing <- read.csv("pml-testing.csv", na.strings=c("", "NA", "NULL"))
```


####II.  Exploratory Data Analysis - streamlining Data 

The first step is to perform some data exploration , clean the data sets and extract information on data types, variables  and so on.  
It was noted that  
- The dependent variable (**Classe**) is already a *factor* with 5 levels  
- There seems to be a lot of non available data (**NA**)  

The next step is to clean and prepare the data sets 


##### 1. Removing columns with "NA" and irrelevant variables (Column 1:7)


```{r,warning=FALSE, message=FALSE}

data_training <- training[ , colSums(is.na(training)) == 0]  
data_training<- data_training[,-(1:7)]

data_testing <- testing[ , colSums(is.na(testing)) == 0]  
data_testing<- data_testing[,-(1:7)]
                    
```

##### 2. Make the nearzero diagnosis and remove


```{r,warning=FALSE, message=FALSE}

nzv_train<- nearZeroVar(data_training, saveMetrics = TRUE)
nzv_train[nzv_train$nzv, ]
                    
```

This diagnosis proposed **no candidate**(variable) for removal.  

The PCA analysis was also performed but later removed from this assignment.  
There was one unknown problem which was taking too much time to solving.

##### 3. Remove highly correlated variables

```{r,warning=FALSE,  message=FALSE}

cor_train <- findCorrelation(cor(data_training[,-53]), cutoff = 0.8)
data_training<-data_training[,-cor_train]
data_testing<-data_testing[,-cor_train]

```


####III.  Split the Data set

The training data set **(data_training)** set is split in 2 :  
 - one set will be used for training the model  
 - the 2nd set will be used to crossvalidate the model

A default 70/30 split will be performed.


```{r, message=FALSE, results='hide'}
# split the training data sets into 2 on a 70/300/30 basis
#
Sp_Index <- createDataPartition(y=data_training$classe, p=0.7, list=FALSE)   
training_1<- data_training[Sp_Index,]     # Put the 70% in this new data set.
testing_1<- data_training[-Sp_Index,]    # Put the remaining 30% in this new data set for cross validation.

```


####IV.  Apply predicting with Random Forest

The **CARET** package was used to perform this *Random Forest* processing.

```{r,warning=FALSE,  message=FALSE}

modfit <- randomForest(classe ~ ., data = training_1, importance = TRUE, ntrees = 200)
varImpPlot(modfit, cex = 0.7)                         
                    
```

\linebreak \newline

**Gini** is defined as *"inequity"* or a measure of *"node impurity"* in tree-based classification.  
A **low Gini** (i.e. higher decrease in Gini) means that a particular predictor variable plays a **greater** role in partitioning the data into the defined classes.
There is a clear visualisation of the **"important"** variables.


####V.  Cross validation and Accuracy testing

```{r,warning=FALSE,  message=FALSE}
crossval<-predict(modfit, training_1)
confusionMatrix(training_1$classe, predict(modfit, training_1))

```

```{r,warning=FALSE,  message=FALSE}

confusionMatrix(testing_1$classe, predict(modfit, testing_1))

```
We apply the **confusion matrix** to both the *training* and *test* set and check the **accuracy** of our model.

We get :  
 - Training set : **100% accurate** as expected  
 - Testing set : **>99% accurate**

The model has a good accuracy and can be confidently used.


####VI.  The 20 files

 
 
```{r,warning=FALSE,  message=FALSE}

Result <- predict(modfit,data_testing)
answers<-Result[1:20]
answers

pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}

pml_write_files(answers)

```

Here are the results for 20 files aswell.

#### Vikas Goyal