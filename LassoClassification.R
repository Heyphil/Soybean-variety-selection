library (glmnet)
library(readxl)


# read imagefeatures from the PT 2018
data.PT = read_excel('imagefeature_38_PT.xlsx')
data.PT = as.data.frame(data.PT)
sum(data.PT$select)

data = data.PT

set.seed(1)
trainSamples=sample (1:nrow(data), size=.80*nrow(data))


### Classification for PT

# get image features day by day
data.single = data[,c(4:42)]   # 0812 PT
data.single = data[,c(4,43:80)] # 0827 PT
data.single = data[,c(4,81:118)] # 0914 PT
data.single = data[,c(4,119:156)] # 0927 PT

# seperate the training and testing set
train.set = data.single[trainSamples,]
test.set = data.single[-trainSamples,]

x=model.matrix(select~.,train.set)[,-1]
y=train.set$select

# cross validation for selecting lambda in the LASSO model 
set.seed(1)
lasso.fit = cv.glmnet(x,y,alpha = 1,family = "gaussian",standardize = TRUE)
bestlam=lasso.fit$lambda.min

# make predictions on the testing set
test.pred = predict(lasso.fit,s=bestlam, model.matrix(select~.,test.set)[,-1])
thres = quantile(test.pred,probs = 0.5) # mean of the probabilities as the threshold

# results
lasso.pred = rep(0,nrow(test.set))
lasso.pred[test.pred>thres] = 1
lasso.tb = table(lasso.pred,test.set$select)
TP = lasso.tb[2,2]
FN = lasso.tb[1,2]
TN = lasso.tb[1,1]
FP = lasso.tb[2,1]

Rec = TP/(FN+TP)
Prec = TP/(TP+FP)
F1 = 2*Prec*Rec/(Prec+Rec)

Rec
F1


# read imagefeatures from the PYT 2019
data.PYT = read_excel('imagefeature_38_PYT.xlsx')
data.PYT = as.data.frame(data.PYT)
sum(data.PYT$select)

data = data.PYT

set.seed(1)
trainSamples=sample (1:nrow(data), size=.80*nrow(data))


### Classification for PYT

# get image features day by day
data.single = data[,c(4:42)]   # 0718 PYT
data.single = data[,c(4,43:80)] # 0730 PYT
data.single = data[,c(4,81:118)] # 0816 PYT
data.single = data[,c(4,119:156)] # 0829 PYT
data.single = data[,c(4,157:194)] # 0920 PYT
data.single = data[,c(4,195:232)] # 1001 PYT

# seperate the training and testing set
train.set = data.single[trainSamples,]
test.set = data.single[-trainSamples,]

x=model.matrix(select~.,train.set)[,-1]
y=train.set$select

# cross validation for selecting lambda in the LASSO model 
set.seed(1)
lasso.fit = cv.glmnet(x,y,alpha = 1,family = "gaussian",standardize = TRUE)
bestlam=lasso.fit$lambda.min

# make predictions on the testing set
test.pred = predict(lasso.fit,s=bestlam, model.matrix(select~.,test.set)[,-1])
thres = quantile(test.pred,probs = 0.5) # mean of the probabilities as the threshold

# results
lasso.pred = rep(0,nrow(test.set))
lasso.pred[test.pred>thres] = 1
lasso.tb = table(lasso.pred,test.set$select)
TP = lasso.tb[2,2]
FN = lasso.tb[1,2]
TN = lasso.tb[1,1]
FP = lasso.tb[2,1]

Rec = TP/(FN+TP)
Prec = TP/(TP+FP)
F1 = 2*Prec*Rec/(Prec+Rec)

Rec
F1


