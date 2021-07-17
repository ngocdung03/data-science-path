rm(list = ls())
memory.limit(size=20000)
library(tidyverse)
library(lubridate)
library(caret)
library(survival)
library(Hmisc)
#library(rms)
data <- read.csv("C:/Users/ngocdung/Dropbox/NCC/DII - perio/[Server]Perio/data/data.csv", stringsAsFactors=F)
urban_raw <- read.csv("C:/Users/ngocdung/Dropbox/NCC/KoGES_DATASET (1)/KoGES_DATASET/city_rawdata.csv", stringsAsFactors=F)
data <- as.data.frame(cbind(data,DS1_DM = urban_raw$DS1_DM))
#### Preprocessing ####
# data already preprocessed

#### Analysis ####
one_hot <- function(vect) {
  #vect[which(vect=='.')] <- NA
  return(as.factor(vect))         #asnumeric?
}
str_cols <- c('sex', 'age_c', 'study', 'Smoke', 'Alcoholgp', 'meno', 'Active', 'BMI_C', 'DS1_DM',
              'DS1_PER','DS1_EDU','DS1_MARRY_A','DS1_MARRY','DS1_INCOME','DS1_SMOKE','DS1_SMOKE_100','DS1_DRINK','DS1_EXER','DS1_MENYN','DS1_MENYN_A','DS1_HEIGHT','DS1_WEIGHT','DS1_DM')
converted <- as.data.frame(
  apply(data[which(colnames(data) %in% str_cols)],2, as.factor))
data2 <- cbind(data[which(!colnames(data) %in% str_cols)],
              converted)
data2$sex <- as.factor(data2$sex)
data2$age_c <- as.factor(data2$age_c)
data2$study <- as.factor(data2$study)
data2$Smoke <- as.factor(data2$Smoke)
data2$Alcoholgp <- as.factor(data2$Alcoholgp)
data2$DS1_DM <- as.factor(data2$DS1_DM)
data2$Active <- as.factor(data2$Active)

# Exclude 8,148 cases at baseline
ds <- data2 %>%
  filter(base_case != 1)

# Excluded outliers male: <800 or >4500; female: <500 or >4200 (1,545; 195; 2,371; 553)
ds <- ds %>%
  filter((sex == 1 & energy >= 800 & energy <=4500)|     #NO NA?
          (sex == 2 & energy >= 500 & energy <=4200))

# [REMOVE LATER] Excluded missing value for DII and time variable (month) (134; 100,163)
ds2 <- ds %>% 
  filter(!is.na(DIINORMAL) & !is.na(month_per))
ds3 <- ds %>%     #134
  filter(!is.na(DIINORMAL))
# Flowchart

# Divide dataset by sex
DS_M <-  ds2 %>% filter(sex==1)
DS_W <-  ds2 %>% filter(sex==2)

# Divide dataset by smoking status
DS_NS <- ds2 %>% filter(DS1_SMOKE==1)
DS_XS <- ds2 %>% filter(DS1_SMOKE==2)
DS_S <- ds2 %>% filter(DS1_SMOKE==3)

# Divide dataset by diabetes
DS_ND <- ds2 %>% filter(DS1_DM=="1")
DS_D <- ds2 %>% filter(DS1_DM=="2")

# Table 1: Count samples by quantiles
quan_cal <- function(df,list) {
  q25 <- unname(quantile(df$DIINORMAL, 0.25))
  q50 <- unname(quantile(df$DIINORMAL, 0.50))
  q75 <- unname(quantile(df$DIINORMAL, 0.75))
  result = t(data.frame(quan = c(1,2,3,4)))
  for (i in na.omit(unique(list))) { 
    d = df %>% filter(list == i) %>%
      mutate(quan = case_when(
        DIINORMAL < q25 ~ 0,
        DIINORMAL < q50 & DIINORMAL >= q25 ~ 1,
        DIINORMAL < q75 & DIINORMAL >= q50 ~ 2,
        DIINORMAL >= q75 ~ 3,
      )) %>% 
      count(quan) %>% t(.)
    rownames(d)[2] <-  paste("Category",i, sep="")
    result <- rbind(d,result)
  }
  result <- result[which(rownames(result) != "quan"),]
  list=df
  return(result)
}
quan_cal(ds2, ds2$sex)
quan_cal(ds2, ds2$age_c)
quan_cal(ds2, ds2$study)
quan_cal(ds2, ds2$Smoke)
quan_cal(ds2, ds2$Alcoholgp)
quan_cal(ds2, ds2$meno)
quan_cal(ds2, ds2$BMI_C)
quan_cal(ds2, ds2$DS1_DM)

# Table 2: Survival analysis
surv_dii <- function(df) {
  q25 <- unname(quantile(df$DIINORMAL, 0.25))
  q50 <- unname(quantile(df$DIINORMAL, 0.50))
  q75 <- unname(quantile(df$DIINORMAL, 0.75))
  df <- df %>% 
  mutate(quan = as.factor(case_when(
    DIINORMAL < q25 ~ 0,
    DIINORMAL < q50 & DIINORMAL >= q25 ~ 1,
    DIINORMAL < q75 & DIINORMAL >= q50 ~ 2,
    DIINORMAL >= q75 ~ 3
  )))
  return(summary(coxph(Surv(month_per, FU1_case==1)~quan,data=df)))}
surv_dii(DS_M)
surv_dii(DS_W)

surv_dii(DS_NS)  #?WHY error
surv_dii(DS_XS)
surv_dii(DS_S)

surv_dii(DS_ND)
surv_dii(DS_D)

# Table 5 nutrients
quan_crude_adjust <- function(df, var){    
  q25 <- unname(quantile(var, 0.25))   
  q50 <- unname(quantile(var, 0.50))
  q75 <- unname(quantile(var, 0.75))
  d <-  df %>% mutate(quan = as.factor(case_when(
    var < q25 ~ 0,
    var < q50 & var >= q25 ~ 1,
    var < q75 & var >= q50 ~ 2,
    var >= q75 ~ 3)))
  return(#c(quantile(var),
    # summary(coxph(Surv(month_per, FU1_case==1)~quan,data=d)),
    summary(coxph(Surv(month_per, FU1_case==1)~quan+sex+age_c+study+Smoke+Alcoholgp+meno+BMI_C+DS1_DM+Active,data=d)))
  #)  
}
quan_crude_adjust(ds3, ds3$vitamin_B12)

##### Main analysis #####
rownames(ds) <- ds$RID
num <- ds %>%       # is.na(DIINORMAL) and is.na(month_per) were retained
  dplyr::select(c(RID,
                  vitamin_B12, vitamin_B6, carotene, caffeine, carbohydrate, cholesterol, total_fat, fiber, folic_acid,  # only 36?      
                  garlic, fe, mg, MUFA, niacin, n_3_fatty_acides, n_6_fatty_acides, onion, protein, PUFA,            
                  riboflavin, saturated_fat, se, thiamin, trans_fat, vitamin_A, vitamin_C, vitamin_D, vitamin_E, Zn,              
                  Green_black_tea, Flavan_3_ol,  Flavones, Flavonols, Flavonones, Anthocyanidins, Isoflavones, FU1_case, month_per))  #60234, ?compare result when month_per NA <- 0

# Describe
glimpse(num)

# Exclude missing - 134 cases
num <- na.omit(num)

# log transformation

# Exclude zero variance feature
head(num[ , which(apply(num, 2, var) == 0)])  #None

# Creating train and test set  ? train:test:validation  70:15:15
set.seed(1)
train.index <- createDataPartition(num$FU1_case, p = .8, list = FALSE)   #error when unique(as.numeric(raw$thyroid))
#train.index[1] == dat5$thyroid
train <- num[train.index,] %>% #128425
  mutate(index = "train")
test <- num[-train.index,] %>%
  mutate(index = "test") #32106

#PCA
# Scaling vs. normalization: https://www.kaggle.com/rtatman/data-cleaning-challenge-scale-and-normalize-data#Get-our-environment-set-up
# Scaling: when you're using methods based on measures of how far apart data points, like support vector machines, or SVM or k-nearest neighbors, or KNN.
# Normalization: if you're going to be using a machine learning or statistics technique that assumes your data is normally distributed (eg. t-tests, ANOVAs, linear regression, linear discriminant analysis (LDA) and Gaussian naive Bayes).
pca <- prcomp(train[,2:37], scale=T)   # 3:182 * scale=T,  cor=? #Scaling, compare with the result ..manual scale 

#pca <- prcomp(train_obs[,20:151])   #* scale=T,  cor=?

#####
# # PCRegression
# #add a training set with principal components
# train0 <- data.frame(FU1_case = train$FU1_case, month_per = train$month_per, pca$x)
# 
# #we are interested in first 5 PCAs
# train1 <- train0[,1:4]      #?number of PCs for tuning later 
# cox.model <- coxph(Surv(month_per, FU1_case==1)~., data=train1)
# cox.model
# 
# #transform test into PCA
# test0 <- predict(pca, newdata = test)
# test0 <- as.data.frame(test0)
# test1 <- data.frame(FU1_case = test$FU1_case, month_per = test$month_per, test0)
# 
# #select the first 5 components?
# test2 <- test0[,1:5]     #!!!remember to remove outcome
# 
# ## Make prediction on test data and calculate c-index:
# # Compare the c-indices of rcorr.cens and summary(cox.model)
# # https://stats.stackexchange.com/questions/254375/concordance-index-in-survival-and-rms-packages
# train_obj <- Surv(time = train1$month_per, event = train1$FU1_case)
# train_validation <-  predict(cox.model, newdata = train1[,3:7])
# # predict() is a part of hazard function, while train_obj is a survival object. 
# # Survival and hazard have opposite directions, therefore, train_validation is multiplied by (-1)
# rcorr.cens(x=train_validation*-1, S=train_obj) 
# summary(cox.model)  #concordance(cox.model)
# 
# # Create survival estimates on validation data
# pred_validation <-  predict(cox.model, newdata = test2)
# 
# # c-index
# rcorr.cens(x=pred_validation*-1, S=Surv(time = test1$month_per, event = test1$FU1_case))['C Index']

#### Tuning and evaluation####
# ? Compare accuracy between total DII model and PCR
#https://www.biostars.org/p/109215/
#https://cran.r-project.org/web/packages/survival/vignettes/concordance.pdf 
which(colnames(local_raw) == "AS1_TRTOTH1NA")
cov_train <- ds[which(ds$RID %in% train$RID),c('sex','age_c','Smoke', 'Alcoholgp', 'DS1_DM', 'Active')]
cov_test <- ds[which(ds$RID %in% test$RID),c('sex','age_c','Smoke', 'Alcoholgp', 'DS1_DM', 'Active')] #?check if indices of these 2 sets match train/test set
pcs_test <- as.data.frame(predict(pca, newdata = test))

c_pca <- function(n_factor){
  train_data <- data.frame(pca$x[,1:n_factor], 
                           FU1_case = train$FU1_case, 
                           month_per = train$month_per, 
                           cov_train)
  model <- coxph(Surv(month_per, FU1_case==1)~., data=train_data)
  test_data <- data.frame(pcs_test[,1:n_factor],
                          FU1_case = test$FU1_case, 
                          month_per = test$month_per, 
                          cov_test)
  pred_val <-  predict(model, newdata = test_data)
  c_index <- rcorr.cens(x=pred_val*-1, S=Surv(time = test$month_per, event = test$FU1_case))['C Index']
  return(c_index)
}

c_indx <- sapply(2:36, c_pca)  #Error with value 1: 'data' must be a data.frame, environment, or list
plot(2:36, c_indx, type="l")

c_indx

# Export PCA data
setwd("C:/Users/ngocdung/Dropbox/NCC/DII - perio/Machine learning analysis/PCA dataset")
data.frame(pcs$x, FU1_case = train$FU1_case, month_per = train$month_per, cov_train) %>% 
  write.csv('train_pca.csv')

data.frame(pcs_test, FU1_case = test$FU1_case, month_per = test$month_per, cov_test) %>% 
  write.csv('test_pca.csv')

#### Random forest ####
library(randomForest)
library(randomForestSRC)
#train_rf <- train(Surv(month_per, FU1_case==1) ~., method = "rpart", data = train)
train_data_r <- data.frame(pca$x[,1:20], month_per=train$month_per, FU1_case=train$FU1_case) %>% 
  filter(!is.na(month_per)&month_per>0)

set.seed(1)
train_rf <- rfsrc(Surv(month_per, FU1_case) ~ ., data=train_data_r)

y_hat5 <- predict(train_rf, newdata=pcs_test[,1:20])   #class and prob will be error because they are only meant for classification trees. 
rcorr.cens(x=-5*y_hat5$predicted^2, S=Surv(time = test$month_per, event = test$FU1_case))

#Function for calculating c-index####
mycindex<-function(days,status,preds){
  permissible<-1
  concordance<-1
  endind=length(preds)-1
  for (i in seq(1,endind)){
    tmp=i+1
    for (j in seq(tmp,length(preds))){
      
      if((days[i]==days[j]) & (status[i]==0) & (status[j]==0)){ next } 
      if((days[i]<days[j]) & (status[i]==0) ){ next } 
      if((days[j]<days[i]) & (status[j]==0) ){ next } 
      
      permissible<-permissible+1
      
      if (status[i]==1 & status[j]==1 &  
          preds[i]>preds[j] & (days[i]>days[j])){
        concordance<-concordance+1
        
        #com_value<-c(concordance,preds[i], preds[j], days[i], days[j])
        #print (com_value)
      }
      if (status[i]==1 & status[j]==1 &  
          preds[i]<preds[j] & days[i]<days[j]){
        concordance<-concordance+1
        
        com_value<-c(concordance,preds[i], preds[j], days[i], days[j])
        print (com_value)
      }
      
      if((days[i]==days[j]) & (status[i]==1) & (status[j]==1) & (preds[i]!=preds[j]))
      {
        concordance<-concordance+0.5
        
      }
      if((days[i]==days[j]) & (status[i]==1) & (status[j]==0) &  (preds[i]<preds[j]))
      {
        concordance<-concordance+1
        
      }
      if((days[i]==days[j]) & (status[i]==0) & (status[j]==1) & (preds[i]>preds[j]))
      {
        concordance<-concordance+1
        
      }
      if((days[i]==days[j]) & (status[i]==1) & (status[j]==0) & (preds[i]>=preds[j]))
      {
        concordance<-concordance+0.5
        
      }
      if((days[i]==days[j]) & (status[i]==0) & (status[j]==1) & (preds[i]<=preds[j])) {
        concordance<-concordance+0.5
        
      }
    }
    
  }
  cindex<- concordance/permissible
  myout<-c(concordance,permissible,cindex)
  return (cindex)
}

mycindex(test$month_per, test$FU1_case, y_hat$predicted)

#### Plots ####
glimpse(pca)
pca$x[1:6, 1:6]
pca$sdev

# get variance of each PC
pr_var  <- (pca$sdev)^2
# compute variance explained by each PC
prop_varex <- pr_var/sum(pr_var)
prop_varex
# Bring a bar plot - quick plot for vector object
barplot(prop_varex, names.arg = colnames(pca$x),
        xlab = "Principal Component",
        ylab = "Proportion of Variance Explained")

# For the cumulative variance explained:
plot(cumsum(prop_varex), 
     xlab = "Principal Component",
     ylab = "Cumulative Proportion of Variance Explained",
     type = "b")

# Plot the reduction of data variance to 2 first components
# install.packages("ggrepel")
library(ggrepel)

# To see that the principal components are actually capturing something important
pcs <- data.frame(pca$rotation, name = colnames(num[,1:36]))
# pcs %>% data.frame(.,
#                    Sample = rownames(pcs),
#                    group = train_obs[match(rownames(pca$x), train_obs$cancer.center.No.) , "group"]) %>%

## TT ref
pcs%>% ggplot(aes(PC1, PC2,label=name)) + 
  geom_point() +
  geom_text_repel()#aes(PC1, PC2),		# The first principle component PC1 shows the difference between critically acclaimed movies on one side.
#data = filter(pcs, 			#PC2: artsy independent films vs. nerd favorites
#              PC1 < -0.1 | PC1 > 0.1 | PC2 < -0.075 | PC2 > 0.1))  #CHECK

pcs%>% ggplot(aes(PC3, PC4,label=name)) + 
  geom_point() +
  geom_text_repel()
#install.packages("factoextra")
library(factoextra)
var <- get_pca_var(pca)
var

#install.packages("corrplot")
library("corrplot")
#corrplot(var$cos2, is.corr=FALSE)  #TOO many features

#install.packages("gridExtra")
library(gridExtra)
p1 <- fviz_contrib(pca, choice="var", axes=1, fill="pink", color="grey", top=10)
p2 <- fviz_contrib(pca, choice="var", axes=2, fill="skyblue", color="grey", top=10)
grid.arrange(p1,p2,ncol=2)
######################


##################################################
## Selecting components 
#https://www.analyticsvidhya.com/blog/2016/03/pca-practical-guide-principal-component-analysis-python/

#add a training set with principal components
train0 <- data.frame(thyroid = train$thyroid, survtime = train$survtime, pca$x)

#we are interested in first 35 PCAs

train1 <- train0[,1:37]

#run model

library(survival)
cox.model <- coxph(Surv(train$survtime, train$thyroid==1)~.,data=train1)
cox.model

#transform test into PCA
test0 <- predict(pca, newdata = test)
test0 <- as.data.frame(test0)
test1 <- data.frame(thyroid = test$thyroid, survtime = test$survtime, test0)

#select the first 30 components
test2 <- test1[,1:37]

#make prediction on test data
cox.prediction <- predict(cox.model, 
                          test2, 
                          type = "response")           #check

fit <- (glm(thyroid ~ ., data = train1[,-2], family = "binomial"))
prediction <- predict(demo, test2[,-2], type ="response")                                                    #The type="response" option tells R to output probabilities of the form P(Y = 1|X)

# Treating with missing value https://stackoverflow.com/questions/19871043/r-package-caret-confusionmatrix-with-missing-categories
u <- union(round(prediction), test$thyroid)                                                               #Why need to be rounded?
t <- table(factor(round(prediction), u), factor(test$thyroid, u))
#Test set
confusionMatrix(t)$overall["Accuracy"]

train_pre <- predict(demo, train1[,-2], type ="response")
tr <- table(round(train_pre), train$thyroid)
confusionMatrix(tr)$overall["Accuracy"]


# SEARCH GOOGLE!!!
# https://stackoverflow.com/questions/50697925/extimate-prediction-accuracy-of-cox-ph
# https://stats.stackexchange.com/questions/79362/how-to-get-predictions-in-terms-of-survival-time-from-a-cox-ph-model
# https://stackoverflow.com/questions/27228256/calculate-the-survival-prediction-using-cox-proportional-hazard-model-in-r

confusionMatrix(cox.prediction, (test$thyroid))$overall["Accuracy"]

confusionMatrix(y_hat, test_set$sex)$overall["Accuracy"]

#For fun, finally check your score of leaderboard
sample <- read.csv("SampleSubmission_TmnO39y.csv")
final.sub <- data.frame(Item_Identifier = sample$Item_Identifier, Outlet_Identifier = sample$Outlet_Identifier, Item_Outlet_Sales = rpart.prediction)

write.csv(final.sub, "pca.csv",row.names = F)

## Make prediction on test data, type of predicted value:
# linear predictor ("lp")
# risk score exp(lp) ("risk")
# expected number of events given the covariates and follow-up time ("expected")
# terms of the linear predictor ("terms"). 
# survival probability for a subject is equal to exp(-expected)

#test3 <- na.omit(test2)
# cox.prediction <- predict(cox.model, 
#                           test2,
#                           type="all")

screeplot(pca, npcs = min(10, length(pca$sdev)),
          +           type = c("barplot", "lines"),
          +           main = deparse(substitute(pca)))
