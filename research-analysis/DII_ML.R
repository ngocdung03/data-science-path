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
#Function for calculating c-index####
harrell_c <- function(y_true, scores, event){   #? catch error if lengths are not equal
  n <- length(y_true)
  #assert (len(scores) == n and len(event) == n)
  concordant <- 0
  permissible <- 0
  ties <- 0
  result <- 0
  for(i in (1:(n-1))){        
    for(j in ((i+1): n)){
      if (event[i] != 0 | event[j] != 0){
        if (event[i] == 1 & event[j] == 1){
          permissible <-  permissible + 1
          if(y_true[i] == y_true[j]){
            ties <- ties+1}
          else if (y_true[i] > y_true[j] & scores[i] < scores[j]){
            concordant <-  concordant+1}
          else if (y_true[i] < y_true[j] & scores[i] > scores[j]){
            concordant <-  concordant+1}
        }
        else if (event[i] == 0 | event[j] == 0){
          censored <-  j
          uncensored <-  i
          if(event[i] == 0){
            censored <- i
            uncensored <- j}
          if(y_true[censored] >= y_true[uncensored]){
            permissible <- permissible+1
            if(scores[i] == scores[j]){
              ties = ties+ 1}
            if(y_true[censored] >= y_true[uncensored] & scores[censored] < scores[uncensored]){
              concordant <- concordant+1}}
        }
      }
    }
  }
  result <- (concordant + 0.5 * ties) / permissible
  return(result)
}

#Dataset preparation####
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

# PCA ####
# Scaling vs. normalization: https://www.kaggle.com/rtatman/data-cleaning-challenge-scale-and-normalize-data#Get-our-environment-set-up
# Scaling: when you're using methods based on measures of how far apart data points, like support vector machines, or SVM or k-nearest neighbors, or KNN.
# Normalization: if you're going to be using a machine learning or statistics technique that assumes your data is normally distributed (eg. t-tests, ANOVAs, linear regression, linear discriminant analysis (LDA) and Gaussian naive Bayes).
pca <- prcomp(train[,2:37], scale=T)   # 3:182 * scale=T,  cor=? #Scaling, compare with the result ..manual scale 

#pca <- prcomp(train_obs[,20:151])   #* scale=T,  cor=?

# Export PCA data
setwd("C:/Users/ngocdung/Dropbox/NCC/DII - perio/Machine learning analysis/PCA dataset")
pcs <- data.frame(pca$rotation, name = colnames(num[,1:36]))
which(colnames(local_raw) == "AS1_TRTOTH1NA")
cov_train <- ds[which(ds$RID %in% train$RID),c('sex','age_c','Smoke', 'Alcoholgp', 'DS1_DM', 'Active')]
pcs_train <- as.data.frame(predict(pca, newdata = train))      #?CHECK
cov_test <- ds[which(ds$RID %in% test$RID),c('sex','age_c','Smoke', 'Alcoholgp', 'DS1_DM', 'Active')] #?check if indices of these 2 sets match train/test set
pcs_test <- as.data.frame(predict(pca, newdata = test))

train_pca <- data.frame(cov_train, pcs_train, FU1_case = train$FU1_case, month_per = train$month_per) %>%
  na.omit(.)
# train_pca %>% 
#   write.csv('train_pca.csv')
# train_pca <- read.csv('train_pca.csv', stringsAsFactors = F)

test_pca <- data.frame(cov_test, pcs_test, FU1_case = test$FU1_case, month_per = test$month_per) %>%
  na.omit(.)
# test_pca %>% 
#   write.csv('test_pca.csv')

# Random forest ####
library(randomForestSRC)
# Random Forest
train_data <- data.frame(num[train.index,], month_per=train$month_per, FU1_case=train$FU1_case) %>%
  filter(!is.na(month_per)&month_per>0)
set.seed(1)
train_rf <- rfsrc(Surv(month_per, FU1_case) ~ ., data=train_data[,2:39]) 
y_hat <- predict(train_rf, newdata=test[,2:39])   #remove RID and index
harrell_c(test$month_per, y_hat$predicted, test$FU1_case)


# Random Forest + PCA
# train_data_r <- data.frame(pca$x, month_per=train$month_per, FU1_case=train$FU1_case) %>%   #pca$x[,1:20]
#   filter(!is.na(month_per)&month_per>0)

set.seed(1)
train_rf_pca <- rfsrc(Surv(month_per, FU1_case) ~ ., data=train_pca)
# harrell_c(train$month_per[1:48183], train_rf_pca$predicted[1:48183], train$FU1_case[1:48183])   #0.601961

y_hat5 <- predict(train_rf_pca, newdata=test_pca)   #class and prob will be error because they are only meant for classification trees. 
harrell_c(test_pca$month_per, y_hat5$predicted, test_pca$FU1_case)  #0.5333363

#? Xem lai cph.predict_partial_hazard() in Python
#cph.fit(one_hot_train, duration_col = 'month_per', event_col = 'FU1_case', step_size=0.1)
cox.model <- coxph(Surv(month_per, FU1_case==1)~., data=train_pca)  #Error
test_validation <-  predict(cox.model, newdata = test_pca) 
scores_values <- read.csv("../scores_values.csv", stringsAsFactors = F)[,2]   
harrell_c(test$month_per, scores_values, test$FU1_case)  # cph.predict_partial_hazard() in Python, highest 0.55

# Tuning and evaluation
#? Cach chon tuning param for paca, mtry, ntree
# Without covariate
tuning_test <- function(pca, mtry, ntree, p){
  c_indice <- vector(length = length(pca)*length(mtry)*length(ntree))
  z <- 1
  for (i in pca){
    for (j in mtry){
      for (e in ntree){
        data <- data.frame(train_pca[1:p,1:(i+6)],
                           month_per = train_pca$month_per[1:p], 
                           FU1_case = train_pca$FU1_case[1:p],
                           row.names = NULL)  
        set.seed(1)
        model <- rfsrc(Surv(month_per, FU1_case) ~ ., 
                       mtry = j,
                       ntree = e,
                       data=data)
        yhat <- predict(model, newdata=data.frame(test_pca[1:p,1:(i+6)],   #? test binh thuong hay la pca_transformed
                                                  row.names = NULL))  #?check lai column cua pcs_test
        c_indice[z] <-  harrell_c(test_pca$month_per[1:p], yhat$predicted, test_pca$FU1_case[1:p])
        z = z+1 #? how to identify z
      }
    }
  }
  return(c_indice)
}

tuning <- function(pca, mtry, ntree, test_data){
  c_indice <- vector(length = length(pca)*length(mtry)*length(ntree))
  z <- 1
  for (i in pca){
    for (j in mtry){
      for (e in ntree){
        data <- data.frame(train_pca[,1:(i+6)],
                           month_per = train_pca$month_per, 
                           FU1_case = train_pca$FU1_case,
                           row.names = NULL)  #?check lai
        set.seed(1)
        model <- rfsrc(Surv(month_per, FU1_case) ~ ., 
                       mtry = j,
                       ntree = e,
                       data=data)
        yhat <- predict(model, newdata=data.frame(test_data[,1:(i+6)],
                                                  row.names = NULL))  #?check lai column cua pcs_test
        c_indice[z] <-  harrell_c(test_data$month_per, yhat$predicted, test_data$FU1_case)
        z = z+1 #? how to identify z
      }
    }
  }
  return(c_indice)
}

tuning(c(1,5), c(5,7), c(5,10), test_pca)  #0.5030026 0.5012298 0.4884505 0.4855985 0.4855709 0.5033132 0.4837405 0.4844403

tuning(c(20,36), c(5,7), c(5,10), test_pca) #0.5094330 0.5049556 0.4713184 0.5105453 0.5214320 0.5146556 0.5092184 0.5092996
tuning(c(20,36), c(5,7), c(5,10), train_pca)

# Q: choose the minimum number of principal components such that 95% of the variance is retained.

##################
# set.seed(1)
# train_rf_pca_tune <- rfsrc(Surv(month_per, FU1_case) ~ ., 
#                       tuneGrid = expand.grid(.mtry=c(1:15),.ntree=seq(40, 400, 40)), #data.frame(cp = seq(0.0, 0.1, len = 25)),
#                       data=train_data_r)
# ggplot(train_rf_pca_tune)
# plot(train_rf_pca_tune)
# y_hat_tune <- predict(train_rf_pca_tune, newdata=pcs_test) 
# harrell_c(test$month_per, y_hat_tune$predicted, test$FU1_case)
# ###
# 
# tune(Surv(month_per, FU1_case) ~ .,
#      data = train_data_r,
#      mtryStart = ncol(train_data_r)/2,
#      ntreeTry =  50)
# 
# 
# 
# # 1
# tune(formula, data,
#     mtryStart = ncol(data) / 2,
#     nodesizeTry = c(1:9, seq(10, 100, by = 5)), ntreeTry = 50)
# 
# # 2 sample
# train_rpart <- train(margin ~ ., method = "rpart", tuneGrid = data.frame(cp = seq(0, 0.05, len = 25)), data = polls_2008)
# ggplot(train_rpart)
# 
# # 3
# train_rpart <- train(y ~ .,
#                      method = "rpart",
#                      tuneGrid = data.frame(cp = seq(0.0, 0.1, len = 25)),
#                      data = mnist_27$train)
# plot(train_rpart)
# 
# # 4
# # use cross validation to choose parameter
# train_rf_2 <- train(y ~ .,
#                     method = "Rborist",		#diff RF algorithm Rborist that is a little bit faster
#                     tuneGrid = data.frame(predFixed = 2, minNode = c(3, 50)),
#                     data = mnist_27$train)
# confusionMatrix(predict(train_rf_2, mnist_27$test), mnist_27$test$y)$overall["Accuracy"]
# 
# library(Rborist)		#Rborist faster than random forest package since Computation time is a big challenge in random forest, 
# control <- trainControl(method="cv", number = 5, p = 0.8)   #Because with random forest, the fitting is the slowest part of the procedure rather than the predicting, as with knn, we will only
# # use five-fold cross-validation.
# grid <- expand.grid(minNode = c(1,5) , predFixed = c(10, 15, 25, 35, 50))
# train_rf <-  train(x[, col_index], y,			#There appears to be an issue with Version 0.1-17 of the Rborist package that causes R sessions to abort/terminate. We recommend using an older version of Rborist or not running this code
#                    method = "Rborist",
#                    nTree = 50,				#reduce no. of tree bcs not yet building the final model
#                    trControl = control,
#                    tuneGrid = grid,
#                    nSamp = 5000)			#nSamp: take a random subset of obs when constructing each tree
# 


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

# Pairplot (beside of biplot)
# https://github.com/kevinblighe/PCAtools


######################
# Random Forest
# model = forest.rfsrc(ro.Formula('Surv(month_per, FU1_case) ~ .'), data=train, ntree=300, nodedepth=5, seed=-1)
# print(model)
# result = R.predict(model, newdata=test)
# scores = np.array(result.rx('predicted')[0])
# 
# print("Cox Model Test Score:", cox_test_scores)  
# print("Survival Forest Test Score:", harrell_c(test['month_per'].values, scores, test['FU1_case'].values))  

# Grid search

