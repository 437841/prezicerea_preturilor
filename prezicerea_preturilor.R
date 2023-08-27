library(neuralnet)
install.packages("e1071")
library(e1071)
library(boot)
library(ISLR)
library(caTools)
View(Housing)
set.seed(500)

Housing$driveway <- ifelse(Housing$driveway=="yes",1,0)
Housing$recroom <- ifelse(Housing$recroom=="yes",1,0)
Housing$fullbase <- ifelse(Housing$fullbase=="yes",1,0)
Housing$gashw <- ifelse(Housing$gashw=="yes",1,0)
Housing$airco <- ifelse(Housing$airco=="yes",1,0)
Housing$prefarea <- ifelse(Housing$prefarea=="no",0,1)

Housing <- subset(Housing,select=-c(X))

data <- Housing
apply(data,2,function(x) sum(is.na(x)))

index <- sample(1:nrow(data),round(0.75*nrow(data)))

#Metoda regresiei liniare
train <- data[index,]
test <- data[-index,]

lm.fit <- glm(price~., data=train)
summary(lm.fit)

pr.lm <- predict(lm.fit,test)

MSE.lm <- sum((pr.lm - test$price)^2)/nrow(test)
MSE.lm


#Metoda retelei neuronale
maxs <- apply(data, 2, max)
mins <- apply(data, 2, min)

scaled <- as.data.frame(scale(data, center = mins, scale = maxs - mins))

train_ <- scaled[index,]
test_ <- scaled[-index,]

n <- names(train_)
f <- as.formula(paste("price ~", paste(n[!n %in% "price"], collapse = " + ")))
nn <- neuralnet(f,data=train_,hidden=c(5,3),rep=12, linear.output=T)

plot(nn)

pr.nn <- neuralnet::compute(nn,test_[,1:12])

pr.nn_ <- pr.nn$net.result*(max(data$price)-min(data$price))+min(data$price)
test.r <- (test_$price)*(max(data$price)-min(data$price))+min(data$price)

#calculam MSE pentru setul de date de test
MSE.nn <- sum((test.r - pr.nn_)^2)/nrow(test_)

print(paste(MSE.lm,MSE.nn))

#comparatie intre performatele modelului liniar si al retelei neuronale
par(mfrow=c(1,2))

plot(test$price,pr.nn_,col='red',main='Valori reale vs prezise NN',pch=18,cex=0.7)
abline(0,1,lwd=2)
legend('bottomright',legend='NN',pch=18,col='red', bty='n')

plot(test$price,pr.lm,col='blue',main='Valori reale vs prezise de lm',pch=18, cex=0.7)
abline(0,1,lwd=2)
legend('bottomright',legend='LM',pch=18,col='blue', bty='n', cex=.95)


plot(test$price,pr.nn_,col='red',main='Valori prezise LM vs prezise NN',pch=18,cex=0.7)
points(test$price,pr.lm,col='blue',pch=18,cex=0.7)
abline(0,1,lwd=2)
legend('bottomright',legend=c('NN','LM'),pch=18,col=c('red','blue'))


#validare incrucisata cu K=10

#incepem cu modelul liniar
set.seed(500)
lm.fit <- glm(price~.,data=data)
cross_valid.lm <- cv.glm(data,lm.fit,K=10)

cross_valid.lm$delta[1]

#acum trecem la reteaua neuronala

cv.error <- NULL
k <- 10
library(plyr)
pbar <- create_progress_bar('text')
pbar$init(k)

for(i in 1:k){
  index <- sample(1:nrow(data),round(0.9*nrow(data)))
  train.cv <- scaled[index,]
  test.cv <- scaled[-index,]
  
  nn <- neuralnet(f,data=train.cv,rep=12, hidden=c(5,3),linear.output=T)
  
  pr.nn <- neuralnet::compute(nn,test.cv[,1:13])
  pr.nn <- pr.nn$net.result*(max(data$price)-min(data$price))+min(data$price)
  
  test.cv.r <- (test.cv$price)*(max(data$price)-min(data$price))+min(data$price)
  
  cv.error[i] <- sum((test.cv.r - pr.nn)^2)/nrow(test.cv)
  
  pbar$step()
}

mean(cv.error)


cv.error

boxplot(cv.error,xlab='MSE CV',col='cyan', border='blue',names='CV error (MSE)',
        main='Eroarea pentru RNA',horizontal=TRUE)


#Metoda SVM
split = sample.split(Housing$price, SplitRatio = 0.75)
head(split)
train_set = subset(Housing, split == TRUE)
test_set = subset(Housing, split == FALSE) 
head(train_set)
train_set[-1] = scale(train_set[-1])
test_set[-1] = scale(test_set[-1]) 
head(train_set)

model_svm <- svm(price~., data=train_set,type = 'C-classification', kernel = 'linear')
summary(model_svm)

pred = predict(model_svm, newdata = test_set[, 2:12])

sum(pred == test_set[,1])/length(pred)


error <- train$price - pred
svm_error <- sqrt(mean(error^2))


# Cautam cel mai bun model (valoarea optima a parametrului cost)
svm_tune <- tune(model_svm, price ~., data = train,
                 ranges = list(cost = 2^(2:19))
)

print(svm_tune)


# Cel mai bun model gasit
best_mod <- svm_tune$best.model

# Folosim cel mai bun model pentru a calcula valorile presize
best_mod_pred <- predict(best_mod, train) 

# Calculam eroarea de predictie
error_best_mod <- train$Salary - best_mod_pred 

# Calculam RMSE
best_mod_RMSE <- sqrt(mean(error_best_mod^2)) 

# Reprezentam grafic modelul
plot(svm_tune)
plot(train,pch=16)
points(train$Educ, best_mod_pred, col = "blue", pch=4)

