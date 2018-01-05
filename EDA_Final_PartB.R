

count01 = rep("Low", nrow(Bike_data))
count01[Bike_data$count > 724] = "High"


Bike_data = Bike_data[,-11]
Bike_data = cbind(Bike_data, count01)
summary(Bike_data)


Bike_data$season <- factor(Bike_data$season)
Bike_data$year <- factor(Bike_data$year)
Bike_data$month <- factor(Bike_data$month)
Bike_data$holiday <- factor(Bike_data$holiday)
Bike_data$weekday <- factor(Bike_data$weekday)
Bike_data$weathersit <- factor(Bike_data$weathersit)

# -------------------------------------------------------------------------------------------------------
createTrainingData <- function(dataset, sampleTrain){
  m <- nrow(dataset)
  set.seed(1)
  train <- sample(m, as.integer(sampleTrain*m))
  trainingData <- dataset[train,]
  return(trainingData)
}
createTestData <- function(dataset, sampleTest){
  train <- as.integer(row.names(createTrainingData(dataset, 1 - sampleTest)))
  testData <- dataset[-train,]
  return(testData)
}
sampleTrain <- 600/nrow(Bike_data)
bikeTrain <- createTrainingData(Bike_data, sampleTrain)
bikeTest <- createTestData(Bike_data, 1- sampleTrain)


#-------------------------------------------------------------------------------------------------------


errorCheck <- function(prediction, actual){
  confusionTable <- table(prediction,actual)
  print(confusionTable)
  accuracy <- mean(prediction==actual)
  errorRate <- 1 - accuracy
  TruePositive = confusionTable[2,2]
  TrueNegative = confusionTable[1,1]
  FalsePositive = confusionTable[2,1]
  FalseNegative = confusionTable[1,2]
  Positives = FalseNegative + TruePositive
  Negatives = FalsePositive + TrueNegative
  sensitivity <- TruePositive/Positives
  specificity <- 1 - FalsePositive/Negatives
  performanceTable <- matrix(c(errorRate,accuracy,sensitivity,specificity))
  rownames(performanceTable) <- c("Error Rate", "Accuracy", "Sensitivity","Specificity")
  return(performanceTable)
}

#---------------------------------------------------------------------------------------------
#Q1
Count01Fit <- glm(count01 ~ ., data = bikeTrain,  family = binomial)
summary(Count01Fit)
Count01cProbs <- predict(Count01Fit, bikeTest, type = 'response')
Count01Prediction <- ifelse(Count01cProbs > 0.5, "High", "Low")
Count01LogisticErrorTable <- errorCheck(Count01Prediction, bikeTest$count01)
colnames(Count01LogisticErrorTable) <- "Logistic"
print(Count01LogisticErrorTable)

#----------------------------------------------------------------------------------------
#Q2

library(tree)
library(boot)
library(randomForest)
treeBikes <- tree(count01 ~ ., data = bikeTrain)
summary(treeBikes)
plot(treeBikes)
text(treeBikes, pretty = FALSE)


treePreds1 <- predict(treeBikes, bikeTest, type = "class")
treeErrorTable <- errorCheck(treePreds1, bikeTest$count01)
colnames(treeErrorTable) <- "Tree"
print(treeErrorTable)

#pruned Tree
cvbikeTree <- cv.tree(treeBikes, FUN = prune.misclass)
bestTree <- cvbikeTree$size[which.min(cvbikeTree$dev)]
prunedTreeBikes <- prune.misclass(treeBikes, best = bestTree)
summary(prunedTreeBikes)
plot(prunedTreeBikes)
text(prunedTreeBikes, pretty = FALSE)

treePrunedErrorTable = predict(prunedTreeBikes, bikeTest, type = "class")
treePruneErrorTable <- errorCheck(treePrunedErrorTable, bikeTest$count01)
colnames(treePruneErrorTable) <- "TreePrune"
print(treePruneErrorTable)
#------------------------------------------------------------------------------------------------

#Q3.
#bagging
set.seed(1)
p <- ncol(Bike_data) - 1
set.seed(1)
bagBikes <- randomForest(count01 ~ ., data = bikeTrain, ntree = 100, mtry = p, importance = TRUE)
bagPreds <- predict(bagBikes, bikeTest)
bagErrorTable <- errorCheck(bagPreds, bikeTest$count01)
colnames(bagErrorTable) <- "Bagging"
print(bagErrorTable)

barplot(sort(bagBikes$importance[,"MeanDecreaseGini"]), col = 'Red',main = "Importance of predictors: Bagging" )

#RandomForest
p <- ncol(Bike_data) - 1
set.seed(1)
rfBikes <- randomForest(count01 ~ ., data = bikeTrain, ntree = 100, mtry = sqrt(p), importance = TRUE)
rfPreds <- predict(rfBikes, bikeTest)
rfErrorTable <- errorCheck(rfPreds, bikeTest$count01)
colnames(rfErrorTable) <- "Random Forest"
print(rfErrorTable)

barplot(sort(rfBikes$importance[,"MeanDecreaseGini"]), col = 'Red', main = "Importance of predictors: Random Forest")

#LDA
library(MASS)
set.seed(1)
ldaBikes <- lda(count01 ~ ., data = bikeTrain)
ldaProbs <- predict(ldaBikes, bikeTest, type = 'response')
ldaPreds <- ldaProbs$class
ldaErrorTable <- errorCheck(ldaPreds, bikeTest$count01)
colnames(ldaErrorTable) <- "LDA"
print(ldaErrorTable)

#QDA
set.seed(1)
qdaBikes <- qda(count01 ~ ., data = bikeTrain)
qdaProbs <- predict(qdaBikes, bikeTest, type = 'response')
qdaPreds <- qdaProbs$class
qdaErrorTable <- errorCheck(qdaPreds, bikeTest$count01)
colnames(qdaErrorTable) <- "QDA"
print(qdaErrorTable)


#SVC
library(e1071)
i <- -3:2
costs <- 10^i
gammas <- seq(0.5,5,by = 0.5)
degrees <- i[5:6]
svcTune <- tune(svm, count01 ~ ., data = bikeTrain, kernel = 'linear', ranges = list(cost = costs))
print(summary(svcTune))

svcBikes <- svm(count01 ~ ., data = bikeTrain,kernel = 'linear', cost = 10, scale = FALSE)
print(summary(svcBikes))

svcPreds <- predict(svcBikes, bikeTest)
svcErrorTable <- errorCheck(svcPreds, bikeTest$count01)
colnames(svcErrorTable) <- "SVC"
print(svcErrorTable)


#SVM Radial
svmRadialTune <- tune(svm, count01 ~ ., data = bikeTrain,kernel = 'radial', ranges = list(cost = costs, gamma = gammas))
print(summary(svmRadialTune))

svmRadialBikes <- svm(count01 ~ ., data = bikeTrain,kernel = 'radial', gamma = 0.5, cost = 10, scale = FALSE)
svmRadialPreds <- predict(svmRadialBikes, bikeTest)
svmRadialErrorTable <- errorCheck(svmRadialPreds, bikeTest$count01)
colnames(svmRadialErrorTable) <- "SVM Radial"
print(svmRadialErrorTable)


#SVM Poly

svmPolyTune <- tune(svm, count01 ~ ., data = bikeTrain, kernel = 'polynomial', ranges = list(cost = costs, degree = degrees))
print(summary(svmPolyTune))


svmPolyBikes <- svm(count01 ~ ., data = bikeTrain,kernel = 'polynomial', degree = 1, cost = 10, scale = FALSE)
svmPolyPreds <- predict(svmPolyBikes, bikeTest)
svmPolyErrorTable <- errorCheck(svmPolyPreds, bikeTest$count01)
colnames(svmPolyErrorTable) <- "SVM Poly"
print(svmPolyErrorTable)

#-----------------------------------------------------------------------------------------------------------------------------------
#Q4

SummaryErrorTable <- cbind(Count01LogisticErrorTable, treeErrorTable, treePruneErrorTable, rfErrorTable, bagErrorTable, 
                           ldaErrorTable, qdaErrorTable, svcErrorTable, svmRadialErrorTable, svmPolyErrorTable)
print(SummaryErrorTable)
