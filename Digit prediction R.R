library("randomForest")
setwd("G://ml//Digit Recognizer data set")

train = read.csv("train.csv")
test = read.csv("test.csv")
RF_Model = randomForest(label ~.,train,importance=FALSE,ntree=100)
prediction = predict(RF_Model,test)

write.csv(prediction,file = "DigitPredictRandomForest.csv")
help("randomForest")


table(prediction)
