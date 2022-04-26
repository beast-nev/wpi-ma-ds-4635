data1 <- read.csv("C:\\Users\\17817\\Downloads\\tabular-playground-series-apr-2022 (2)\\sample_submission.csv")
datatest <- read.csv("C:\\Users\\17817\\Downloads\\tabular-playground-series-apr-2022 (2)\\test.csv")
datatrain <- read.csv("C:\\Users\\17817\\Downloads\\tabular-playground-series-apr-2022 (2)\\train.csv")
datatrainlables <- read.csv("C:\\Users\\17817\\Downloads\\tabular-playground-series-apr-2022 (2)\\train_labels.csv")

str(datatrain)
summary(datatrain)
str(datatrainlables)
summary(datatrainlables)

#correlations in sensor data
sensors <- paste0('sensor_0', 0:9)
sensors <- c(sensors, paste0('sensor_', 10:12))
corrplot::corrplot(cor(datatrain[,sensors]))
corrplot::corrplot(cor(datatest[,sensors]))

dataexample <- datatrain[datatrain$sequence==1,]
summary(dataexample)
str(dataexample)

ggplot(data=datatrain, aes(x=sensor_00))+ geom_histogram(bins=75, fill='burlywood', col='white') + xlim(30,400)
ggplot(data=datatrain, aes(x=sensor_01))+ geom_histogram(bins=75, fill='burlywood', col='white') + xlim(30,400)
ggplot(data=datatrain, aes(x=sensor_02))+ geom_histogram(bins=75, fill='burlywood', col='white') + xlim(30,400)
ggplot(data=datatrain, aes(x=sensor_03))+ geom_histogram(bins=75, fill='burlywood', col='white') + xlim(30,400)
ggplot(data=datatrain, aes(x=sensor_04))+ geom_histogram(bins=75, fill='burlywood', col='white') + xlim(30,400)
ggplot(data=datatrain, aes(x=sensor_05))+ geom_histogram(bins=75, fill='burlywood', col='white') + xlim(30,400)
ggplot(data=datatrain, aes(x=sensor_06))+ geom_histogram(bins=75, fill='burlywood', col='white') + xlim(30,400)
ggplot(data=datatrain, aes(x=sensor_07))+ geom_histogram(bins=75, fill='burlywood', col='white') + xlim(30,400)
ggplot(data=datatrain, aes(x=sensor_08))+ geom_histogram(bins=75, fill='burlywood', col='white') + xlim(30,400)
ggplot(data=datatrain, aes(x=sensor_09))+ geom_histogram(bins=75, fill='burlywood', col='white') + xlim(30,400)
ggplot(data=datatrain, aes(x=sensor_10))+ geom_histogram(bins=75, fill='burlywood', col='white') + xlim(30,400)
ggplot(data=datatrain, aes(x=sensor_11))+ geom_histogram(bins=75, fill='burlywood', col='white') + xlim(30,400)
ggplot(data=datatrain, aes(x=sensor_12))+ geom_histogram(bins=75, fill='burlywood', col='white') + xlim(30,400)


#Bayes Classifier

str(dataexample)
'data.frame':	60 obs. of  16 variables:
  $ sequence : int  1 1 1 1 1 1 1 1 1 1 ...
$ subject  : int  66 66 66 66 66 66 66 66 66 66 ...
$ step     : int  0 1 2 3 4 5 6 7 8 9 ...
$ sensor_00: num  -6.658 1.634 1.863 -2.846 0.594 ...
$ sensor_01: num  -0.142 0.586 -2.144 2.012 -0.613 ...
$ sensor_02: num  -2.33 -2.13 -2 -2 -2 ...
$ sensor_03: num  -0.716 5.637 -4.159 -1.712 0.958 ...
$ sensor_04: num  0.789 0.613 -1.418 -1.344 0.457 ...
$ sensor_05: num  -0.484 -1.288 1.675 -4.603 4.559 ...
$ sensor_06: num  -4.568 -0.885 2.775 -3.163 2.065 ...
$ sensor_07: num  -1.44 5.39 -3.86 -0.782 -0.349 ...
$ sensor_08: num  -0.1 0.5 -1.2 2 -1.8 1.8 -1.7 -0.6 0.3 0.1 ...
$ sensor_09: num  -3.517 0.632 1.471 -1.801 0.254 ...
$ sensor_10: num  1.17 1.18 -1.92 -1.28 1.44 ...
$ sensor_11: num  1.525 2.387 -3.511 0.852 0.111 ...
$ sensor_12: num  -12.45 84.49 -35.94 -160.23 -9.15 ...
