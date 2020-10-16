from pyspark import SparkContext, SQLContext
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.util import MLUtils
from pyspark.mllib.tree import RandomForest, RandomForestModel
from numpy import array
from time import time

sc = SparkContext("local", "asm2")

# ====================== PREPARE & LOAD DATA ========================================
# assignment only allow to submit within 20MB
# so I can not include csv file here 
# file download here: 
# https://www.kaggle.com/c/allstate-claims-severity/data?select=train.csv
# https://www.kaggle.com/c/allstate-claims-severity/data?select=test.csv
raw_data = sc.textFile("input/train.csv") 
header = raw_data.first()  # extract header
raw_data = raw_data.filter(lambda x: x != header)  # ignore header

# split the data into training and test sets (30% held out for testing)
train_data, test_data = raw_data.randomSplit([0.7, 0.3])


# ====================== PARSE DATA =================================================
raw_data_csv = raw_data.map(lambda x: x.split(','))
train_data_csv = train_data.map(lambda x: x.split(','))
test_data_csv = test_data.map(lambda x: x.split(','))

print("raw_data_csv count {}".format(raw_data_csv.count()))
print("train_data_csv count {}".format(train_data_csv.count()))
print("test_data_csv count {}".format(test_data_csv.count()))


# ====================== BUILD LABELED POINT ========================================
cat_set = raw_data_csv.flatMap(lambda x: x[1:117]).distinct().collect()

def parse_labled_point(line_split):
    # convert to numeric categorical variable
    for i in range(1, 117):
        line_split[i] = cat_set.index(line_split[i])
    loss = line_split[len(line_split)-1]
    return LabeledPoint(loss, array(line_split[1:len(line_split)-1]))

train_data_labeled_point = train_data_csv.map(parse_labled_point)
test_data_labeled_point = test_data_csv.map(parse_labled_point)


# ======================= TRAIN MODEL =================================================
t0 = time()
# smaller MSE generally indicates a better estimate
# after tweak round parameters
# FeatureSubsetStrargety=auto => it will help us analyse and choose best algorithm base on dataset
# larger numTrees and maxDepth will be more accurate but it will take long time to train
# so I think 10 wound be balance
model = RandomForest.trainRegressor(train_data_labeled_point, 
                                    categoricalFeaturesInfo={}, numTrees=10, maxDepth=10,
                                    featureSubsetStrategy="auto")
tt = time() - t0
print("RandomForest trained in {} seconds".format(round(tt, 3)))


# ======================= TEMPORARY TEST PREDICT MODEL =================================================
t0 = time()
predictions = model.predict(test_data_labeled_point.map(lambda x: x.features))
labels_preds = test_data_labeled_point.map(lambda x: x.label).zip(predictions)
testMSE = labels_preds.map(lambda lp: (lp[0] - lp[1]) * (lp[0] - lp[1])).sum() / float(test_data_labeled_point.count())
tt = time() - t0
print("Prediction made in {} seconds.".format(round(tt, 3)))
print('Test Mean Squared Error = ' + str(testMSE))


# ======================= REAL TEST PREDICT MODEL =================================================
# make sure to download this file:
# https://www.kaggle.com/c/allstate-claims-severity/data?select=test.csv
real_test_data = sc.textFile("input/test.csv")
real_test_header = real_test_data.first()  # extract header
real_test_raw_data = real_test_data.filter(lambda x: x != real_test_header)  # ignore header

real_test_data_csv = real_test_raw_data.map(lambda x: x.split(','))
print("real_test_data_csv count {}".format(real_test_data_csv.count()))

def parse_test(line_split):
    # convert to numeric categorical variable
    for i in range(1, 117):
        try:
            line_split[i] = cat_set.index(line_split[i])
        except:
            line_split[i] = len(line_split[i])
    return array(line_split)

t0 = time()
predictions = model.predict(real_test_data_csv.map(parse_test))
result = real_test_data_csv.map(lambda x: x[0]).zip(predictions)
tt = time() - t0
print("Prediction made in {} seconds.".format(round(tt, 3)))

sql = SQLContext(sc)
df = sql.createDataFrame(result, ['id', 'loss'])

# get result and submit
# https://www.kaggle.com/c/allstate-claims-severity/submit
df.write.csv("out_{}".format(time()), sep=',', header=True) 
