from pyspark import SparkContext, SparkConf
import urllib.request
from pyspark.mllib.regression import LabeledPoint
from numpy import array
from pyspark.mllib.classification import LogisticRegressionWithLBFGS
from time import time

conf = SparkConf().setAll([('spark.executor.memory', '16g'),('spark.driver.memory','16g')])
sc = SparkContext("local", "lesson8", conf=conf)

data_file = "kddcup.data.gz"
raw_data = sc.textFile(data_file)
print("Train data size is {}".format(raw_data.count()))

test_data_file = "corrected.gz"
test_raw_data = sc.textFile(test_data_file)
print("Test data size is {}".format(test_raw_data.count()))


def parse_interaction(line):
    line_split = line.split(",")
    clean_line_split = line_split[0:1]+line_split[4:14]
    attack = 1.0
    if line_split[41] == 'normal.':
        attack = 0.0
    return LabeledPoint(attack, array([float(x) for x in clean_line_split]))


training_data = raw_data.map(parse_interaction)
test_data = test_raw_data.map(parse_interaction)

t0 = time()
logist_model = LogisticRegressionWithLBFGS.train(training_data)
tt = time() - t0
print("Classifier trainned in {} seconds".format(round(tt, 3)))

labels_and_preds = test_data.map(lambda p: (p.label, logist_model.predict(p.features)))

t0 = time()
test_accuracy = labels_and_preds.filter(lambda p: p[0] == p[1]).count() / float(test_data.count())
tt = time() - t0
print("Prediction made in {} seconds. Test accuracy is {}".format(round(tt, 3), round(test_accuracy, 4)))


