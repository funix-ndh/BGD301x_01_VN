from pyspark import SparkConf, SparkContext
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.tree import DecisionTree, DecisionTree
from numpy import array
from time import time

conf = SparkConf().setAll([('spark.executor.memory', '16g'),('spark.driver.memory','16g')])
sc = SparkContext("local", "lesson9", conf=conf)

data_file = "kddcup.data.gz"
raw_data = sc.textFile(data_file)
print("Train data size is {}".format(raw_data.count()))

test_file = "corrected.gz"
test_data = sc.textFile(test_file)
print("Test data size is {}".format(test_data.count()))

csv_data = raw_data.map(lambda x: x.split(","))
test_csv_data = test_data.map(lambda x: x.split(","))

protocols = csv_data.map(lambda x: x[1]).distinct().collect()
services = csv_data.map(lambda x: x[2]).distinct().collect()
flags = csv_data.map(lambda x: x[3]).distinct().collect()


def create_labeled_point(line_split):
    clean_line_split = line_split[0:41]

    # to numeric categorical variable
    try:
        clean_line_split[1] = protocols.index(clean_line_split[1])
    except:
        clean_line_split[1] = len(protocols)

    try:
        clean_line_split[2] = services.index(clean_line_split[2])
    except:
        clean_line_split[2] = len(services)

    try:
        clean_line_split[3] = flags.index(clean_line_split[3])
    except:
        clean_line_split[3] = len(flags)

    attack = 1.0
    if line_split[41] == 'normal.':
        attack = 0.0

    return LabeledPoint(attack, array([float(x) for x in clean_line_split]))


training_data = csv_data.map(create_labeled_point)
test_data = test_csv_data.map(create_labeled_point)

t0 = time()
tree_model = DecisionTree.trainClassifier(training_data, numClasses=2,
                                          categoricalFeaturesInfo={1: len(protocols), 2: len(services), 3: len(flags)},
                                          maxDepth=4, maxBins=100)
tt = time() - t0
print("Classifier trained in {} seconds".format(round(tt, 3)))


t0 = time()
predictions = tree_model.predict(test_data.map(lambda x: x.features))
labels_and_preds = test_data.map(lambda x: x.label).zip(predictions)
test_accuracy = labels_and_preds.filter(lambda x: x[0] == x[1]).count() / float(test_data.count())
tt = time() - t0
print("Prediction made in {} seconds. Test accuracy is {}".format(round(tt, 3), round(test_accuracy, 3)))


# def create_labeled_point_minimal(line_split):
#     clean_line_split = line_split[3:4] + line_split[5:6] + line_split[22:23]

#     try:
#         clean_line_split[0] = flags.index(clean_line_split[0])
#     except:
#         clean_line_split[0] = len(flags)

#         attack = 1.0
#         if line_split[41] == 'normal.':
#             attack = 0.0

#         return LabeledPoint(attack, array([float(x) for x in clean_line_split]))


# training_data_minimal = csv_data.map(create_labeled_point_minimal)
# test_data_minimal = test_csv_data.map(create_labeled_point_minimal)

# t0 = time()
# tree_model_minimal = DecisionTree.trainClassifier(training_data_minimal, numClasses=2,
#                                                   categoricalFeaturesInfo={
#                                                       0: len(flags)}, maxDepth=3, maxBins=32)
# tt = time() - t0
# print("Classifier trained in {} seconds".format(round(tt, 3)))

# predictions_minimal = tree_model_minimal.predict(
#     test_data_minimal.map(lambda p: p.features))
# labels_and_preds_minimal = test_data_minimal.map(
#     lambda p: p.label).zip(predictions_minimal)

# t0 = time()
# test_accuracy = labels_and_preds_minimal.filter(lambda p: p[0] == p[1]).count() / float(test_data_minimal.count())
# tt = time() - t0

# print("Prediction made in {} seconds. Test accuracy is {}".format(
#     round(tt, 3), round(test_accuracy, 4)))
