from pyspark import SparkContext
import numpy as np
from pprint import pprint
from pyspark.mllib.stat import Statistics
from math import sqrt
import pandas as pd

file_path = "kddcup.data_10_percent.gz"

sc = SparkContext("local", "lesson7")

raw_data = sc.textFile(file_path)

# local vectors
# A local vector is often used as a base type for RDDs in Spark MLlib.
# A local vector has integer-typed and 0-based indices and double-typed values, stored on a single machine.
# MLlib supports two types of local vectors: dense and sparse.
# A dense vector is backed by a double array representing its entry values,
# while a sparse vector is backed by two parallel arrays: indices and values.

# An RDD of dense vectors
# Let's represent each network interaction in our dataset as a dense vector.
# For that we will use the *NumPy* `array` type.


def parse_interactions(line):
    line_split = line.split(",")
    symbolic_indexes = [1, 2, 3, 41]
    clean_line_split = [item for i, item in enumerate(
        line_split) if i not in symbolic_indexes]
    return np.array([float(x) for x in clean_line_split])


vector_data = raw_data.map(parse_interactions)

# Summary statistics
# Compute column summary statistics.
summary = Statistics.colStats(vector_data)

print("Duration Statistics:")
print(" Mean: {}".format(round(summary.mean()[0], 3)))
print(" St. deviation: {}".format(round(sqrt(summary.variance()[0]), 3)))
print(" Max value: {}".format(round(summary.max()[0], 3)))
print(" Min value: {}".format(round(summary.min()[0], 3)))
print(" Total value count: {}".format(summary.count()))
print(" Number of non-zero values: {}".format(summary.numNonzeros()[0]))


# Summary statistics by label
# The interesting part of summary statistics, in our case,
# comes from being able to obtain them by the type of network attack or 'label' in our dataset.
# By doing so we will be able to better characterise our dataset dependent variable
# in terms of the independent variables range of values.


def parse_interaction_with_key(line):
    line_split = line.split(",")
    # keep just numeric and logical values
    symbolic_indexes = [1, 2, 3, 41]
    clean_line_split = [item for i, item in enumerate(
        line_split) if i not in symbolic_indexes]
    return (line_split[41], np.array([float(x) for x in clean_line_split]))


# label_vector_data = raw_data.map(parse_interaction_with_key)
# normal_label_data = label_vector_data.filter(lambda x: x[0] == "normal.")
# normal_summary = Statistics.colStats(normal_label_data.values())
# print("Duration Statistics for label: {}".format("normal"))
# print(" Mean: {}".format(round(normal_summary.mean()[0], 3)))
# print(" St. deviation: {}".format(round(sqrt(normal_summary.variance()[0]), 3)))
# print(" Max value: {}".format(round(normal_summary.max()[0], 3)))
# print(" Min value: {}".format(round(normal_summary.min()[0], 3)))
# print(" Total value count: {}".format(normal_summary.count()))
# print(" Number of non-zero values: {}".format(normal_summary.numNonzeros()[0]))

def summary_by_label(raw_data, label):
    label_vector_data = raw_data.map(parse_interaction_with_key).filter(lambda x: x[0]==label)
    return Statistics.colStats(label_vector_data.values())

normal_sum = summary_by_label(raw_data, "normal.")
print("Duration Statistics for label: {}".format("normal"))
print(" Mean: {}".format(round(normal_sum.mean()[0],3)))
print(" St. deviation: {}".format(round(sqrt(normal_sum.variance()[0]),3)))
print(" Max value: {}".format(round(normal_sum.max()[0],3)))
print(" Min value: {}".format(round(normal_sum.min()[0],3)))
print(" Total value count: {}".format(normal_sum.count()))
print(" Number of non-zero values: {}".format(normal_sum.numNonzeros()[0]))

guess_passwd_summary = summary_by_label(raw_data, "guess_passwd.")
print("Duration Statistics for label: {}".format("guess_password"))
print(" Mean: {}".format(round(guess_passwd_summary.mean()[0],3)))
print(" St. deviation: {}".format(round(sqrt(guess_passwd_summary.variance()[0]),3)))
print(" Max value: {}".format(round(guess_passwd_summary.max()[0],3)))
print(" Min value: {}".format(round(guess_passwd_summary.min()[0],3)))
print(" Total value count: {}".format(guess_passwd_summary.count()))
print(" Number of non-zero values: {}".format(guess_passwd_summary.numNonzeros()[0]))




label_list = ["back.","buffer_overflow.","ftp_write.","guess_passwd.",
              "imap.","ipsweep.","land.","loadmodule.","multihop.",
              "neptune.","nmap.","normal.","perl.","phf.","pod.","portsweep.",
              "rootkit.","satan.","smurf.","spy.","teardrop.","warezclient.",
              "warezmaster."]

stats_by_label = [(label, summary_by_label(raw_data, label)) for label in label_list]

# duration_by_label = [ 
#     (stat[0], np.array([float(stat[1].mean()[0]), float(sqrt(stat[1].variance()[0])), float(stat[1].min()[0]), float(stat[1].max()[0]), int(stat[1].count())])) 
#     for stat in stats_by_label]

# pd.set_option('display.max_columns', 50)
# stats_by_label_df = pd.DataFrame.from_dict(dict(duration_by_label), columns=["Mean", "Std Dev", "Min", "Max", "Count"], orient='index')
# print("Duration statistics, by label")
# print(stats_by_label_df)

def get_variable_stats_df(stats_by_label, column_i):
    column_stats_by_label = [
        (stat[0], np.array([float(stat[1].mean()[column_i]), float(sqrt(stat[1].variance()[column_i])), float(stat[1].min()[column_i]), float(stat[1].max()[column_i]), int(stat[1].count())])) 
        for stat in stats_by_label
    ]
    return pd.DataFrame.from_dict(dict(column_stats_by_label), columns=["Mean", "Std Dev", "Min", "Max", "Count"], orient='index')

print(get_variable_stats_df(stats_by_label, 0))