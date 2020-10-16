from pyspark import SparkContext
import re

sc = SparkContext("local", "asm1")

raw_data = sc.textFile("input")

raw_split_data = raw_data.flatMap(lambda line: re.split("[#,-.;@\']", line))
raw_integer_data = raw_split_data.filter(lambda x: x.isnumeric())
integer_data = raw_integer_data.map(lambda x: int(x))

max_each_partition_integer_data = integer_data.repartition(10).mapPartitions(
    lambda it: [max(it)]).collect()

print("=========================================================")
# print max number in each partition
for idx, val in enumerate(max_each_partition_integer_data):
    print("max number in partition {}: {}".format(idx + 1, val))

print("=========================================================")
# print max number in set of max each partition
max_number = max(max_each_partition_integer_data)
print('max number in total: {}'.format(max_number))

print("=========================================================")
# print average
sum_number = integer_data.reduce(lambda x, y: x+y)
print("average = {}".format(sum_number/integer_data.count()))

# find top 10 frequency
# using api
pair_integer_data = integer_data.map(lambda x: (x, 1))
count_pair = pair_integer_data.reduceByKey(lambda x, y: x+y)
top_10_frequency1 = count_pair.takeOrdered(10, lambda x: -x[1])
print("=========================================================")
print("top 10 frequency number using api:")
for x in top_10_frequency1:
    print("number {} occur {} time(s)".format(x[0], x[1]))

# using merge sort
top_10_frequency2 = count_pair.repartitionAndSortWithinPartitions(
    numPartitions=10, ascending=False
).map(lambda x: (x[1], x[0])).sortByKey(False).take(10)
print("=========================================================")
print("top 10 frequency number using merge sort:")
for x in top_10_frequency2:
    print("number {} occur {} time(s)".format(x[1], x[0]))
