from pyspark import SparkContext
from pprint import pprint


sc = SparkContext("local", "lesson6")

file_path = "kddcup.data_10_percent.gz"

raw_data = sc.textFile(file_path)

csv_data = raw_data.map(lambda x: x.split(","))
key_value_data = csv_data.map(lambda x: (x[41], x))

pprint(key_value_data.take(1))

# Data aggregations with key/value pair RDDs
key_value_duration = csv_data.map(lambda x: (x[41], float(x[0])))
durations_by_key = key_value_duration.reduceByKey(
    lambda x, y: x+y)  # duration that group by interactions
pprint(durations_by_key.collect())

# Count by key
counts_by_key = key_value_data.countByKey()
pprint(counts_by_key)

# Using `combineByKey`
# This is the most general of the per-key aggregation functions.
# Most of the other per-key combiners are implemented using it.
# We can think about it as the `aggregate` equivalent since it allows the user to return values that are not the same type as our input data.
sums_counts = key_value_duration.combineByKey(
    lambda x: (x, 1),
    lambda acc, value: (acc[0] + value, acc[1] + 1),
    lambda acc1, acc2: (acc1[0] + acc2[0], acc1[1] + acc2[1])
)

pprint(sums_counts.collect())
pprint(sums_counts.collectAsMap())


duration_means_by_type = sums_counts.map(lambda pair: (pair[0], round(pair[1][0]/pair[1][1], 3))).collectAsMap()
# Print them sorted
for tag in sorted(duration_means_by_type, key=duration_means_by_type.get, reverse=True):
    print(tag, duration_means_by_type[tag])
