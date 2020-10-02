import urllib.request
from pyspark import SparkContext
from time import time
from pprint import pprint

sc = SparkContext("local", "lesson2")

download_url = "http://kdd.ics.uci.edu/databases/kddcup99/kddcup.data_10_percent.gz"
file_path = "kddcup.data_10_percent.gz"

urllib.request.urlretrieve(download_url, file_path)

raw_data = sc.textFile(file_path)

# ========================= TRANSFORMATION ==============================================
# The `filter` transformation
# This transformation can be applied to RDDs in order to keep just elements that satisfy a certain condition.
# More concretely, a function is evaluated on every element in the original RDD.
# The new resulting RDD will contain just those elements that make the function return `True`.
normal_raw_data = raw_data.filter(lambda x: 'normal.' in x)

# count total element on new RDD
t0 = time()
print('normal raw data: %i ' % normal_raw_data.count())
tt = time() - t0
print('count completed in {} seconds'.format(round(tt, 3)))


# The `map` transformation
# By using the `map` transformation in Spark, we can apply a function to every element in our RDD.
# Python's lambdas are specially expressive for this particular.
csv_data = raw_data.map(lambda x: x.split(','))
t0 = time()
# pprint(csv_data.take(5))
csv_data.take(5)
tt = time() - t0
print('parse completed in {} seconds'.format(round(tt, 3)))

# all action happens once we call the first Spark *action* (i.e. *take* in this case).
t0 = time()
# pprint(csv_data.take(1000))
csv_data.take(1000)
tt = time() - t0
print('take 1000 items - parse completed in {} seconds'.format(round(tt, 3)))

# Using `map` and predefined functions


def parse_interaction(line):
    el = line.split(",")
    tag = el[41]
    return (tag, el)  # (key - val) pair


key_csv_data = raw_data.map(parse_interaction)
head_rows = key_csv_data.take(5)
pprint(key_csv_data)

# ============================ ACTION =========================
# The collect action
# Basically it will get all the elements in the RDD into memory for us to work with them.
# For this reason it has to be used with care, specially when working with large RDDs.
# That took longer as any other action we used before, of course.
# Every Spark worker node that has a fragment of the RDD has to be coordinated in order to retrieve its part, and then *reduce* everything together.
t0 = time()
all_raw_data = raw_data.collect()
tt = time() - t0
print("Data memory collected in {} seconds".format(round(tt, 3)))

# filter normal key interactions
normal_key_interactions = key_csv_data.filter(lambda x: x[0] == 'normal.')
t0 = time()
all_normal = normal_key_interactions.collect()
tt = time() - t0
print("Normal key interactions - Data memory collected in {} seconds".format(round(tt, 3)))