import urllib.request
from pyspark import SparkContext
from time import time
from pprint import pprint

sc = SparkContext("local", "lesson3")
# download_url = "http://kdd.ics.uci.edu/databases/kddcup99/kddcup.data.gz"
file_path = "kddcup.data.gz"

# urllib.request.urlretrieve(download_url, file_path)

raw_data = sc.textFile(file_path)
total_size = raw_data.count()
# =========================== Sampling RDDs =================================
# In Spark, there are two sampling operations, the transformation `sample` and the action `takeSample`.
# By using a transformation we can tell Spark to apply successive transformation on a sample of a given RDD.
# By using an action we retrieve a given sample and we can have it in local memory to be used by any other standard library


# The `sample` transformation takes up to three parameters.
# First is whether the sampling is done with replacement or not.
# Second is the sample size as a fraction.
# Finally we can optionally provide a *random seed*.
raw_data_sample = raw_data.sample(False, 0.1, 1234)
sample_size = raw_data_sample.count()
print("Sample size is {} of {} total size".format(sample_size, total_size))

raw_data_sample_items = raw_data_sample.map(lambda x: x.split(','))
sample_normal_tags = raw_data_sample_items.filter(lambda x: 'normal.' in x)

t0 = time()
sample_normal_tags_count = sample_normal_tags.count()
tt = time() - t0

sample_normal_ratio = sample_normal_tags_count / float(sample_size)

print("The ratio of 'normal' interaction is {} ".format(
    round(sample_normal_ratio, 3)))

print("count sample normal tag done in {} seconds".format(round(tt, 3)))

# The `takeSample` action
# If what we need is to grab a sample of raw data from our RDD into local memory in order to be used by other non-Spark libraries, `takeSample` can be used.
# The syntax is very similar, but in this case we specify the number of items instead of the sample size as a fraction of the complete data size.
t0 = time()
raw_data_take_sample = raw_data.takeSample(False, 40000)
normal_data_take_sample = [x.split(',')
                           for x in raw_data_take_sample if "normal." in x]
tt = time() - t0

normal_ratio = len(normal_data_take_sample) / 400000.0
print("The ratio of 'normal' interactions is {} ".format(round(normal_ratio, 3)))
print("Count done in {} seconds".format(round(tt, 3)))
