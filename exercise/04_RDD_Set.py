import urllib.request
from pyspark import SparkContext
from time import time
from pprint import pprint

sc = SparkContext("local", "lesson4")

file_path = "kddcup.data_10_percent.gz"

raw_data = sc.textFile(file_path)

normal_data = raw_data.filter(lambda x: 'normal.' in x)

attack_data = raw_data.subtract(normal_data)

# count all
t0 = time()
total_count = raw_data.count()
tt = time() - t0
print("All count in {} seconds".format(round(tt, 3)))

# count normal data
t0 = time()
normal_count = normal_data.count()
tt = time() - t0
print("Normal data count in {} seconds".format(round(tt, 3)))

# count attack data
t0 = time()
attack_count = attack_data.count()
tt = time() - t0
print("Attack data count in {}".format(round(tt, 3)))

# count
print("There are {} normal interactions and {} attacks, from a total of {} interactions".format(
    normal_count, attack_count, total_count))


# Protocol and service combinations using `cartesian`
# Obviously, for such small RDDs doesn't really make sense to use Spark cartesian product.
# We could have perfectly collected the values after using `distinct` and do the cartesian product locally.
# Moreover, `distinct` and `cartesian` are expensive operations so they must be used with care when the operating datasets are large.
csv_data = raw_data.map(lambda x: x.split(","))
protocols = csv_data.map(lambda x: x[1]).distinct()
pprint(protocols.collect())

services = csv_data.map(lambda x: x[2]).distinct()
pprint(services.collect())

product = protocols.cartesian(services)
pprint(product.collect())
