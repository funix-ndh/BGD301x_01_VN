from pyspark import SparkContext
from pyspark.sql import SQLContext, Row
from time import time

sc = SparkContext("local", "lesson10")
sql = SQLContext(sc)

file_path = "kddcup.data_10_percent.gz"

raw_data = sc.textFile(file_path).cache()
csv_data = raw_data.map(lambda x: x.split(","))
row_data = csv_data.map(lambda p: Row(
    duration=int(p[0]),
    protocol_type=p[1],
    service=p[2],
    flag=p[3],
    src_bytes=int(p[4]),
    dst_bytes=int(p[5])
))

interactions_df = sql.createDataFrame(row_data)
interactions_df.registerTempTable("interactions")

# Now we can run SQL queries over our data frame that has been registered as a table.
tcp_interaction = sql.sql("""
  SELECT duration, dst_bytes
  FROM interactions
  WHERE
    protocol_type = 'tcp'
    AND duration > 1000
    AND dst_bytes = 0
""")
tcp_interaction.show()

# The results of SQL queries are RDDs and support all the normal RDD operations.
tcp_interactions_out = tcp_interaction.rdd.map(
    lambda p: "Duration: {}, Dest. bytes: {}".format(p.duration, p.dst_bytes))

for ti_out in tcp_interactions_out.collect():
    print(ti_out)

# We can easily have a look at our data frame schema using `printSchema`.
interactions_df.printSchema()


# Queries as `DataFrame` operations
t0 = time()
interactions_df.select("protocol_type", "duration", "dst_bytes").groupBy(
    "protocol_type").count().show()
tt = time() - t0

print("Query perform in {} seconds".format(round(tt, 3)))

# Now imagine that we want to count how many interactions last more than 1 second,
# with no data transfer from destination, grouped by protocol type.
# We can just add to filter calls to the previous.
t0 = time()
interactions_df.select("protocol_type", "duration", "dst_bytes").filter(interactions_df.duration > 1000).filter(
    interactions_df.dst_bytes == 0).groupBy("protocol_type").count().show()
tt = time() - t0
print("Query perform in {} seconds".format(round(tt, 3)))


# add label column to dataframe
def get_label_type(label):
    if label != 'normal.':
        return 'attack'
    return 'normal'


row_labeled_data = csv_data.map(lambda p: Row(
    duration=int(p[0]),
    protocol_type=p[1],
    service=p[2],
    flag=p[3],
    src_bytes=int(p[4]),
    dst_bytes=int(p[5]),
    label=get_label_type(p[41])))

interactive_label_df = sql.createDataFrame(row_labeled_data)

t0 = time()
interactive_label_df.select('label').groupBy('label').count().show()
tt = time() - t0
print('Query performed in {} seconds'.format(round(tt, 3)))

t0 = time()
interactive_label_df.select('label', 'protocol_type').groupBy('protocol_type', 'label').count().show()
tt = time() - t0
print('Query performed in {} seconds'.format(round(tt, 3)))

t0 = time()
interactive_label_df.select('label', 'protocol_type', 'dst_bytes').groupBy('protocol_type', 'label', interactive_label_df.dst_bytes == 0).count().show()
tt = time() - t0
print('Query performed in {} seconds'.format(round(tt, 3)))
