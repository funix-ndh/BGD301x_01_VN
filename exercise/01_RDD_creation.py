import urllib.request
from pyspark import SparkContext


##### SECTION 1
##### Creating a RDD from a file  
sc = SparkContext("local", "lesson1")
download_url = "http://kdd.ics.uci.edu/databases/kddcup99/kddcup.data_10_percent.gz"
file_path = "kddcup.data_10_percent.gz"

# download and save to file
urllib.request.urlretrieve(download_url, file_path)

# Now we have our data file loaded into the `raw_data` RDD.
raw_data = sc.textFile(file_path)

# count number of lines loaded from file to the RDD
print("count: %i" % raw_data.count())

# take 5 item from RDD
print(raw_data.take(5))


##### SECTION 2
##### Creating and RDD using `parallelize`
a = range(100)

data = sc.parallelize(a)

# count the number of elements in the RDD.
print("count: %i" % data.count())

# take 5 item from RDD
print(data.take(5))