from pyspark import SparkContext
from os import listdir

sc = SparkContext("local", "asm3")

# initialize RDD for 3 parts
training_rdd = sc.parallelize([])
validation_rdd = sc.parallelize([])
testing_rdd = sc.parallelize([])

# note:
# because assignment submission only allow 20MB
# make sure to download `input data` file:
#   http://qwone.com/~jason/20Newsgroups/
# and locate it inside `20news-18828` directory

# loop & read all directories
input_base_path = "20news-18828"
list_file = listdir(input_base_path)
for f in list_file:
    print('begin processing file {} ...'.format(f))

    file_path = input_base_path + "/" + f
    raw_data = sc.textFile(file_path)

    # split 3 parts
    # training 60% - validation 20% - testing 20%
    training, validation, testing = raw_data.randomSplit([0.6, 0.2, 0.2])

    # key as (label, file_name)
    # value as text data
    training = training.map(lambda x: (('training', f), x))
    validation = validation.map(lambda x: (('validation', f), x))
    testing = testing.map(lambda x: (('testing', f), x))

    print('training count {}'.format(training.count()))
    print('validation count {}'.format(validation.count()))
    print('testing count {}'.format(testing.count()))

    # union data
    training_rdd += training
    validation_rdd += validation
    testing_rdd += testing

    print('finish processing file {}.'.format(f))
    print('==================================================')

# print sample result data
print('sample training data:')
print(training_rdd.take(1))
print('==================================================')

print('sample validation data:')
print(validation_rdd.take(1))
print('==================================================')

print('sample testing data:')
print(testing_rdd.take(1))
