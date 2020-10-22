from pyspark import SparkContext
from os import listdir
from pprint import pprint
import re

sc = SparkContext("local", "asm4")

# initialize RDD for training only
training_rdd = sc.parallelize([])

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

    # skip validation & testing
    # since we have only worked with training data 60%
    training = raw_data.sample(withReplacement=False, fraction=0.6)

    # union data
    training_rdd += training

    print('finish processing file {}.'.format(f))
    print('==================================================')

    # full processing may take very long time to run
    # so we can `comment` or `uncomment` out line `break` bellow
    # this trick will help to toggle `real` and `debug` mode
    # it help verify & debug very quickly
    break

# stop keyword
stop_word = set(['in', 'on', 'of', 'out', 'by', 'from', 'to', 'over', 'under', 'the', 'a', 'an', 'when', 'where', 'what', 'who', 'whom', 'you', 'thou', 'go', 'must', 'i', 'me', 'my', 'myself', 'for', 'and', 'x', 'it', 'are', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'be', 'thi', 'with', 'this', 'that', 'or', 'if', 'have', 't', 'an', 'db', 'but', 'at', 'wa', 'they', 'will', 'can', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'one', 'zero',
                 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'do', 'did', 'here', 'there', 'all', 'subject', 'about', 'we', 'other', 'no', 're', 'ha', 'which', 'your', 'so', 'would', 'some', 'their', 'he', 'any', 'more', 'how', 'only', 'may', 'might', 'also', 'new', 'should', 'up', 'hi', 'dear', 'them', 'then', 'first', 'second', 'third', 'don', 'doe', 'were', 'know', 'than', 'less', 'most', 'get', 'year', 'like', 'been', 'use', 'many', ' few', 'little', 'just', 'make', 'these', 'those', 'because', 'not', 'into'])

# util function to normalize case
def normalize_case(line):
    return re.sub(r'[^a-z]', ' ', line.lower().strip()).strip()

# step 0: convert to lowercase
training_rdd = training_rdd.map(normalize_case).distinct()

# util function to split sentence to word and also strip out empty word
def split_sentence_and_strip_empty_word(line):
    return list(filter(lambda x: "" != x, line.split(" ")))

# step 1-1: load all words to RDD
# step 1-2: filter out stop keyword
# step 1-3: convert to tuple for counting
# step 1-4: count by key
# step 1-5: take top 200 largest frequency
top_frequency_word = training_rdd.flatMap(split_sentence_and_strip_empty_word) \
    .filter(lambda x: x not in stop_word) \
    .map(lambda x: (x, 1)) \
    .reduceByKey(lambda x, y: x+y) \
    .takeOrdered(200, lambda x: -x[1])

pprint(top_frequency_word)

# stop bigram
stop_bigram = set(['of the', 'x x', 'in the', 'to the', 'it i', 'on the', 'to be', 'for the', 'i a', 'subject re', 'and the', 'if you', 'don t', 'that the', 'in article', '0 1', '1 1', 'from the', 'thi i', 'with the', 'i not', 'it ', 'i the', 'the same', 'in a', 'of a', 'that i', 'for a', 'by the', 'will be', 'i m', 'i have', 'there i', 'the first', 'you are', 'with a', 'n x', 'a a', 'what i', 'doe not', 'to a', 'at the', 'do not', 'would be', 'can be', 'there are', '1 0', 'i am', 'they are', 'are not', 'you can', 'on a', 'and i', 'should be', 'may be', 'and a', 'have a', 'have been', 'such a', 'number of', 'that you', 'i ve', 'about the', ' want to', 'that it', 'which i', 'the following', 'x printf', 'but i', 'i don', 'can t', 'x if', 'file x',
                        'to do', 'to get', 'you have', 'one of', 'and other', 'a the', 'doesn t', '1 2', 'i can', 'that they', 'out of', 'i to', 'i that', 'all the', 'the other', 'how to', 'of thi', 'into the', 'be a', 'to have', 'c si', 'i think', 'are the', 'to make', 'ha been', 'isn t', '0 0', 'x char', 'ha a', 'must be', 'mov bh', 'it wa', 'have to', 'in thi', 'for example', 'if the', ' a', 'not a', 'that ', 'x the', 'of course', 'at least', 'a good', 'you re', 'write in', 'not to', 'part of', 'i one', '2 2', 'your entry', 'but the', 'a few', 'the u', 'the only', 'i would', 'i an', 'a well', 'u ', 'and that', 'i wa', 'sort of', 'lot of', '0 2', 'but it', 'if i', 'the most', 'and at', 'all of', 'to use', 'seem to', 'and it', 'i know', 'bl bh', 'to see', 'want to'])

# step 2-1: convert line to list of words
# step 2-2: load all bigrams to RDD (using zip & map)
# step 2-3: filter out stop bigram
# step 2-4: convert to tuple for counting
# step 2-5: count by key
# step 2-6: take top 100 largest frequency
top_frequency_bigram = training_rdd.map(split_sentence_and_strip_empty_word) \
    .flatMap(lambda x: zip(x, x[1:])).map(lambda x: x[0] + " " + x[1]) \
    .filter(lambda x: x not in stop_bigram) \
    .map(lambda x: (x, 1)) \
    .reduceByKey(lambda x, y: x+y) \
    .takeOrdered(100, lambda x: -x[1])

pprint(top_frequency_bigram)
