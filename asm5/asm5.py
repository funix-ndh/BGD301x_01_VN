from pyspark import SparkContext, SQLContext
from pyspark.sql import Row
from pyspark.sql.functions import monotonically_increasing_id, udf
from pyspark.ml.feature import Word2Vec
from pyspark.mllib.linalg import Vectors, VectorUDT
from collections import Counter
import re


sc = SparkContext("local", "asm5")
sql = SQLContext(sc)


# note:
# because assignment submission only allow 20MB
# make sure to download `input data` file:
#   http://qwone.com/~jason/20Newsgroups/
# and locate it inside `20news-18828` directory


# debug mode
file_path = "20news-18828/alt.atheism"


# real mode (it will process a very big data as result will very slow)
# file_path = "20news-18828/*"


# util function to normalize case
def normalize_case(line):
    return re.sub(r'[^a-z]', ' ', line.lower().strip()).strip()


# util function to split sentence to word and also strip out empty word
def split_sentence_and_strip_empty_word(line):
    return list(filter(lambda x: "" != x, line.split(" ")))


# document
inp = sc.textFile(file_path) \
    .sample(withReplacement=False, fraction=0.6) \
    .map(normalize_case) \
    .distinct() \
    .map(split_sentence_and_strip_empty_word) \
    .filter(lambda x: len(x) > 0)


# top 200 words as result of asm4
top_200_words = ['is', 'was', 'as', 'people', 'had', 'israel', 'edu', 'his', 'armenian', 'said', 'armenians', 'has', 'jews', 'us', 'turkish', 'our', 'article', 'writes', 'israeli', 'its', 'jewish', 'even', 'arab', 'such', 'him', 'after', 'government', 'now', 'she', 'could', 'her', 'time', 'right', 'say', 'war', 'well', 'think', 'why', 'see', 'against', 'world', 'state', 'turkey', 'killed', 'years', 'way', 'didn', 'armenia', 'being', 'children', 'greek', 'muslim', 'turks', 'same', 'does', 'genocide', 'soldiers', 'muslims', 'went', 'let', 'com', 'told', 'rights', 'want', 'am', 'fact', 'down', 'human', 'going', 'arabs', 'back', 'before', 'between', 'during', 'azerbaijan', 'peace', 'off', 'still', 'very', 'too', 'while', 'never', 'came', 'anti', 'take', 'since', 'come', 'left', 'last', 'own', 'something', 'home', 'adl', 'again', 'says', 'russian', 'started', 'history', 'palestinian', 'country', 'group', 'much', 'population',
                 'll', 'part', 'another', 'both', 'soviet', 'through', 'away', 'long', 'saw', 'made', 'women', 'called', 'land', 'nothing', 'apr', 'took', 'really', 'point', 'old', 'case', 'states', 'believe', 'name', 'american', 'around', 'times', 'cs', 'ottoman', 'policy', 'later', 'university', 'military', 'police', 'law', 'greece', 'army', 'political', 'find', 've', 'palestinians', 'information', 'number', 'today', 'without', 'give', 'party', 'things', 'someone', 'good', 'kill', 'azerbaijani', 'ed', 'anything', 'got', 'day', 'news', 'troops', 'building', 'question', 'religious', 'apartment', 'true', 'villages', 'tell', 'problem', 'put', 'men', 'look', 'themselves', 'read', 'course', 'city', 'happened', 'press', 'whole', 'used', 'nazi', 'place', 'door', 'however', 'palestine', 'need', 'dead', 'baku', 'sure', 'work', 'th', 'among', 'everything', 'man', 'kuwait', 'person', 'mr', 'heard', 'according', 'international', 'anyone']


# top 100 bigrams as result of asm4
top_100_bigrams = ['it is', 'it was', 'they were', 'is a', 'didn t', 'the armenians', 'the armenian', 'is not', 'it s', 'the turkish', 'there is', 'as a', 'this is', 'i was', 'is the', 'had been', 'there were', 'he was', 'there was', 'was a', 'do you', 'of them', 'that is', 't know', 'the jews', 'going to', 'that s', 'the world', 'during the', 'as the', 'human rights', 'is that', 'said that', 'the jewish', 'of their', 'article apr', 'and they', 'did not', 'does not', 'that he', 'the greek', 'according to', 'of this', 'in this', 'was the', 'the adl', 'of israel', 'are you', 'right to', 'has been',
                   'couldn t', 'the people', 'when the', 'against the', 'that there', 'the israeli', 'he said', 'we were', 'u s', 'i had', 'i said', 'trying to', 'article c', 'in his', 'they had', 'the door', 'some of', 'is no', 'the fact', 'the way', 'and he', 'those who', 'when they', 'in their', 'who were', 'and then', 'what is', 'x soviet', 'the united', 'genocide of', 'most of', 'over the', 'the arab', 'when i', 'you know', 'fact that', 'say that', 'people who', 'more than', 'the ottoman', 'the whole', 'of his', 'to me', 'what you', 'of these', 'we have', 'was in', 'the city', 'in israel', 'able to']


# utils function to convert line to vector by feature
def convert_vector(line_split, feature):
    result = [0] * len(feature)  # init array of 0 with length of feature
    for x in line_split:
        try:
            result[feature.index(x)] = 1  # set 1 if item exist in feature
        except:
            pass  # keep 0 value
    return result


# utils function to convert line to bigram
def convert_bigram(line_split):
    tup = zip(line_split, line_split[1:])  # bigram as tuple
    return [t[0] + " " + t[1] for t in tup]  # return bigram as list of string


# utils function to get top 100 frequency word of line
def top_100_frequency_word(line_split):
    return [x[0] for x in Counter(line_split).most_common(100)]


# build vector 1 dataframe with feature of 200 frequency words as result of asm4
vec1_df = sql.createDataFrame(inp.map(lambda x: Row(label1=x, vector1=convert_vector(x, top_200_words))))


# build vector 2 dataframe with feature of 100 frequency bigrams as result of asm4
vec2_df = sql.createDataFrame(inp.map(convert_bigram).map(lambda x: Row(label2=x, vector2=convert_vector(x, top_100_bigrams))))


# to build vector 3 we should have a dataframe of top 100 frequency words each row
top_100_word_df = sql.createDataFrame(inp.map(top_100_frequency_word).map(lambda x: Row(label3=x)))


# Word2Vec training model for NLP & ML
w2v = Word2Vec(vectorSize=200, inputCol="label3", outputCol="vector3")
model = w2v.fit(sql.createDataFrame(inp.map(lambda x: Row(label3=x))))


# DOCUMENT: https://github.com/apache/spark/blob/c68f1a38af67957ee28889667193da8f64bb4342/mllib/src/main/scala/org/apache/spark/ml/feature/Word2Vec.scala#L258-L261
# Transform a sentence column to a vector column to represent the whole sentence.
# The transform is performed by averaging all word vectors it contains.
# transform to vector 3 - equivalent to average 100 vectors each row
vec3_df = model.transform(top_100_word_df)


# all solution have been consulted & confirmed by mentor thanhnh@funix.edu.vn
print("======== vector 1 ========")
vec1_df.show()


print("======== vector 2 ========")
vec2_df.show()


print("======== vector 3 ========")
vec3_df.show()


print("======== concatenate all 3 vectors ========")
vec1_df = vec1_df.withColumn("id", monotonically_increasing_id())
vec2_df = vec2_df.withColumn("id", monotonically_increasing_id())
vec3_df = vec3_df.withColumn("id", monotonically_increasing_id())

concat_vector = udf(lambda x, y: Vectors.dense(list(x)+list(y)), VectorUDT())

vec1_df \
    .join(vec2_df, "id") \
    .join(vec3_df, "id") \
    .withColumn("concat_temp", concat_vector("vector1", "vector2")) \
    .withColumn("concat", concat_vector("concat_temp", "vector3")) \
    .drop("concat_temp", "id") \
    .show()
