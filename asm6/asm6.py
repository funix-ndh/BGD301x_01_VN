#########################################################################################
#                                   ASM6 - FX03393                                      #
#     all solutionS have been consulted & confirmed by mentor thanhnh@funix.edu.vn      #
#########################################################################################
import re
from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext
from pyspark.sql.functions import input_file_name, udf, size
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import IndexToString, StringIndexer, CountVectorizer
from pyspark.ml.feature import Word2Vec
from pyspark.sql.types import ArrayType, StringType

#########################################################################################
#                                       SET UP                                          #
#########################################################################################

conf = SparkConf() \
    .setAll([('spark.executor.memory', '16g'), ('spark.driver.memory', '16g')])

sc = SparkContext("local", "asm6", conf=conf)
sc.setLogLevel("WARN")

sql = SQLContext(sc)


#########################################################################################
#                                   LOAD DATAFRAME                                      #
#########################################################################################
# note:
# because assignment submission only allow 20MB
# make sure to download `input data` file:
#   http://qwone.com/~jason/20Newsgroups/
# and locate it inside `20news-18828` directory


# loop & read all directories
input_base_path = "20news-18828/*"


# util function to normalize case
def normalize_case(line):
    return re.sub(r'[^a-z]', ' ', line.lower().strip()).strip()


# util function to split sentence to word and also strip out empty word
def sanity_and_split(line):
    return list(filter(lambda x: "" != x, line.split(" ")))


# util function to get parent folder from file path
def get_folder(path):
    path_split = path.split("/")
    return path_split[len(path_split) - 2]


normalize_case_udf = udf(normalize_case, StringType())
sanity_and_split_udf = udf(sanity_and_split, ArrayType(StringType()))
get_folder_udf = udf(get_folder, StringType())


# load data to df
df = sql.read.text(input_base_path)

#########################################################################################
#                                     DEBUG MODE                                        #
#########################################################################################
debug_mode = True
if debug_mode:
    df = df.sample(withReplacement=False, fraction=0.005)
#########################################################################################


df = df.withColumnRenamed("value", "features") \
    .withColumn("features", normalize_case_udf("features")) \
    .withColumn("features", sanity_and_split_udf("features")) \
    .withColumn("label", input_file_name()) \
    .withColumn("label", get_folder_udf("label")) \
    .where(size("features") > 0)


# split 3 parts
# training 60% - validation 20% - testing 20%
training_df, validation_df, testing_df = df.randomSplit([0.6, 0.2, 0.2])


#########################################################################################
#                                   STEP 1, 2 ,3                                        #
#########################################################################################
# label indexer
labelIndexer = StringIndexer(inputCol="label",
                             outputCol="indexedLabel").fit(df)


# features indexer
featureIndexer = CountVectorizer(inputCol="features",
                                 outputCol="indexedFeatures").fit(df)


# config random forest classifier
rf = RandomForestClassifier(labelCol="indexedLabel",
                            featuresCol="indexedFeatures",
                            maxDepth=10,
                            numTrees=10)


# convert back to original label
labelConverter = IndexToString(inputCol="prediction",
                               outputCol="predictedLabel",
                               labels=labelIndexer.labels)


# set up train pipeline
pipeline = Pipeline(stages=[labelIndexer, featureIndexer, rf, labelConverter])


# train random forest model
rf_model = pipeline.fit(training_df)


# judge on validation
predictions = rf_model.transform(validation_df)
predictions.select("predictedLabel", "label", "features").show(5)
evaluator = MulticlassClassificationEvaluator(labelCol="indexedLabel",
                                              predictionCol="prediction",
                                              metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Validation Test Error = %g" % (1.0 - accuracy))


# judge on real testing
predictions = rf_model.transform(testing_df)
predictions.select("predictedLabel", "label", "features").show(5)
evaluator = MulticlassClassificationEvaluator(labelCol="indexedLabel",
                                              predictionCol="prediction",
                                              metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Real Test Error = %g" % (1.0 - accuracy))


#########################################################################################
#                                   STEP 4.1                                            #
#                                 CUSTOM VECTOR                                         #
#########################################################################################
# using Word2Vec
# to natural language processing and machine learning process.
w2v = Word2Vec(vectorSize=400,
               inputCol="features",
               outputCol="indexedFeatures_v2")
w2v_model = w2v.fit(df)
w2v_df = w2v_model.transform(df)


# split 3 parts
# training 60% - validation 20% - testing 20%
w2v_training_df, w2v_validation_df, w2v_testing_df = \
    w2v_df.randomSplit([0.6, 0.2, 0.2])


# config random forest classifier
rf_v2 = RandomForestClassifier(labelCol="indexedLabel",
                               featuresCol="indexedFeatures_v2",
                               maxDepth=10,
                               numTrees=10)


# set up train pipeline
pipeline_v2 = Pipeline(stages=[labelIndexer, rf_v2, labelConverter])


# train random forest model
rf_v2_model = pipeline_v2.fit(w2v_training_df)


#########################################################################################
#                                   STEP 4.2                                            #
#                              NEW VECTOR PREDICTION                                    #
#########################################################################################
# judge on validation
predictions = rf_v2_model.transform(w2v_validation_df)
predictions.select("predictedLabel", "label", "features").show(5)
evaluator = MulticlassClassificationEvaluator(labelCol="indexedLabel",
                                              predictionCol="prediction",
                                              metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Validation Test Error = %g" % (1.0 - accuracy))


# judge on real testing
predictions = rf_v2_model.transform(w2v_testing_df)
predictions.select("predictedLabel", "label", "features").show(5)
evaluator = MulticlassClassificationEvaluator(labelCol="indexedLabel",
                                              predictionCol="prediction",
                                              metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Real Test Error = %g" % (1.0 - accuracy))
