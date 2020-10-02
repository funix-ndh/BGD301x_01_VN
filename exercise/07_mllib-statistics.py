from pyspark import SparkContext
import numpy as np
from pprint import pprint

file_path = "kddcup.data_10_percent.gz"

sc = SparkContext("local", "lesson7")

raw_data = sc.textFile(file_path)

def parse_interactions(line):
    line_split = line.split(",")
    symbolic_indexes = [1, 2, 3, 41]
    clean_line_split = [item for i, item in enumerate(line_split) if i not in symbolic_indexes]
    return np.array([float(x) for x in clean_line_split])

vector_data = raw_data.map(parse_interactions)
pprint(vector_data)