from pyspark import SparkContext

sc = SparkContext("local", "lesson5")

file_path = "kddcup.data_10_percent.gz"

raw_data = sc.textFile(file_path)

csv_data = raw_data.map(lambda x: x.split(','))

normal_csv_data = csv_data.filter(lambda x: x[41] == 'normal.')
attack_csv_data = csv_data.filter(lambda x: x[41] != 'normal.')

normal_duration_data = normal_csv_data.map(lambda x: int(x[0]))
attack_duration_data = attack_csv_data.map(lambda x: int(x[0]))

total_normal_duration = normal_duration_data.reduce(lambda x, y: x+y)
total_attack_duration = attack_duration_data.reduce(lambda x, y: x+y)

print("total duration for 'normal' interactions is {}".format(total_normal_duration))

print("total duration for 'attack' interactions is {}".format(total_attack_duration))

# A better way, using `aggregate`
# The `aggregate` action frees us from the constraint of having the return be the same type as the RDD we are working on.
# Like with `fold`, we supply an initial zero value of the type we want to return.
# Then we provide two functions.
# The first one is used to combine the elements from our RDD with the accumulator.
# The second function is needed to merge two accumulators.
# Let's see it in action calculating the mean we did before.
normal_sum_count = normal_duration_data.aggregate(
    (0, 0),  # initial value as pair
    # combination with element in RDD
    lambda acc, value: (acc[0] + value, acc[1] + 1),
    lambda acc1, acc2: (acc1[0] + acc2[0], acc1[1] + acc2[1]),  # combine accumulators
)
print("Mean duration for 'normal' interactions is {}".
      format(round(normal_sum_count[0]/float(normal_sum_count[1]), 3)))
