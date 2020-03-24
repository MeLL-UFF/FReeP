#!/usr/local/bin/python3.5

# SEND DATA TO HDFS
# import os
# import pyarrow as pa

# lib_dir = os.getenv('ARROW_LIBHDFS_DIR')
# fs = pa.hdfs.connect()
# input_data = 'data.csv'
# with open(input_data) as f:
#     pa.hdfs.HadoopFileSystem.upload(fs, '/user/freep/data/data.csv', f)

# print(fs.ls('/data/'))

# alias hadoop_streaming='/usr/local/hadoop/share/hadoop/tools/lib/hadoop-streaming-2.7.1.jar'
# hadoop fs -ls

import sys
import pandas as pd

def read_input(file):
    for line in file:
        # split the line into words
        yield line.split()

def main():
    rows = sys.stdin.readlines()
    # input comes from STDIN (standard input)
    # data = pd.read_csv(sys.stdin)
    for row in rows:
        print(row)
        # write the results to STDOUT (standard output);
        # what we output here will be the input for the
        # Reduce step, i.e. the input for reducer.py
        #
        # tab-delimited; the trivial word count is 1
        # for word in words:
        #     print '%s%s%d' % (word, separator, 1)

if __name__ == "__main__":
    main()

# hadoop fs -mkdir -p /user/freep/input
# hadoop fs -put /app/data.csv /user/freep/input

# hadoop jar /usr/local/hadoop/share/hadoop/tools/lib/hadoop-streaming-*.jar \
# -file /app/recommender_mapper.py \
# -mapper /app/recommender_mapper.py \
# -file /app/recommender_reducer.py  \
# -reducer /app/recommender_reducer.py  \
# -input /user/freep/input/data.csv \
# -output /user/freep/output

# hadoop fs -rm -r /user/freep/output