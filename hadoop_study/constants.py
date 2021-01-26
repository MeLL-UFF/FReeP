class Constants():
    HDFS_PATH = '/user/freep/'
    HDFS_INPUT_PATH = HDFS_PATH +'input/'
    HDFS_OUTPUT_PATH = HDFS_PATH + 'output/'
    
    HIVE_LOCATION = 'hdfs://quickstart.cloudera:8020'+HDFS_INPUT_PATH
    HIVE_DROP_DATABASE = "DROP SCHEMA IF EXISTS freep CASCADE"
    HIVE_CREATE_DATABASE = 'CREATE SCHEMA IF NOT EXISTS freep'
    HIVE_EXTERNAL_TO_ORC_QUERY = 'INSERT INTO TABLE freep.orc_table SELECT * FROM freep.csv_table'
    HIVE_USERNAME = 'root'
    HIVE_HOST='localhost'
    HIVE_PORT=10000