import csv
from pyhive import hive
import pydoop.hdfs as hdfs
import subprocess
import logging

logging.basicConfig(
    format='%(levelname)s:%(asctime)s: %(message)s', datefmt='%d/%m/%Y %H:%M:%S')
logging.getLogger().setLevel(logging.DEBUG)

def run_cmd(args_list):
    """
    run linux commands
    """
    # import subprocess
    print('Running system command: {0}'.format(' '.join(args_list)))
    proc = subprocess.Popen(
        args_list, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    s_output, s_err = proc.communicate()
    s_return = proc.returncode
    return s_return, s_output, s_err

DROP_DATABASE = "DROP SCHEMA IF EXISTS freep CASCADE"
CREATE_DATABASE = 'CREATE SCHEMA IF NOT EXISTS freep'
TRANSFER_QUERY = 'INSERT INTO TABLE freep.orc_table SELECT * FROM freep.csv_table'

def generate_orc_table(header):
	external_table = "CREATE TABLE freep.orc_table "+\
"({}) ".format(','.join(["%s STRING" % x for x in header]))+\
"ROW FORMAT DELIMITED "+\
"STORED AS ORC"
	return external_table

def generate_external_table(header, location):
	external_table = "CREATE EXTERNAL TABLE IF NOT EXISTS freep.csv_table " + \
"({}) ".format(','.join(["%s STRING" % x for x in header])) + \
"ROW FORMAT DELIMITED " + \
"FIELDS TERMINATED BY ',' " + \
"STORED AS TEXTFILE " + \
"LOCATION '{}' ".format(location) +\
"TBLPROPERTIES (\"skip.header.line.count\"=\"1\")"
	return external_table 


logging.debug('Removendo dados de execução anterior...')
(ret, out, err) = run_cmd(['hadoop', 'fs', '-rm','-r','/user/freep'])
input_path = '/user/freep/input/'
logging.debug('Recriando diretórios...')
hdfs.mkdir(input_path)
logging.debug('Copiando dados para HDFS...')
hdfs.put('data.csv', input_path)

location = 'hdfs://quickstart.cloudera:8020'+input_path
with open('data.csv','r') as opened_in_file:
	reader = csv.DictReader(opened_in_file)
	header = reader.fieldnames

conn = hive.Connection(host="localhost", port=10000, username="cloudera")
cursor = conn.cursor()
logging.debug('Criando base no HIVE...')
cursor.execute(DROP_DATABASE)
cursor.execute(CREATE_DATABASE)
logging.debug('Gerando tabela externa...')
cursor.execute(generate_external_table(header, location))
logging.debug('Gerando tabela ORC...')
cursor.execute(generate_orc_table(header))
logging.debug('Realizando consulta...')
cursor.execute('SELECT * FROM freep.csv_table LIMIT 10')
logging.debug(cursor.fetchone())

logging.debug('Transferindo dados...')
cursor.execute(TRANSFER_QUERY)
logging.debug('Realizando consulta...')
cursor.execute('SELECT * FROM freep.orc_table LIMIT 10')
logging.debug(cursor.fetchall())
logging.debug('Finalizando...')