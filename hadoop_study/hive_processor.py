from pyhive import hive
from TCLIService.ttypes import TOperationState
import freep
import numpy as np
import pandas as pd
import random
import csv
import json
import time

from freep.utils.preference_processor import is_parameter_in_preferences
from freep.utils.preference_processor import parameters_in_preferences
from freep.utils.preference_processor import parameter_from_encoded_parameter
from freep.utils.preference_processor import get_preferences_for_partition
from freep.utils.encode_decode_processor import encode
from freep.partitioners.commons import all_columns_present
from freep.partitioners import pca
from freep.recommenders import classifier
from freep.utils.commands import run_cmd
from freep.utils.log import setup_custom_logger
logging = setup_custom_logger('root')
from constants import Constants

class HiveProcessor():

    def __init__(self):
        self.connection = hive.Connection(host=Constants.HIVE_HOST, 
                                          port=Constants.HIVE_PORT, 
                                          username=Constants.HIVE_USERNAME)
        self.cursor = self.connection.cursor()

    def generate_orc_table(self, header):
        external_table = "CREATE TABLE freep.orc_table "+\
    "({}) ".format(','.join(["%s STRING" % x for x in header]))+\
    "ROW FORMAT DELIMITED "+\
    "STORED AS ORC"
        return external_table

    def generate_external_table(self, header, location):
        external_table = "CREATE EXTERNAL TABLE IF NOT EXISTS freep.csv_table " + \
    "({}) ".format(','.join(["%s STRING" % x for x in header])) + \
    "ROW FORMAT DELIMITED " + \
    "FIELDS TERMINATED BY ',' " + \
    "STORED AS TEXTFILE " + \
    "LOCATION '{}' ".format(location) +\
    "TBLPROPERTIES (\"skip.header.line.count\"=\"1\")"
        return external_table 

    def load_data_to_hive(self, columns):

        logging.debug('Criando base no HIVE...')
        self.cursor.execute(Constants.HIVE_DROP_DATABASE)
        self.cursor.execute(Constants.HIVE_CREATE_DATABASE)
        logging.debug('Gerando tabela externa...')
        self.cursor.execute(self.generate_external_table(columns, Constants.HIVE_LOCATION))
        logging.debug('Gerando tabela ORC...')
        self.cursor.execute(self.generate_orc_table(columns))
        # logging.debug('Realizando consulta...')
        # self.cursor.execute('SELECT * FROM freep.csv_table LIMIT 10')
        #logging.debug(self.cursor.fetchone())

        logging.debug('Transferindo dados...')
        self.cursor.execute(Constants.HIVE_EXTERNAL_TO_ORC_QUERY)
        # logging.debug('Realizando consulta...')
        # cursor.execute('SELECT * FROM freep.orc_table LIMIT 10')
        # logging.debug(cursor.fetchall())
        logging.debug('Finalizando...')
    
    def query_orc_table(self, columns, filter):
        self.cursor.execute('SELECT {} \
                        FROM freep.orc_table\
                        WHERE {}'.format(columns, filter))
        
        values = []
        while True:
            row = self.cursor.fetchone()
            if row == None:
                break
            values.append(row)
        return values

    def close(self):
        self.cursor.close()
        self.connection.close()
