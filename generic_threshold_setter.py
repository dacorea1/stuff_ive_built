"""
  This script acts as an "outlier detector" via
  simulating a normal distribution with a bootsrapped derived mean

  A historical db table contains some data.
  Script generates boundary thresholds, beyond which outliers can be identified
"""

import os
from sys import exit, exc_info, path
from datetime import datetime, date, timedelta
import pandas as pd
from json import load,dumps
from requests import post
import traceback
import argparse

from sklearn.utils import resample
import numpy as np

import data_sci_utilities as dsu


parser = argparse.ArgumentParser(description='All arguments required by script are listed in --help')
parser.add_argument('-name', '--metric_name', dest='metric_name',
                        help = 'name of the metric', 
                        type = str)
parser.add_argument('-ht', '--historical_table', dest='historical_table',
                        help = 'name of the database table with historical data', 
                        type = str)
parser.add_argument('-at', '--all_thresholds', dest='all_thresholds',
                        help = 'name of the database table with historical threshold values', 
                        type = str)
parser.add_argument('-cn', '--column_names', dest='column_names',
                        help = 'comma separated string of column names.\nIe \'name1, name2, etc\' ', 
                        type = str)
parser.add_argument('-pc', '--partition_column', dest='partition_column',
                        help = 'Name of the column for which alert thresholds should be partitioned by.  Ie Thresholds could vary by day of week, day of month, etc', 
                        type = str)
parser.add_argument('-ac', '--aggregation_columns', dest='aggregation_columns',
                        help = 'Name of the column to be aggregated to determine threshold for triggering alert.  Ie Number of Users, Total Dollars, etc', 
                        type = str)
parser.add_argument('-sd', '--standard_deviations', dest='standard_deviations',
                        help = 'Number of Standard Deviations Above or Below Average where threshold should be set', 
                        type = int)
parser.add_argument('-l', '--low', dest='alert_type',
                        help = 'Alert threshold should be a lower bound.\nIe Alert should trigger if metric is below a certain number. (default True)', 
                        action = 'store_true')
args = parser.parse_args()

current_machine = dsu.identify_machine()
mysql_creds, database_creds, aws_creds, database_temp_dir = dsu.get_creds(current_machine)

print("Attempting to connect to database")
conn_snow, cur_snow = dsu.connect_to_database(database_creds, database_temp_dir)
print("Successfully connected")

# Global vars
metric_name = args.metric_name
historical_table = args.historical_table
column_names = args.column_names.replace(" ", "").split(",")
partition_column = args.partition_column.replace(" ", "").split(",")
aggregation_columns = args.aggregation_columns.replace(" ", "").split(",")
standard_deviations = args.standard_deviations
database_historical_thresholds_table = args.all_thresholds
alert_type = args.alert_type


## Tell someone that something related to alerts is happening
log_message = {'text': 'System Alerter Threshold Setter script invoked for {metric_name}. Thresholds will be generated and written to {threshold_table}'.format(metric_name = metric_name, threshold_table = database_historical_thresholds_table)} 
dsu.send_logs_to_slack(log_message, is_data_sci_alert = True)


def query_for_historical_data(historical_table):
  sql_to_run = 'SELECT * FROM {0}'.format(historical_table)
  results = dsu.execute_single_sql_statement('database', 'select', sql_to_run, conn_snow, cur_snow)
  return(results)

results = query_for_historical_data(historical_table)
# storing [r]esults in [d]ata [f]rame
rdf = pd.DataFrame(results, columns = column_names)


def generate_alert_thresholds(agg_column):
  """
  This method applies the `determine_lower_threshold function` to all the data associated
  with each unique element in the column passed as the argument to this function
  """
  def determine_lower_threshold(x):
    """
    Takes a numpy array and resamples with replacement to produce a gaussian representation of source data
    via bootsrap resampling
    
    Computes a lower threshold, defined at n standard deviations of bootsrap mean statistic
    """
    va = x
    # The size of the dataset determines the number of samples to take from the dataset
    boot_resample = [ np.mean(resample(va, n_samples = int(rdf.shape[0] * 1/3), replace = True)) for num in range(0,rdf.shape[0]) ]
    
    # By default, using  average - N Std Devs  as a lower threshold for trigger an alert
    if alert_type:
      alert_threshold = int(np.mean(boot_resample) - (standard_deviations * np.std(boot_resample)))
      # This is likely the floor for bottom thresholds we care about for business metrics
      if alert_threshold < 0:
        alert_threshold = 0
      else:
        alert_threshold = alert_threshold
    # If a metric requires an upper threshold, 
    #  using  average + N Std Devs  instead
    else:
      alert_threshold = int(np.mean(boot_resample) + (standard_deviations * np.std(boot_resample)))
    return(alert_threshold)
    
  series_thresholds = rdf.groupby(partition_column)[agg_column].agg(determine_lower_threshold)
  return(series_thresholds)


# Storing alerting threshold output into a dictionary to load into a new dataframe
thresholds_dict = {}

for col in aggregation_columns:
  thresholds_dict[col] = generate_alert_thresholds(col)

td = str(date.today())
dt = datetime.strptime(td, '%Y-%m-%d')
week_start = (dt - timedelta(days = dt.weekday())).date()


result_df_w_thresholds = pd.DataFrame(thresholds_dict)
result_df_w_thresholds['week_start_date'] = week_start
result_df_w_thresholds.reset_index(inplace = True)

###
# Write resulting dataframe to a file
dir_path = '/alerts/temp_files'
filename = 'threshold_df_for_' + metric_name.lower().replace(' ', '') + td + '.csv'
path_to_file = dir_path + '/' + filename
s3bucket = 's3://{s3_bucket_name}/alerting'.format('s3_bucket_name')
database_loader = '/scripts/generic_database_loader.py'

# Append data to historical database table
result_df_w_thresholds.to_csv(path_to_file, index = False)
#
#
append_to_database_command = 'python3 {database_loader} -csv {dir_path} -file {filename} -s3 {s3bucket} -snow {database_historical_thresholds_table} -a'.format(database_loader= database_loader,
                                                                                                                                                                   dir_path = dir_path,
                                                                                                                                                                   filename = filename,
                                                                                                                                                                   s3bucket = s3bucket,
                                                                                                                                                                   database_historical_thresholds_table = database_historical_thresholds_table
                                                                                                                                                                  )
os.system(append_to_database_command)


conn_snow.close()


