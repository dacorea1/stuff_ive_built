#!/usr/bin/python3

"""
  This script is meant to iterate over and execute SQL files.
"""

import snowflake.connector
import os

from sys import exit, exc_info, path
from json import load
from datetime import datetime
from json import load, dumps
from requests import post
import argparse
import traceback

# Custom class with a series of utility functions for handling
# common tasks such as database connections, queries, and slack reporting
import data_sci_utilities as dsu

parser = argparse.ArgumentParser(description='All arguments required by script are listed in --help')
parser.add_argument('-f', '--filepath', dest='filepath',
                    help = 'path to analyst routine config file', 
                    type = str, 
                    default='/config_files/daily.json') # JSON config file pointing to a series of SQL scripts

args = parser.parse_args()


now = datetime.now()
print('Analyst Routine Initialized at {0}'.format(now))

# Global Vars
job_config = args.filepath

## Determine what machine this script is runnign on
def identify_machine():
  ###
  ## Identifies the current machine this script is running on
  ##  by checking the user
  ###
  current_user = os.popen('whoami').read().strip('\n')
  if current_user == 'deploy' or 'root':
    machine = 'analytics_remote'
    home = os.path.expanduser("~")
  elif current_user == 'davidcorea':
    machine = 'davids_personal'
    home = os.path.expanduser("~")    
  else:
    # Someone other than David Corea is running this etl locally on their box
    exit() # BYE

  return(machine, home)

# Get all the relevant credentials necessary for SQL execution
def get_creds(current_machine):
  if current_machine == 'analytics_remote':
    secrets_file = '/utils/secrets_file.txt'
  elif current_machine == 'davids_personal':
    secrets_file = '/secrets_file.json'
  else:
    # This codepath really should never be reached 
    # Someone other than David Corea is running this etl locally on their box
    print("Script is gonna kill itself now.")
    print("-----")
    print("If you need to run this, modify identify_machine() script to account for your user.")
    exit()

  with open(secrets_file) as sf:
    f = load(sf)

  snowflake_creds = f['snowflake']['analytics']
  aws_creds = f['creds']['aws']

  return(snowflake_creds, aws_creds, snowflake_temp_dir)
  

def connect_to_snowflake(snowflake_creds, snowflake_temp_dir):
  # Python API Docs:
  # https://docs.snowflake.net/manuals/user-guide/python-connector-install.html
  ctx = snowflake.connector.connect(
      user = snowflake_creds['user'],
      password = snowflake_creds['password'],
      account = snowflake_creds['account']
      )
  cs = ctx.cursor()

  # Define a temporary directory for Snowflake data to be intermittently written to
  temp_dir_command = 'export TMPDIR={0}'.format(snowflake_temp_dir)
  os.system(temp_dir_command)

  return(ctx, cs)


def execute_single_sql_statement(sql_statement, connector, cursor):
  try:
    print("Attempting to execute the following SQL statement:")
    print(sql_statement)
    cursor.execute(sql_statement)
    results = cursor.fetchall()
    print("Successfully executed SQL statement!")
  except:
    error_encountered = exc_info()[0].read()
    now = datetime.now()
    print("{1}\nError while running SQL statement:  {0}".format(error_encountered, now))
    print("Rolling transaction back")
    connector.rollback()

  return(results)


def get_array_of_jobs_to_run(job_config):
  with open(job_config) as djc:
    j = load(djc)

    files_to_read_and_execute = j['sql_files']

  return(files_to_read_and_execute)

def run_analyst_routine():
  current_machine, home = identify_machine()
  snowflake_creds, aws_creds, snowflake_temp_dir = get_creds(current_machine)
  conn_snow, cur_snow = connect_to_snowflake(snowflake_creds, snowflake_temp_dir)

  job_config_fullpath = home + job_config

  files_to_read_and_execute = get_array_of_jobs_to_run(job_config_fullpath)

  # Read the files to execute
  for table, path_to_file in files_to_read_and_execute.items():
    print("Attempting to run the SQL found at {0} to create/modify table {1}".format(path_to_file, table))
    
    current_sql_file_path = home + path_to_file

    try:
      with open(current_sql_file_path) as sf:
        sql_statements = sf.read().split(';')
      print('Successfully read in SQL file')
    except:
      print('Unable to read in file')
      print('Most likely situation is a wrong file path.  Please double check it :-D')
      print('MOVING ON TO THE NEXT SQL FILE.')
      error_message = {'text' : '*Analyst Routine Error*  Following SQL script failed to run: _{0}_'.format(current_sql_file_path)}
      dsu.send_errors_to_slack(error_message)
      next

    # Run only one SQL statement at a time
    for sql in sql_statements:
      print('Attempting to run following SQL statement:\n{0}'.format(sql))
      try:
        execute_single_sql_statement(sql, conn_snow, cur_snow)
      except:
        error_encountered, error_value, error_traceback = exc_info()
        now = datetime.now()
        error_message = "Error encountered while running SQL statement:\nTime of Error:\n{0}\n Error Stacktrace:\n{1}".format(now,  traceback.format_exception(error_encountered, error_value, error_traceback)  )
        slack_error_message = {'text' : '*Analyst Routine Error*\n{0}'.format(error_message)}

  now = datetime.now()
  print('Analyst Routine Completed at {0}\n\n'.format(now))

if __name__ == "__main__":
  run_analyst_routine()