#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
  Script models a customer requiring a professional's service as a random, binomial variable
  'Professional' is defined as any business owner (coffee shop owner, restaurant owner, etc) w

  prob(returning customer) ~ Binom(n, p, k)
  p: prob(success) of repeat customer returning to business, defined by repeat customer frequency of that business
  k: prob(X = k) for vector k defined by num months of life for a Pro.  For this script's purposes, minimum K set at 18
  n: # expected return visits (trials) per month, defined by business' avg num customers per month


"""
import os
from sys import exit, path, exc_info
from datetime import datetime
from json import dumps, load
from requests import post

import pandas as pd
import numpy as np
from scipy.stats import binom

from data_sci_utilities import identify_machine, get_creds, connect_to_database, traceback, execute_single_sql_statement, copy_from_s3_into_database

def identify_churn_cutoff_month(pro_data_frame):
  """
  Script identifies the month at which the probability of a repeat customer returning to a Pro drops the most
  """
  pro = pro_data_frame
  n = int(pro.mean_num_jobs_per_month.values[0])
  p = round(float(pro.repeat_customer_perc.values[0]), 3)
  num_months_in_business = pro.num_months_pro_got_paid.values[0]
  #
  if num_months_in_business < 18:
    k = range(0,19)
  else:
    k = range(0,num_months_in_business)
  #
  binom_prob_distr = binom.pmf(k, n, p)
  #
  # Compute which month has the largest probability drop
  #
  # list of probability changes between months
  prob_changes = [ ((p - binom_prob_distr[i+1])) for i, p in enumerate(binom_prob_distr) if i < (len(binom_prob_distr) - 1) ]
  max_prob_change_index = prob_changes.index(max(prob_changes)) # this is the month before the biggest drop
  #
  return(binom_prob_distr, n, p,  max_prob_change_index)


# Store this data in database 
def load_into_database(file_directory, filename, s3_location, database_table):
  generic_database_loader_system_command = 'python3 /scripts/generic_database_loader.py -csv {0} -file {1} -s3 {2} -snow {3}'.format(file_directory, filename, s3_location, database_table)
  os.system(generic_database_loader_system_command)

current_machine = identify_machine()
mysql_creds, database_creds, aws_creds, database_temp_dir = get_creds(current_machine)

print("Attempting to connect to database")
conn_snow, cur_snow = connect_to_database(database_creds, database_temp_dir)
print("Successfully connected")

# Fetch data from database
customer_ltv_raw_sql = 'SELECT * FROM initial_dataset'
customer_ltv_raw = execute_single_sql_statement('database', 'select', customer_ltv_raw_sql, conn_snow, cur_snow)
conn_snow.close()

customer_ltv_column_names = ['user_id', 
                        'mean_monthly_net_rev_per_customer', 
                        'repeat_customer_perc', 
                        'num_months_pro_got_paid',
                        'mean_num_jobs_per_month'
                      ]
customer_ltv_df = pd.DataFrame(customer_ltv_raw, columns = customer_ltv_column_names)


# Iterate over each org (1 row per org in dataset)
prob_distributions = []
churn_cutoffs = []
repeat_customer_perc = []
mean_num_jobs_per_month = []
unique_orgs = customer_ltv_df.user_id.unique()

for k, org in enumerate(unique_orgs):
  print(k, org)
  pro_df = customer_ltv_df[customer_ltv_df.user_id == org]
  binom_prob_distr, n , p, churn_cutuff_month = identify_churn_cutoff_month(pro_df)
  churn_cutoffs.append(churn_cutuff_month)
  binom_def_string = 'binom({0}, {1})'.format(n, p)
  binom_distr = list(binom_prob_distr)
  binom_dict = dict({binom_def_string : binom_distr})
  prob_distributions.append(binom_dict)
  repeat_customer_perc.append(p)
  mean_num_jobs_per_month.append(n)

customer_ltv_df['churn_cutoff_month'] = churn_cutoffs
# customer_ltv_df['prob_customer_returning_to_pro'] = prob_distributions


csv_filename = 'customer_churn_cuttoff_months.csv'
csv_filepath = '/datasets/customer_ltv'
s3_bucket = 's3://.../model_output/customer_retention_prob_after_n_months'
database_table_name_to_load_results_into = 'some_db_table'
csv_full_filepath = csv_filepath + '/' + csv_filename
customer_ltv_df.to_csv(csv_full_filepath, index = False)


load_into_database(csv_filepath, csv_filename, s3_bucket, database_table_name_to_load_results_into)
