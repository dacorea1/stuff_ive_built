#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Lead Win-Propensity Model

This model seeks to predict the propensity
of a pro enrolling into a paid Housecall Pro
plan, given interactions with the Marketing site,
various Marketing channels, the Sales organization,
and event temporal dynamics.

For additional info, see David Corea in Data Science

@author: davidcorea
Created on Sep 2019
"""

import pandas as pd
import numpy as np
from joblib import dump

import lightgbm as lgb
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.calibration import calibration_curve
from skopt.space import Real, Integer
from skopt.utils import use_named_args
from skopt import gp_minimize
from skopt.plots import plot_convergence
from inspect import signature
import matplotlib.pyplot as plt

import os
from sys import exit, exc_info, path, stdout
import traceback
from datetime import datetime, date
from json import loads,dumps
import requests

path.insert(0, '/home/deploy/core-analytics/data_sci_projects')
import data_sci_utilities as dsu

# This training scipt is likely to trigger an alarm on the playground ec2 box
# Data Eng will want a heads up
model_training_message = {
    "text": ":robot_face: Friendly Alert from Data Science",
    "attachments": [
        {
            "title": "A new lead scoring model is being trained",
            "fields": [
                {
                    "value": "Cross validation is a part of this process and may trigger a CPU usage alarm.\nThanks for understanding.",
                    "short": True
                }
            ]
        }
    ]
}

dsu.send_logs_to_slack(model_training_message)

###
## Connect to Snowflake and fetch data
###
current_machine = dsu.identify_machine()
mysql_creds, snowflake_creds, aws_creds, snowflake_temp_dir = dsu.get_creds(current_machine)

# Important for this list to be globally acccessible in this script
feature_column_names = ['pro_email',
                        'industry',
                        'most_recent_marketing_channel',
                        'state',
                        'ltv',
                        'org_size',
                        'num_housecall_pro_site_pageviews',
                        'num_employees_in_org',
                        'num_pros_to_login',
                        'num_housecall_pro_site_logins',
                        'days_since_trial_expired',
                        'days_since_web_form_submission',
                        'num_housecall_pro_site_logouts',
                        'pro_connected_qbo',
                        'num_email_opens',
                        'num_email_sends',
                        'num_email_unsubscribe_requests',
                        'days_since_last_email_unsubscribe',
                        'num_email_spam_flags',
                        'total_calls_made_to_pro',
                        'days_since_last_call',
                        'days_since_last_sms',
                        'num_connected_calls',
                        'num_conversations',
                        'prop_conversations',
                        'num_hcp_reps_interacted_w_pro',
                        'num_sms_messages_sent_to_pro',    
                        'attended_demo',
                        'duration_in_days_between_booking_and_attending_demo',
                        'total_touches',
                        'num_marketing_touches',
                        'num_unique_marketing_channels',
                        'num_jobs_created_first_14_days',
                        'num_estimates_created_first_14_days', 
                        'num_jobs_scheduled_first_14_days',
                        'scheduled_job_amount_first_14_days',
                        'scheduled_estimate_amount_first_14_days',
                        'num_estimated_jobs_first_14_days',
                        'num_inprogress_jobs_first_14_days',
                        'inprogress_job_amount_first_14_days',
                        'num_completed_jobs_first_14_days',
                        'completed_job_amount_first_14_days',
                        'num_estimates_completed_first_14_days',
                        'completed_estimates_amount_first_14_days',
                        'num_jobs_w_onmyway_sent_first_14_days',
                        'num_estimates_w_onmyway_sent_first_14_days',
                        'num_jobs_w_invoice_sent_first_14_days',
                        'num_ios_jobs_first_14_days',
                        'ios_job_amount_first_14_days',
                        'num_android_jobs_first_14_days',
                        'android_job_amount_first_14_days',
                        'num_online_booking_jobs_first_14_days',
                        'online_booking_job_amount_first_14_days',
                        'yelp_job_count_first_14_days',
                        'yelp_job_amount_first_14_days',
                        'mu',
                        'alpha',
                        'theta',
                        'booked_demo',
                        'pro_enrolled']

categorical_feature_list = ['industry', 'state', 'most_recent_marketing_channel']

# Setting this as global param for easy logging / easy future tweaking 
num_cv_folds = 7

def fetch_data_from_snowflake(feature_column_names):
  """
  Fetches dataset from Snowflake view
  DATAZOO.mod_out.lead_scoring_features
  """

  snowflake_view = 'DATAZOO.mod_out.lead_scoring_features'
  sql_statement = 'SELECT * FROM {0}'.format(snowflake_view)
  sql_results = dsu.execute_single_sql_statement('snowflake', 'select', sql_statement, conn_snow, cur_snow)
  sql_results_df = pd.DataFrame(sql_results, columns = feature_column_names)

  conn_snow.close()
  return(sql_results_df)

"""
Loading data into memory from Snowflake
"""
print("Attempting to connect to Snowflake")
conn_snow, cur_snow = dsu.connect_to_snowflake(snowflake_creds, snowflake_temp_dir)
print("Successfully connected")

# [f]eature [d]ata [f]rame
print('Querying Snowflake for lead scoring features')
fdf = fetch_data_from_snowflake(feature_column_names)
conn_snow.close()


def all_the_data_preprocessing(cat_feat_list = categorical_feature_list, fdf = fdf, feature_column_names = feature_column_names):
  """
    Method applies relevant pre processing to 
    data retrieved from Snowflake

    When this method is completed, data is prepped
    for model development

    Returns a model ready data frame
  """

  def replace_bottom_fifty_percent_of_verticals_w_minority_string(fdf):
    """
    This method identifies bottom 50% of verticals and renames them all to 'minority vertical'
    Opting to do this as these are verticals which mostly are just noise in the data.
    A pro can enter free form text as his / her industry if he / she doesn't opt for the pre determined choices:
      "HVAC, Carpet Cleaning, etc".

    This method seeks to sanitize what are likely the low volume, free form entries
    """

    industry_counts_df = pd.DataFrame(fdf.industry.value_counts())
    industry_ct_quantiles = industry_counts_df.quantile(q = list(np.arange(0.,1.05,.05)))

    quantile_assignments = pd.DataFrame(
                              pd.cut(industry_counts_df.industry, industry_ct_quantiles.industry, 
                                labels = False, 
                                duplicates = 'drop',  # handy method for handling duplicate values among the quantiles
                                retbins = True,
                                right = False
                              )[0] # cut returns a tuple.  All the quantiles are in the 1st element
                            )

    # Extreme values within the intervals might not get labeled.  This is a catch all
    quantile_assignments.fillna(value = 20.0, inplace = True)
    quantile_assignments.rename(columns = {'industry' : 'industry_quantile'}, inplace = True)

    fdf_prime = fdf.merge(quantile_assignments, how = 'left', left_on = 'industry', right_index = True)
    
    # Everything under 50th percentile, just rename to 'minority vertical'
    fdf_prime.loc[fdf_prime['industry_quantile'] <= 10, 'industry'] = 'Minority Vertical'
    fdf_prime.drop(columns = ['industry_quantile'], inplace = True)

    return(fdf_prime)

  def replace_bottom_fifty_percent_of_channels_w_minority_string(fdf):
    """
    This method identifies bottom 50% of marketing channels and renames them all to 'minority channel'
    Opting to do this as many of the channel names are unsanitized and at times meaningless to even the
    marketing team.  The channels that carry substantial lead volume are important to the marketing team,
    and the long tail of unsanitized marketing channel names with low volume are often forgotten to even exist.

    This method seeks to sanitize what are likely the low volume, forgotten, and meaningless channel names
    """

    channel_counts_df = pd.DataFrame(fdf.most_recent_marketing_channel.value_counts())
    channel_ct_quantiles = channel_counts_df.quantile(q = list(np.arange(0.,1.05,.05)))

    quantile_assignments = pd.DataFrame(
                              pd.cut(channel_counts_df.most_recent_marketing_channel, 
                                channel_ct_quantiles.most_recent_marketing_channel, 
                                labels = False, 
                                duplicates = 'drop',  # handy method for handling duplicate values among the quantiles
                                retbins = True,
                                right = False
                              )[0] # cut returns a tuple.  All the quantiles are in the 1st element
                            )

    # Extreme values within the intervals might not get labeled.  This is a catch all
    quantile_assignments.fillna(value = 20.0, inplace = True)
    quantile_assignments.rename(columns = {'most_recent_marketing_channel' : 'most_recent_marketing_channel_quantile'}, inplace = True)

    fdf_prime = fdf.merge(quantile_assignments, how = 'left', left_on = 'most_recent_marketing_channel', right_index = True)

    # Everything under 50th percentile, just rename to 'minority channel'
    fdf_prime.loc[fdf_prime['most_recent_marketing_channel_quantile'] <= 10, 'most_recent_marketing_channel'] = 'Minority Channel'
    fdf_prime.drop(columns = ['most_recent_marketing_channel_quantile'], inplace = True)

    return(fdf_prime)

  fdf = replace_bottom_fifty_percent_of_verticals_w_minority_string(fdf)
  fdf = replace_bottom_fifty_percent_of_channels_w_minority_string(fdf)

  # Additional trivial clean up
  fdf.state = fdf.state.str.lower().fillna('unknown')
  fdf.industry = fdf.industry.str.lower().fillna('unknown')
  fdf.most_recent_marketing_channel = fdf.most_recent_marketing_channel.str.lower().fillna('unknown')

  def add_an_unseen_catch_all_to_all_categorical_fields(fdf):
    """
    During prediction time, unseen data will cause the model to error out. 
    Production model will incorporate a simple transformation replacing new verticals with 'unseen'
    This method introduces 'unseen' as an observation within the data to ensure graceful model predictions,
    even on truly unseen data

    Stealing this approach shamelessly from Olaf Weid
    """

    # Categorical features undergoing minor transform in prep for model development
    none_of_the_categorical_features = list( set(feature_column_names) - set(categorical_feature_list) )

    # Storing all data except the categorical features in an array
    # Converted categorical features will be concatenated into this list
    # after appropriate type casting operation
    all_features_except_categorical = [fdf[none_of_the_categorical_features]]

    # Following step ensures model prediction will never fail on unseen categorical features
    # Shamelessly stealing from Olaf
    cat_dict = {}
    for cat in categorical_feature_list:
      cat_dict[cat] = list(fdf[cat].unique()) + ['unseen']
    
    # Need to convert categorical features to categorical type for lgbm
    categorical_feature_collection = []
    for cat in categorical_feature_list:
      categorical_feature_collection.append(pd.Series(
                                                      pd.Categorical(fdf[cat].fillna('unseen'), 
                                                                     categories = cat_dict[cat]
                                                                    ),
                                                      name = cat
                                                     )
                                           )

    # Bringing all the features back together and ensuring only complete cases 
    # remain for model development
    model_ready_df = pd.concat( (all_features_except_categorical + categorical_feature_collection), 
                                axis = 1, levels = None
                              )
    model_ready_df = model_ready_df.loc[model_ready_df.notnull().all(axis = 1)]

    # Target feature vector that I'm trying to predict
    known_pro_enrollments = model_ready_df[['pro_enrolled', 'pro_email']]
    del model_ready_df['pro_enrolled'] # shouldn't be in model ready data frame

    return(model_ready_df, known_pro_enrollments, cat_dict, fdf)

  model_ready_df, known_pro_enrollments, cat_dict, fdf = add_an_unseen_catch_all_to_all_categorical_fields(fdf)

  # Adding retroactively.  "Prop Conversations" feature gets read in as an object by default for some reason
  model_ready_df.prop_conversations = model_ready_df.prop_conversations.astype('float')

  return(model_ready_df, known_pro_enrollments, cat_dict, fdf) 

def model_param_search_and_fitting(fdf, model_ready_df, known_pro_enrollments, num_cv_folds = num_cv_folds):
  """
  This method explores the hyper-parameter space RANOMDLY and fits
  a series of parameters to the data for gradient boosting 

  Result is best set params which minimize prediction error
  """

  def split_data_into_train_test_and_validation(fdf = fdf, train_perc = .85, eval_perc = .15, model_ready_df = model_ready_df, known_pro_enrollments = known_pro_enrollments):
    """
    Training on a random 85% of the data
    Testing on remaining 15%
    Validating on random 15% of orgs from training set
    """
    print('Splitting data into training, test, and validation sets')

    # Following arrays define which pros are in which datasets
    unique_pros = fdf.pro_email.unique()
    num_pros = len(unique_pros)
    pros_to_train_on = np.random.choice(unique_pros, 
                                        size = int(train_perc * len(unique_pros)), 
                                        replace = False
                                       )
    pros_to_test_on = list( set(unique_pros) - set(pros_to_train_on) )
    pros_to_evaluate_model_against = np.random.choice(pros_to_train_on,
                                                      size = int(eval_perc * len(pros_to_train_on)),
                                                      replace = False
                                                     )
    pros_to_train_on = list( set(pros_to_train_on) - set(pros_to_evaluate_model_against) )

    # Following code block creates data subsets, given pro subsetted lists from above
    data_train = model_ready_df.loc[ model_ready_df['pro_email'].isin(pros_to_train_on) ]
    data_train__target = known_pro_enrollments.loc[known_pro_enrollments[ 'pro_email'].isin(pros_to_train_on), 'pro_enrolled']

    data_test = model_ready_df.loc[ model_ready_df['pro_email'].isin(pros_to_test_on) ]
    data_test__target = known_pro_enrollments.loc[known_pro_enrollments[ 'pro_email'].isin(pros_to_test_on), 'pro_enrolled']

    data_eval = model_ready_df.loc[ model_ready_df['pro_email'].isin(pros_to_evaluate_model_against) ]
    data_eval__target = known_pro_enrollments.loc[known_pro_enrollments[ 'pro_email'].isin(pros_to_evaluate_model_against), 'pro_enrolled']

    del data_train['pro_email'], data_test['pro_email'], data_eval['pro_email']

    return(data_train, data_train__target, data_test, data_test__target, data_eval, data_eval__target)

  data_train, data_train__target, data_test, data_test__target, data_eval, data_eval__target = split_data_into_train_test_and_validation(fdf = fdf, train_perc = .85, 
                                                                                                                                          eval_perc = .15, 
                                                                                                                                          model_ready_df = model_ready_df, 
                                                                                                                                          known_pro_enrollments = known_pro_enrollments)

  def param_search_and_cross_validation(num_estimators = 5000, early_stopping_rounds = 15, 
                                        data_train = data_train, data_train__target = data_train__target, data_test = data_test, 
                                        data_test__target = data_test__target, 
                                        data_eval = data_eval, data_eval__target = data_eval__target,
                                        num_cv_folds = num_cv_folds):
    """
      Trains a sequence of weak, boosted learners on training data
      Stops after error no measured improvements on prediction accuracy 
      after n stopping rounds

      Best params identified using log loss function via SK Opt library
 
      Function returns a series of best params on which to train a usable model on
    """
    
    """
      Additional params to consider if models overfit
    
      ## Learning Params
      early_stopping_rounds : will stop training if one metric of one validation data doesn’t improve in last early_stopping_round rounds

      ## IO Params
        max_bin : (default is 255, decrease the number to mitigate over fitting.  Risks drop in accuracy)
    """
    hyperparameters = [
      Integer(4 ,96, name = 'num_leaves'), # max number of leaves in one tree
      Integer(3,5, name = 'max_depth'), # max depth of an individual stump
      Real(2**-8, 2**-2, 'log-uniform', name = 'learning_rate'), # booster's learning rate, array of 50 numbers uniformly spaced in [ 1/(2^8), 1/(2^2) ]
      Integer(2,96, name = 'min_data_in_leaf'), # minimal number of data in one leaf
      Real(.5, 1.0, 'uniform', name = 'bagging_fraction'), # will randomly select part of data without resampling,
      Real(0.5, 1.0, 'uniform', name = 'feature_fraction'), # will select n% of features w/o resampling before training each stump
      Real(0.3, 1.0, "uniform", name ='colsample_bytree'), # subsample ratio of columns when constructing each tree
      Integer(25,150, name = 'max_bin') # max number of bins that feature values will be bucketed in
    ]
    
    clf = lgb.LGBMClassifier(n_estimators = num_estimators, # number of boosted trees to build
                            objective = 'binary',
                            silent = False, 
                            importance_type = 'gain', # gain in _some metric_ when a feature is included
                            seed = 12759081, # setting this, but underlying C++ seeds may overwrite
                            num_threads = 4, # number of real CPUs available on the playground machine
                            class_weight = 'balanced' # uses the values of y to automatically adjust weights inversely proportional to class frequencies in the input data
                            )

    # Per LGBM docs  https://scikit-optimize.github.io/#skopt.gp_minimize :
    # hyper params can be passed to a pre-defined fitted model's objective function via the following decorator
    @use_named_args(hyperparameters)
    def objective_fxn(**params):
      """
      Trains a series of models across 7 (hardcoded) folds of the training data
       & over the various pre-defined hyper params
      Leveraging SKLearn's `cross_val_score()` method for this
      """
      print('Training a model using the following params:')
      print(params)
      clf.set_params(**params)

      # Using cross val score() for now, but consider using cross_validate() for more info in a later iteration
      # https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation
      # Scoring docs also available at link above
      cv_score_scores__mean = -1.0 * cross_val_score(clf, # model
                                        data_train,  # data to fit a model to
                                        data_train__target, # target variable
                                        cv = num_cv_folds, # number of folds to iterate over
                                        scoring = 'neg_log_loss', 
                                        fit_params = {
                                            'early_stopping_rounds' : early_stopping_rounds,
                                            'eval_set'              : (data_eval, data_eval__target),
                                            'verbose'               : True
                                          }
                                        ).mean()
      print('Cross Validation Complete on Hyperparam Permutation\nMean Log Loss: {0}'.format(cv_score_scores__mean))
      print('Logging results')

      num_leaves.append(params['num_leaves'])
      max_depth.append(params['max_depth'])
      learning_rate.append(params['learning_rate'])
      min_data_in_leaf.append(params['min_data_in_leaf'])
      bagging_fraction.append(params['bagging_fraction'])
      feature_fraction.append(params['feature_fraction'])
      colsample_bytree.append(params['colsample_bytree'])
      max_bins.append(params['max_bin'])
      cross_val_score__mean.append(cv_score_scores__mean )
      table_record_udpate_timestamp.append(datetime.now())  

      return(cv_score_scores__mean)

    # Leveraging SK Opt's Gaussian Process Bayesian Optimization `gp_minimize()` method to approximate the 'best params' to
    # use in a final model
    # https://scikit-optimize.github.io/#skopt.gp_minimize
    # Method returns an OptimizeResult object.  See link above for full docs on all the data returned
    print('Starting Cross Validation via Random Search, with early stopping in place')

    gaussian_process_results_array = gp_minimize(objective_fxn, 
                                                 hyperparameters, # list of search space dimensions
                                                 n_calls = 30, # number of calls to make against the objective function
                                                 random_state = 215235 # seeding the optimizer for reproducible results
                                                )

    return(gaussian_process_results_array)

  ##
  # Invoke model training
  # Following arrays will store training results
  ##
  num_leaves = []
  max_depth = []
  learning_rate = []
  min_data_in_leaf = []
  bagging_fraction = []
  feature_fraction = []
  colsample_bytree = []
  max_bins = []
  cross_val_score__mean = []
  table_record_udpate_timestamp = []

  gaussian_process_results_array = param_search_and_cross_validation(num_estimators = 5000, 
                                                                      early_stopping_rounds = 15, 
                                                                      data_train = data_train, 
                                                                      data_train__target = data_train__target, 
                                                                      data_test = data_test, 
                                                                      data_test__target = data_test__target, 
                                                                      data_eval = data_eval, 
                                                                      data_eval__target = data_eval__target)

  print('Storing cross validation results to data frame')

  cross_val_results = pd.DataFrame(data = {
      'num_leaves'       : num_leaves,
      'max_depth'        : max_depth,
      'learning_rate'    : learning_rate,
      'min_data_in_leaf' : min_data_in_leaf,
      'bagging_fraction' : bagging_fraction,
      'feature_fraction' : feature_fraction,
      'colsample_bytree' : colsample_bytree,
      'max_bins'         : max_bins,
      'cross_val_score__mean'  : cross_val_score__mean,
      'table_record_udpate_timestamp' : table_record_udpate_timestamp
    })

  return(gaussian_process_results_array, cross_val_results, data_train, data_train__target, data_test, data_test__target, data_eval, data_eval__target)


def log_cross_validation_results(gaussian_process_results_array, cross_val_results):
  """
    Stores results for future review / analysis to local CSVs
  """
  # Local file storage directory
  file_dir = '/home/deploy/data_sci/datasets/lead_scoring/cross_validation_results/'
  file_name = 'cross_val_scores_w_balanced_class_weight_param_set' + str(date.today()).replace('-', '_') + '.csv'
  full_path = file_dir + file_name

  # Log cross val results
  cross_val_results.to_csv(full_path)
  pd.DataFrame(gaussian_process_results_array.x).to_csv(file_dir + 'best_params_w_balanced_class_weight_param_set.csv')


def refit_best_model(gaussian_process_results_array, 
                     early_stopping_rounds,
                     num_estimators,
                     data_train, data_train__target, 
                     data_eval, data_eval__target
                    ):
  """
  Method takes the best params as decided by the gp_minimizer() method and refits a new GBC model
  """
  print('Refitting a new model using hyperparam permutation that minimized log-loss on predicted probabilities')
  optimizer_results = gaussian_process_results_array.x

  best_params = {}
  best_params['num_leaves'] = optimizer_results[0]
  best_params['max_depth'] = optimizer_results[1]
  best_params['learning_rate'] = optimizer_results[2]
  best_params['min_data_in_leaf'] = optimizer_results[3]
  best_params['bagging_fraction'] = optimizer_results[4]
  best_params['feature_fraction'] = optimizer_results[5]
  best_params['colsample_bytree'] = optimizer_results[6]
  best_params['max_bins'] = optimizer_results[7]

  lead_scorer = lgb.LGBMClassifier(n_estimators = num_estimators, # number of boosted trees to build
                          objective = 'binary',
                          silent = False, 
                          importance_type = 'gain', # gain in prediction accuracy for a specific tree when a feature is included
                          seed = 12759081, # setting this, but underlying C++ seeds may overwrite
                          num_threads = 4, # number of real CPUs available on the playground machine
                          class_weight = 'balanced', # uses the values of y to automatically adjust weights inversely proportional to class frequencies in the input data
                          **best_params
                          ).fit(
                                  data_train,  # data to fit a model to
                                  data_train__target, # target variable)
                                  early_stopping_rounds = early_stopping_rounds,
                                  eval_set = (data_eval, data_eval__target),
                                  verbose = 50
                                )
  return(lead_scorer)


def evaluate_model_and_return_performance_stats(model, data_test, data_test__target):
  """
  Method evaluates model performance against unseen test data
  Sends some basic stats to Slack
  """
  predictions_on_test_data = model.predict(data_test)
  lead_win_probabilities = model.predict_proba(data_test)

  tn, fp, fn, tp = confusion_matrix(data_test__target, predictions_on_test_data).ravel()
  precision, recall, threshold = precision_recall_curve(data_test__target, predictions_on_test_data)
  classification_report_for_slack = classification_report(data_test__target, predictions_on_test_data, target_names = ['Not Enrolled', 'Enrolled'])

  stat_dict = {'True Positives' : tp, 'True Negatives' : tn, 'False Positives': fp, 'False Negatives' : fn}
  stat_report = pd.DataFrame(stat_dict, index = [0,1,2,3]).loc[0]

  # Save predictions

  def plot_a_binary_precision_recall_curve(precision, recall):
    """
      Method creates and stores a plot locally for logging purposes
      Docs: https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html#plot-the-precision-recall-curve
    """
    step_kwargs = ({'step' : 'post'} if 'step' in signature(plt.fill_between).parameters else {} )
    plt.step(recall, precision, color = 'b', alpha = 0.2, where = 'post')
    plt.fill_between(recall, precision, alpha = 0.2, color = 'b', **step_kwargs)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Binary Precision - Recall Curve: Precision = {0}'.format( round(tp / (tp + fp), 2)  ))

    # Save plot locally
    plot_filepath = '/home/deploy/data_sci/datasets/lead_scoring/model_output/precision_recall_curve' + str(date.today()).replace('-', '_') + '.png'
    plt.savefig(plot_filepath)

    print('Precision Recall plot saved locally at {0}'.format(datetime.now()))

    # Save plot in s3
    s3_fullpath = 's3://housecall-datascience/model_output/precision_recall_curve' + str(date.today()).replace('-', '_') + '.png'
    s3_command = 'aws s3 cp {0} {1}'.format(plot_filepath, s3_fullpath)

    print('Precision Recall plot saved to s3 at {0}'.format(datetime.now()))

  # Saves a plot locally and to s3 for logging purposes
  plot_a_binary_precision_recall_curve(precision, recall)

  slack_performance_header_message = {'text' : '*Most Recent Lead Scoring Model Performance Summary*'}
  dsu.send_logs_to_slack(slack_performance_header_message)
  slack_performance_log = {'text' : '```{0}```'.format(classification_report_for_slack)}
  dsu.send_logs_to_slack(slack_performance_log)
  slack_performance_log = {'text' : '```{0}```'.format(stat_report)}
  dsu.send_logs_to_slack(slack_performance_log)

  return(predictions_on_test_data, lead_win_probabilities)

def save_model_locally_and_to_s3(best_lead_scorer, cat_dict):
  """
  Method makes model available for reference both locally on server and remotely via s3
  """
  model_file_storage_directory = '/home/deploy/data_sci/datasets/lead_scoring/model_files/'
  model_filename = 'lead_win_pred' + str(date.today()).replace('-', '_') + '_model.joblib'

  full_model_filepath = model_file_storage_directory + model_filename
  print('Saving lead scoring model locally : {0}'.format(datetime.now()))

  # Saving seen categories for refrence at inference time
  cat_dict_filename = 'seen_categories.txt'
  cat_dict_full_path = model_file_storage_directory + cat_dict_filename
  
  with open(cat_dict_full_path, 'w') as cat_dict_file_handler:
    cat_dict_file_handler.write(dumps(cat_dict))
  
  try:
    dump(best_lead_scorer, full_model_filepath)
    print('Succesfully saved retrained lead scoring model locally at {0}'.format(datetime.now()))
    model_io_success_log_message = {'text' : 'Succesfully saved retrained model locally at {0}'.format(datetime.now())}
    dsu.send_logs_to_slack(model_io_success_log_message)

    print('Sending model file to s3 {0}'.format(datetime.now()))
    s3_filepath = 's3://housecall-datascience/model_output/lead_scoring/'
    full_s3_path = s3_filepath + model_filename
    s3_upload_command = 'aws s3 cp {0} {1}'.format(full_model_filepath, full_s3_path)
    model_io_success_log_message = {'text' : 'Succesfully saved retrained lead scoring model to s3 at {0}'.format(datetime.now())}
    dsu.send_logs_to_slack(model_io_success_log_message)
  except:
    error_encountered, error_value, error_traceback = exc_info()
    print("Error occurred while trying to save lead scoring model file locally : {0}".format(datetime.now()))
    formatted_stack_trace = repr(traceback.format_exception(error_encountered, error_value, error_traceback))
    print('Result from Stack Trace:')
    print(formatted_stack_trace)
    model_io_failure_error_message = {'text' : "Error occurred while trying to save lead scoring model file locally : {0}\n\nStack Trace: {1}".format(datetime.now(), formatted_stack_trace)}
    dsu.send_errors_to_slack(model_io_failure_error_message)

"""
Actual script execution starts here
"""

# fdf = pd.read_csv('/home/deploy/data_sci/datasets/lead_scoring/lead_scoring_features20191009.csv')
model_ready_df, known_pro_enrollments, cat_dict, fdf = all_the_data_preprocessing(categorical_feature_list, fdf, feature_column_names)

gaussian_process_results_array, cross_val_results, data_train, data_train__target, data_test, data_test__target, data_eval, data_eval__target = model_param_search_and_fitting(fdf, model_ready_df, known_pro_enrollments)

log_cross_validation_results(gaussian_process_results_array, cross_val_results)

best_lead_scorer = refit_best_model(gaussian_process_results_array = gaussian_process_results_array, 
                                     early_stopping_rounds = 15,
                                     num_estimators = 5000,
                                     data_train = data_train, data_train__target = data_train__target, 
                                     data_eval = data_eval, data_eval__target = data_eval__target)

predictions_on_test_data, lead_win_probabilities = evaluate_model_and_return_performance_stats(best_lead_scorer, data_test = data_test, data_test__target = data_test__target)

save_model_locally_and_to_s3(best_lead_scorer, cat_dict)




