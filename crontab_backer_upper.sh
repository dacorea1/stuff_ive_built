#!/bin/bash

# Useful script for copying a crontab file to s3 

now=$(date +%Y_%m_%d__%H:%M)
filename="crontab_$now"
s3_path="s3://$s3_bucket_name/crontab_backups/$filename"
 

sudo aws s3 cp /var/spool/cron/deploy $s3_path