#!/bin/bash

set -e

common/scripts/create_component \
  --name my_r_metric \
  --language r \
  --type metric
