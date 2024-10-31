#!/bin/bash

set -e

common/scripts/create_component \
  --name my_python_metric \
  --language python \
  --type metric
