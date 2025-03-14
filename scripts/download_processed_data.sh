#!/bin/bash

set -e

# If gdown doesn't work, you can download files from mentioned URLs manually
# and put them at appropriate locations.

mkdir -p .temp/

# URL: https://drive.google.com/file/d/1t2BjJtsejSIUZI54PKObMFG6_wMMG3bC/view?usp=sharing
gdown "1t2BjJtsejSIUZI54PKObMFG6_wMMG3bC&confirm=t" -O .temp/processed_data.zip
unzip  -o .temp/processed_data.zip -x "*.DS_Store"

rm -rf .temp/

mv processed_data/* datasets
rm -r processed_data
