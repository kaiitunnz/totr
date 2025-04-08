#!/bin/bash

set -e

# If gdown doesn't work, you can download files from mentioned URLs manually
# and put them at appropriate locations.


# URL: https://drive.google.com/file/d/15j_BymIVPcDNAAI5m72a8d70jf_r0CN0/view?usp=sharing
gdown "15j_BymIVPcDNAAI5m72a8d70jf_r0CN0&confirm=t" -O raw_results.tar.gz
tar -xzvf raw_results.tar.gz
rm raw_results.tar.gz
