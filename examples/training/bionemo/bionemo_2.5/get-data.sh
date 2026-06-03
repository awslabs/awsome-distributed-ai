#!/bin/bash

docker run --rm -v /fsxl/awsankur/bionemo:/root/.cache/bionemo bionemo:aws download_bionemo_data esm2/testdata_esm2_pretrain:2.0
