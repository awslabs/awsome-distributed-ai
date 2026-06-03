#!/bin/bash

file_name=/fsxl/awsankur/esm2/esm.sqsh
[ -f $file_name ] && rm $file_name

enroot import -o $file_name dockerd://esm:aws