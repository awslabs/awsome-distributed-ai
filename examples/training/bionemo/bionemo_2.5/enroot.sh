#!/bin/bash

rm /fsxl/awsankur/bionemo.sqsh

enroot import -o /fsxl/awsankur/bionemo/bionemo.sqsh dockerd://bionemo:aws
