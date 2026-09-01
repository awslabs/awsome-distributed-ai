#!/bin/bash
aws cloudformation describe-stacks --stack-name vpc-stack-ml | jq -r .Stacks[].StackStatus

