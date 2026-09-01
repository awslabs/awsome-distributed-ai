#!/bin/bash

terraform -chdir=hyperpod-eks-tf output -json > terraform_outputs.json

jq -r 'to_entries[] | .key + "=" + (.value.value | tostring)' terraform_outputs.json | \
  while IFS="=" read -r key value; do
    echo "export $(echo "$key" | tr '[:lower:]' '[:upper:]')=\"$value\"" >> env_vars.sh
  done
