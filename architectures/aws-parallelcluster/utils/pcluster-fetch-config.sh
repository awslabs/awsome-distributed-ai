#!/bin/bash

set -eo pipefail

# Dependency: jq (required), bat or yq (optionals)

declare -a HELP=(
    "[-h|--help]"
    "[-r|--region REGION]"
    "CLUSTER_NAME"
    "[-s|--syntax-highlighter <cat|bat|jq> (default: auto detect)]"
)

IS_SYNTAX_HIGHLIGHTER_ARGS=0
region=""
cluster_name=""
declare -a pcluster_args=()
declare -a syntax_highlighter=()
parse_args() {
    local key
    while [[ $# -gt 0 ]]; do
        key="$1"
        case $key in
        -h|--help)
            echo "Get the cluster config YAML of a ParallelCluster cluster."
            echo "Usage: $(basename ${BASH_SOURCE[0]}) ${HELP[@]}"
            exit 0
            ;;
        -r|--region)
            region="$2"
            shift 2
            ;;
        -s|--syntax-highlighter)
            syntax_highlighter=( "$2" )
            shift 2
            ;;
        --)
            SYNTAX_HIGHLIGHTER_ARGS=1
            shift
            ;;
        *)
            if [[ $IS_SYNTAX_HIGHLIGHTER_ARGS == 0 ]]; then
                [[ "$cluster_name" == "" ]] \
                    && cluster_name="$key" \
                    || { echo "Must define one cluster name only" ; exit -1 ; }
            else
                syntax_highlighter_args+=($key)
            fi
            shift
            ;;
        esac
    done

    [[ "$cluster_name" != "" ]] || { echo "Must define a cluster name" ; exit -1 ; }
    [[ "${region}" == "" ]] ||  { pcluster_args+=(--region $region) ; }
}

parse_args $@

pcluster_response=$(pcluster describe-cluster -n $cluster_name "${pcluster_args[@]}" || true)
cluster_config_url=$(echo "$pcluster_response" | jq -r .clusterConfiguration.url)
[[ "$cluster_config_url" == "null" ]] && { echo "${pcluster_response}" | jq . ; exit 1 ; }
cluster_config=$(curl --silent "$cluster_config_url")

# By default, auto-detect the syntax highlighter
if [[ ${#syntax_highlighter[@]} -eq 0 ]]; then
    if command -v bat &> /dev/null; then
        syntax_highlighter=( bat -pp --language yaml )
    elif command -v batcat &> /dev/null; then
        syntax_highlighter=( batcat -pp --language yaml )
    elif command -v yq &> /dev/null; then
        syntax_highlighter=( yq )
    else
        syntax_highlighter=( cat )
    fi
fi

echo "$cluster_config" | "${syntax_highlighter[@]}"
