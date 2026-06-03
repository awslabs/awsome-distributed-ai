#!/bin/bash
# walk through ec2 meta-data tree path in depth and export corresponding variables including json values
# replace ec2-metadata CLI
# WARNING: this command is exporting any environment variable given as argument

#####################
##################### env variables
#####################

# default values
app_name="ec2md" # app_name="$0"
dep_cmds="aws base64 curl awk tr jq sed getopt"
app_version="0.1 (2024 May 30th)"

# AWS IMDS metdata

imds_metadata_prefix="ec2" # "metadata"
imds_dynamic_prefix="ec2" # "dynamic"
declare -A prefix2swap=(
    [meta_data_]="${imds_metadata_prefix}_" # DO NOT TOUCH
    [dynamic_]="${imds_dynamic_prefix}_" # DO NOT TOUCH
    [_instance_identity_]="_"
    [_placement_]="_"
    [availability_zone]="az"
    [_network_interfaces_]="_"
    [security_groups]="sg"
    [block_device]="ebs"
)
imds_vars_path="dynamic/ meta-data/" # IMDS path for vars
imds_user_data_path="user-data"
imds_url="http://169.254.169.254/latest"
imds_header_token="X-aws-ec2-metadata-token"
imds_header_imds_token_ttl="X-aws-ec2-metadata-token-ttl-seconds"
imds_token_ttl=10

# global variables
verbose=0           # opt
quiet="yes"         # opt enforced
debug=""            # opt
nokey=""            # opt
inited=""           # state
exported=""         # state
vars_file=""        # global internal file

#####################
##################### functions
#####################

# remove global tmp file and quit
leave(){
    [ -f "${vars_file}" ] && rm -f "${vars_file}"
    exit $1
}

# check external dependencies
dep_check(){
    for c in ${@} ;do
        # $c --version | head -n1
        if ! command -v $c &> /dev/null ; then
            echo "[ERROR] Command \"${c}\" can not be found." >&2
            leave
        fi
    done
}

# print app version
version_print(){
    echo "${app_name} - ${app_version}"
}

# print app help
help_print(){
    version_print
    cat << EOF

Walk through ec2 imds v2 meta-data tree path in depth and export corresponding variables including json values flattening.
It replaces ec2-metadata CLI:
from "ec2-metadata --help":
    << Use to retrieve EC2 instance metadata from within a running EC2 instance.
    For more information on Amazon EC2 instance meta-data, refer to the documentation at
    https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-instance-metadata.html >>

Usage:
   ${app_name} [OPTION]|[OPTION <FILE>]... [VARIABLE] [[VARIABLE]...]

Arguments:
   [VARIABLE]...                   Output the specific variable only (no file written). It must start with "${imds_metadata_prefix}" or "${imds_dynamic_prefix}".
   --nokey             -n          Remove the key and print only its value, option for "[VARIABLE]" only.
   --vars-export       -e <FILE>   Write "${imds_vars_path}" variables export in FILE. Do not forget to "source FILE" in your local shell environment.
   --vars-all          -a          Output all "${imds_vars_path}" variables export on stdout.
   --user-data         -u <FILE>   Write "${imds_user_data_path}" in FILE.
   --user-data-import  -i <FILE>   Import and swap the "${imds_user_data_path}" content in the local instance IMDS by the FILE content.
   
   --user-data-delete  -r          Delete the "${imds_user_data_path}" on the local instance IMDS.
   --on                -o          Enable the metadata on the local instance.
   --off               -f          Disable the metadata on the local instance.
   
   --verbose           -v          Output extra information.
   --quiet             -q          Remove output of variables export while exporting to FILE, option for "--vars-export" only.
   --help              -h          Display this help and exit.
   --version           -V          Print version information and exit.
   
 Examples:
    ${app_name} user-data
    ${app_name} -u ./user-data.sh
    
    ${app_name} -a > ec2-metadata.sh && source ec2-metadata.sh
    ${app_name} -e   /etc/profile.d/ec2-metadata.sh
    
    ${app_name} -n metadata_placement_availability_zone_id
    ${app_name} -n meta-data/placement/availability-zone-id
    
    
EOF
}

# enable imds on the local instance
imds_enable(){
    aws ec2 modify-instance-metadata-options --region "${metadata_placement_region}" --instance-id "${metadata_instance_id}" --http-endpoint enabled
}

# disable imds on the local instance
imds_disable(){
    aws ec2 modify-instance-metadata-options --region "${metadata_placement_region}" --instance-id "${metadata_instance_id}" --http-endpoint disabled
}

# initialize an IMDS token on the local instance for a short duration due to new imds_v2 requirement
imds_initv2(){
    [ -n "${inited}" ] && return # single exec
    
    imds_token=$(curl -s -f -X PUT "${imds_url}/api/token" -H "${imds_header_imds_token_ttl}: ${imds_token_ttl}")
    if [ -z "${imds_token}" ];then
        echo "[ERROR] Could not get IMDSv2 token. Instance Metadata might have been disabled or this is not an EC2 instance." >&2
        leave 1
    else
        inited="done" # single exec
    fi
}

# metadata retrieval using the IMDS token
imds_get(){
    curl -s -H "${imds_header_token}: ${imds_token}"  "${imds_url}/${1}"
    if [ "${?}" -gt 0 ] ;then
        echo "[ERROR] Could not use IMDSv2 token to retrieve \"${1}\". Instance Metadata might have been disabled or this is not an EC2 instance." >&2
    fi
}

# cleaning of the metadata key in order to exported it as a Bash variable name
key_clean(){
    local k="${1}"
    k="$(echo "${k}" | tr "/.-" "_" | tr "A-Z" "a-z" | tr -d ":" )"
    # sed -e "s#^meta_data_#${imds_metadata_prefix}_#g" | sed -e "s#^dynamic_#${imds_dynamic_prefix}_#g" # moved into prefix2swap array
    for from in "${!prefix2swap[@]}" ;do
        to="${prefix2swap[${from}]}"
        k="$(echo "${k}" | sed "s#${from}#${to}#g")"
    done
    echo "${k}"
}

# recursive walk in depth across the IMDS URLs to retrieve all metadata available from the local instance
imds_walk(){
    if [[ $1 == *"<?xml"* ]];then # 404
        return
    elif [[ $1 == *"/" ]] ;then # folder
        for d in $(imds_get "${1}") ;do
            imds_walk "${1}${d}"
        done
    else # file
        val=$(imds_get "${1}")
        [ ${verbose} -ge 2 ] && echo "${1} --> ${val}" >&2
        val=$(echo "${val}" | sed -e "s#'#\\\'#g" ) # tr '\n' ';' | tr -d '\n' # extra caution # val=$(imds_get ${1})
        key="$(key_clean "${1}")"
        echo "export ${key}='${val}'"
        if [[ "${val}" == "{"* ]] ;then # sub part export of key value
            echo "${val}" | tr -d "':\"{}," | tr "A-Z" "a-z" | awk -v key="${key}" 'NF {print "export " key "_" $1 "=" "\x27" $2 "\x27" }'
        fi
    fi
}

# export the user-data from the local instance IMDS to the specified file or stdout
user_data_export(){
    if [ -z "${1}" ] ;then
        imds_get ${imds_user_data_path}
    else
        local export_file="${1}"
        # > ${export_file}
        imds_get "${imds_user_data_path}" > "${export_file}"
        if [ $? -eq 0 ]; then
            [ ${verbose} -ge 1 ] && echo "Created \"${export_file}\" with user-data content. You still need to set it executable."
        else
            echo "[ERROR] Cannot write the user-data content into ${export_file}." >&2
            ls -lhd ${export_file} >&2
        fi
    fi
}

# output the user-data from the local instance IMDS to stdout
user_data_output(){
    user_data_export $@
}

# import a specified file as the user-data content of the local instance
user_data_import(){
    local import_file="${1}"
    local tmp_file="$(mktemp)"
    cat "${import_file}" | base64 > "${tmp_file}"
    aws ec2 modify-instance-attribute --region "${metadata_placement_region}" --instance-id "${metadata_instance_id}" --attribute userData --value "file://${tmp_file}"
    rm -f "${tmp_file}"
}

# retrieve once all metadata ("variable export statements") and store it into an internal file and source it to export all its variables
vars_get(){
    [ -n "${exported}" ] && return # single exec
    
    vars_file=$(mktemp)
    > ${vars_file}
    for d in ${imds_vars_path} ;do
        imds_walk ${d} >> ${vars_file}
    done
    source ${vars_file}
    
    exported="done" # single exec
}

# store the "variable export statements" from our internal file into the specified file
vars_export(){
    local export_file="${1}"
    if [ -n "${export_file}" ] ;then
        cat ${vars_file} > ${export_file}
        if [ $? -eq 0 ]; then
            [ -z "${quiet}" ] && cat ${vars_file}
            [ ${verbose} -ge 1 ] && echo -e "Created \"${export_file}\" with \"export key='value'\" IMDS metadata content. You still need to source ${export_file} in your local shell environment."
        else
            echo "[ERROR] Cannot write the IMDS metadata content into ${export_file}." >&2
            ls -lhd ${export_file} >&2
        fi
    else
        echo "[WARNING] No file to export into." >&2
    fi
}

# output the "variable export statements" from our internal file to stdout or display a single item given
vars_output(){
    if [ -z "${1}" ] ;then
        cat ${vars_file}
    else
        key="$(key_clean "${1}")"
        # if [[ ! -v "${key}" ]] || [[ ! ${key} == "${imds_metadata_prefix}_"* ]] || ! [[ ! ${key} == "${imds_dynamic_prefix}_"* ]] ;then # extra security not needed
        if [[ ! -v "${key}" ]] ;then
            echo "[WARNING] Variable \"$1\" mapped to \"$key\" is unknown." >&2
        else
            if [ -z $nokey ] ;then
                echo "${key}: ${!key}"
            else
                echo "${!key}"
            fi
        fi
    fi
}

# arg parsing and exec
main(){
    short_options="dvqnhVu:e:ofri:a"
    long_options="debug,verbose,quiet,nokey,help,version,user-data:,vars-export:,on,off,delete,user-data-import:,all"
    getopt_return=$(getopt -a -o "${short_options}" -l "${long_options}" -n "${app_name}" -- "$@")
    eval set -- "${getopt_return}"
    
    # arg parsing and exec in the same loop
    declare -a actions=()
    while true ; do
        # echo "DEBUG2 argc:$# argv:$@ OPTIND:${OPTIND} getopt_return:${getopt_return}"
        # echo "DEBUG2 quiet:$quiet verbose:$verbose debug:$debug nokey:$nokey inited:$inited exported:$exported vars_file:$vars_file"
        case "$1" in
            --) shift ; break ;;
            -*debug|-d)              debug="yes" ; set -x                          ; shift ;;
            -*verbose|-v)            verbose=$((verbose + 1))                      ; shift ;;
            -*quiet|-q)              quiet="yes"                                   ; shift ;;
            -*nokey|-n)              nokey="yes"                                   ; shift ;;
            -*help|-h)               help_print                                    ; shift ; leave ;;
            -*version|-V)            version_print                                 ; shift ; leave ;; 
            -*on|-o)                 actions+=(imds_enable)                        ; shift ;;
            -*off|-f)                actions+=(imds_disable)                       ; shift ;;
            -*user-data|-u)          actions+=("user_data_export ${2}")            ; shift 2 ;;
            -*user-data-import|-i)   actions+=("user_data_import ${2}")            ; shift 2 ;;
            -*user-data-delete|-r)   actions+=("user_data_import /dev/null")       ; shift ;;
            -*vars-export|-e)        actions+=("vars_export ${2}")                 ; shift 2 ;;
            -*vars-all|-a)           actions+=(vars_output)                        ; shift ;;
            user[-_]data|userdata)   actions+=(user_data_output)                   ; shift ;;
            ${imds_metadata_prefix}[/_]*) actions+=("vars_output ${1}")                 ; shift ;;
            ${imds_dynamic_prefix}[/_]*)  actions+=("vars_output ${1}")                 ; shift ;;
            --*|-*)                  echo "[ERROR] Unkown option \"${2}\"" >&2     ; shift ; help_print ; leave -1 ;;
            *)                       echo "[ERROR] Unknown variable \"${2}\"" >&2  ; shift ; help_print ; leave -1 ;;
        esac
    done
    shift $((OPTIND-1))
   
    # systematic load for simplification
    imds_initv2
    vars_get
    
    for a in "${actions[@]}" ;do
        [ ${verbose} -ge 1 ] && echo "Executing: $a"
        $a
    done
    
    # display each variable
    for i in $@ ;do
        case "$i" in
            user[-_]data|userdata)                           user_data_output   ;;
            ${imds_metadata_prefix}[/_]*|${imds_dynamic_prefix}[/_]*)  vars_output "${i}" ;;
            *) echo "[ERROR] Unknown variable \"${i}\"" >&2 ;;
        esac
    done
} # main()

#####################
##################### exec
#####################

dep_check ${dep_cmds}
main $@
leave


exit -1 # should not be executed
