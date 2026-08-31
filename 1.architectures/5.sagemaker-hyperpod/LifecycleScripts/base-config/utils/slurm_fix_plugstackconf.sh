#!/bin/bash

# Slurm cannot expand wildcard in etc/plugstack.conf while used with configless.
# The wildcard character '*' is interpreted as a character.
# The purpose of the script is to expand the wildcard by listing all the files matching the wildcard.
# BUG:
# Slurm configless bug workaround for pyxis SPANK plugin reference in etc/plugstack.conf
# known limitation of configless, where included files must be adjacent to their parent file.
# slurmctld is attempting to read '*' wildcard as a file, marking it not responsive
# The slurmctld not responsive issue was fixed in 23.11.5, but the wildcard limitation still exists.
# WORKAROUND:
# This script changes the Slurm configuration file /opt/slurm/etc/plugstack.conf expanding wildcard to include directly config files absolute path
# like /opt/slurm/etc/plugstack.conf.d/pyxis.conf instead of using the generic /opt/slurm/etc/plugstack.conf.d/*
# Avoids slurmctld restart hanging for minutes
# Avoids "slurmctld[]: slurmctld: error: s_p_parse_file: cannot stat file /opt/slurm/etc/plugstack.conf.d/*: No such file or directory, retry"


slurm_dir="/opt/slurm"
plugstack_conf="${slurm_dir}/etc/plugstack.conf"
plugstack_dir="${slurm_dir}/etc/plugstack.conf.d"
pyxis_conf="${plugstack_dir}/pyxis.conf"

smhp_conf="/opt/ml/config/resource_config.json"

# pretty echo
pecho(){
    echo
    echo "### $@"
}

# checking if user is root, exit if not
check_root(){
    if [ "$EUID" -ne 0 ] ; then
        echo "Please run as root"
        exit -1
    fi
}

# in Slurm configuration plugstack.conf: swap the generic "include" by expanding wildcard into "include" with direct config files absolute path
# ex: from "include ${plugstack_dir}/*" to "include ${pyxis_conf}"
# https://slurm.schedmd.com/spank.html # include keyword must appear on its own line, and takes a glob as its parameter
slurm_fix_plugstack(){
    pecho "start ${FUNCNAME}:"
    
    pecho "Fixing ${plugstack_conf}"
    cat "${plugstack_conf}"
    
    # expand all wildcard path
    while read -r line; do
        for f in $(ls $line) ;do
            echo "include $f" >> "${plugstack_conf}"
        done
    done < <(cat "${plugstack_conf}" | grep -v "^#" | grep '*' | egrep "^include|" | awk '{print $2}')

    # comment the whole wildcard line
    # sed -i "${plugstack_conf}" -e "s#^\(include[ ][ ]*.*${plugstack_dir}/\*.*\)#\#\1#g"
    sed -i "${plugstack_conf}" -e "s#^\(include[ ][ ]*.*\*.*\)#\#\1#g"

    cat "${plugstack_conf}"
    pecho "leave ${FUNCNAME}."
}

# main
main(){
    pecho "start ${FUNCNAME}:"

    check_root
    slurm_fix_plugstack # need to run on all instances (controller-machine and work-group-1, etc.)

    pecho "leave ${FUNCNAME}."
}

# main exec
main $@

exit

