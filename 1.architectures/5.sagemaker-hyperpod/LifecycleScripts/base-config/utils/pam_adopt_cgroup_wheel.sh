#!/bin/bash

# This script is implementing 3 specific items following the documentation from https://slurm.schedmd.com/pam_slurm_adopt.html
# 1. Limit host memory usage at 99% MaxRAMPercent using cgroup enforcement
# 2. Prevent user to ssh without jobs running on that node using adding pam_slurm_adopt PAM module
# 3. Since pam_slurm_adopt will block ssh user access, we add a wheel group mechanism to authorize admin users to ssh with 2 different PAM modules.
#
# pam_slurm_adopt will always allow the root user access. To allow other admins to the system, there are 2 PAM implemented options to allow users to ssh:
# - pam_access.so using ${access_conf}
# - pam_listfile.so using ${wheel_list}
#
# pam_slurm_adopt
# The purpose of this module is to prevent users from sshing into nodes that they do not have a running job on, and to track the ssh connection and
# any other spawned processes for accounting and to ensure complete job cleanup when the job is completed. This module does this by determining the job
# which originated the ssh connection. The user's connection is "adopted" into the "external" step of the job.
# When access is denied, the user will receive a relevant error message.
#
# Implementing the content of https://slurm.schedmd.com/pam_slurm_adopt.html to add cgroups, pam_slurm_adopt and ssh admin access
#
# https://github.com/SchedMD/slurm/blob/master/contribs/pam_slurm_adopt/pam_slurm_adopt.c
# https://slurm.schedmd.com/slurm.conf.html
# root needed


# CGROUP --> check slurm_cgroups() to set specific Slurm options
slurm_dir="/opt/slurm"
slurm_cgroup_conf="${slurm_dir}/etc/cgroup.conf"
slurm_conf="${slurm_dir}/etc/slurm.conf"
slurm_conf_accounting="${slurm_dir}/etc/accounting.conf"

# admin users ssh without having jobs running on a node
admin_users="ubuntu" # list of admin users who can ssh without having jobs running on the node
admin_group="" # ex: admin_group="admin" name of the admin group used by pam_access.so. Set to "" to deactivate
access_conf="/etc/security/access.conf" # used by pam_access.so
shared_mount="/fsx" # to share files needed by all nodes
admin_dir="${shared_mount}/admin" # to store admin files containing admin users (${wheel_list}) used by pam_listfile.so
wheel_list="${admin_dir}/admin.lst" # text file listing the admins used by pam_listfile.so. Set to "" to deactivate
pam_conf="/etc/pam.d/sshd"
sshd_conf="/etc/ssh/sshd_config" # to add UsePAM=yes

# to clone and compile pam_slurm_adopt
install_dir="/tmp" # pam_slurm_adopt compilation
slurm_git_version="$(sinfo --version | tr ". " "-")-1" # slurm_git_version="slurm-23-11-3-1"
slurm_git_url="SchedMD/slurm"
slurm_git_dir="$(basename ${slurm_git_url})"
slurm_adopt_apt="libpam-slurm-adopt"

smhp_conf="/opt/ml/config/resource_config.json" # get_node_type

# APT CLI options to avoid failure
apt_opts='' # apt_opts='-o Dpkg::Options::="--force-confdef" -o Dpkg::Options::="--force-confold"'
export DEBIAN_FRONTEND=noninteractive

if [[ -n "${admin_group}" ]] && [[ -n "${wheel_list}" ]] ;then
    echo "WARNING: using both PAM options - pam_access.so using ${access_conf} AND pam_listfile.so using ${wheel_list}"
fi
if [[ -z "${admin_group}" ]] && [[ -z "${wheel_list}" ]] ;then
    echo "WARNING: none of PAM options is selected - there is no admin to ssh without jobs running on a node."
fi

# pretty echo
pecho(){
    #pretty echo
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

# retrieve the node type to detect if we are running on the "controller-machine" or not 
get_node_type(){
    headnode_ip="$(cat "${smhp_conf}" | jq -r '.InstanceGroups[] | select(.Name == "controller-machine") | .Instances[0].CustomerIpAddress')"
    login_node_ips="$(cat "${smhp_conf}" | jq -r '.InstanceGroups[] | select(.Name == "login-group") | .Instances[].CustomerIpAddress')"
    if [[ $(hostname -I | grep -c $headnode_ip ) -ge 1 ]]; then
        node_type="controller-machine"
    elif [[ $(hostname -I | grep -c "$login_node_ips" ) -ge 1 ]]; then
        node_type="login-group"
    else
        node_type="compute"
    fi
    pecho "node_type=${node_type}" # controller-machine, worker-group-1
}

# Configure the pam_slurm_adopt Slurm module to prevent users from sshing into nodes that they do not have a running job
slurm_pam_adopt(){
    pecho "start ${FUNCNAME}:"

    pecho "Adding UsePAM to ${sshd_conf}:"
    grep -Hn "^UsePAM " ${sshd_conf} 
    sed -i ${sshd_conf} -e "s/^UsePAM .*/UsePAM yes/g"
    grep -Hn "^UsePAM " ${sshd_conf} 

    pecho "Remove OS slurm-pam packages:"
    apt -y ${apt_opts} remove libpam-slurm-adopt libpam-slurm
    
    pecho "Download, compile and install the pam_slurm_adopt module from ${slurm_git_dir} GitHub repo:"
    path_orig="$(pwd)"
    cd ${install_dir}
    rm -rf ${slurm_git_dir}
    git clone --depth 1 -b ${slurm_git_version} https://github.com/${slurm_git_url}.git
    cd ${slurm_git_dir}
    ./configure --prefix=/opt/slurm > /dev/null
    cd contribs/pam_slurm_adopt/
    make -j > /dev/null
    make install 2>&1 | tail -n20
    cd ${path_orig}
    rm -rf ${slurm_git_dir}
    
    pecho "check PAM modules availability:"
    # ls -lh /lib/security /lib/x86_64-linux-gnu/security
    ls -lh /lib/security/pam_slurm_adopt.so
    ls -lh /lib/x86_64-linux-gnu/security/pam_listfile.so
    
    # if using pam_listfile.so using ${wheel_list}, create it
    if [[ -n "${wheel_list}" ]] ;then
        pecho "Setting ${wheel_list}:"
        mkdir -p "$(dirname "${wheel_list}" )"
        touch "${wheel_list}"
        chmod 0600 "${wheel_list}"
    fi
    
    # if using pam_access.so with ${access_conf}, set conf and create ${admin_group} if not existing
    if [[ -n "${admin_group}" ]] ;then
        pecho "Setting ${access_conf}:"
        getent group ${admin_group}
        if ! [[ $? ]] ;then
            groupadd ${admin_group}
        fi
        if [[ $(cat ${access_conf} | grep -v '^#' | grep -c ${admin_group} ) -eq 0 ]] ;then
            echo "+:(${admin_group}):ALL" >> "${access_conf}"
            echo "-:ALL:ALL" >> "${access_conf}"
        fi
    fi
    # add users to admin group and add it to the ${wheel_list} if it exists
    for user in ${admin_users} ;do
        [[ -n "${admin_group}" ]] && usermod -a -G "${admin_group}" "${user}"
        [[ -n "${wheel_list}" ]] && echo "${user}" >> "${wheel_list}"
    done
    
    # clean ${wheel_list} from double entries - reentrant
    if [[ -n "${wheel_list}" ]] ;then
        cat "${wheel_list}" | sort -u | tee "${wheel_list}"
    fi
    
    # if pam_slurm_adopt.so has NOT been added yet, add it, and add the 2 other options to ssh without jobs (for admin activities)
    if [[ $(cat ${pam_conf} | grep -v '^#' | grep -c pam_slurm_adopt.so) -eq 0 ]] ;then
        pecho "Adding pam_slurm_adopt.so at the bottom of ${pam_conf}"
        [[ -n "${admin_group}" ]] && echo "-account    sufficient    pam_access.so" | tee -a ${pam_conf}
        [[ -n "${wheel_list}" ]] && echo "-account    sufficient    pam_listfile.so item=user sense=allow onerr=fail file=${wheel_list}" | tee -a ${pam_conf}
        echo "-account    required      pam_slurm_adopt.so" | tee -a ${pam_conf}
    else
        pecho "pam_slurm_adopt.so already in ${pam_conf}. Not adding pam_access.so nor pam_listfile.so : clean ${pam_conf} first."
    fi
    
    pecho "leave ${FUNCNAME}."
}

# sub function to swap option from Slurm configuration files, it takes 2 + 1 args
# $1 is the parameter name
# $2 is the parameter new value
# $3 is slurm config file path, default is ${slurm_conf}
slurm_swap_opt(){
    pecho "start ${FUNCNAME}:"
    
    p="$(echo "$1" | cut -d'=' -f1)" # parameter
    v="$(echo "$1" | cut -d'=' -f2)" # value
    f="${2:-${slurm_conf}}" # slurm config file
    
    # pecho "Checking if ${p} is present in ${slurm_conf}:"
    c=$(grep -ic "^${p}=" "${f}")
    case $c in
        1)
            pecho "${p} present in ${f}:"
            echo "from: $(grep "^${p}=" "${f}")"
            sed -i "${f}" -e "s#^${p}=.*#${p}=${v}#g"
            echo "to:   $(grep "^${p}=" "${f}")"
        ;;
        0)
            pecho "${p} not present in ${f}, adding ${p}=${v} at the end of ${f}:"
            echo "${p}=${v}" >> "${f}"
            grep "^${p}=" "${f}"
        ;;
        *)
            pecho "Warning ${p} present multiple times, swapping all occurences..."
            sed -i "${f}" -e "s#^${p}=.*#${p}=${v}#g"
            grep "^${p}=" "${f}"
        ;;
    esac
    
    pecho "leave ${FUNCNAME}."
}

# add and configure the cgroups feature to Slurm configuration
slurm_cgroups(){
    pecho "start ${FUNCNAME}:"

    # Cgroup settings
    cp ${slurm_conf} ${slurm_conf}.old
    echo >> ${slurm_conf} # add \n to avoid missing \n collapse future append
    slurm_swap_opt ProctrackType=proctrack/cgroup
    slurm_swap_opt TaskPlugin=task/cgroup,task/affinity # even if TaskPlugin=task/none is "Required for auto-resume feature."
    slurm_swap_opt PrologFlags=Contain
    slurm_swap_opt SelectTypeParameters=CR_Core_Memory
    slurm_swap_opt JobAcctGatherType=jobacct_gather/cgroup "${slurm_conf_accounting}"
    slurm_swap_opt LaunchParameters=enable_nss_slurm,ulimit_pam_adopt
    
    
    cat << EOF > ${slurm_cgroup_conf}
CgroupPlugin=autodetect
ConstrainDevices=yes
ConstrainRAMSpace=yes
ConstrainSwapSpace=yes
SignalChildrenProcesses=yes
MaxRAMPercent=99

EOF
    pecho "leave ${FUNCNAME}."
}


main(){
    pecho "start ${FUNCNAME}:"
    
    check_root
    get_node_type
    
    if [[ $node_type == "controller-machine" ]] ;then
        slurm_cgroups
        # scontrol reconfigure
        systemctl --no-pager restart slurmctld
        systemctl --no-pager status slurmctld
    elif [[ $node_type == "login-group" ]] ;then
        # nothing to do
        echo "nothing to do for pam_adopt on login node"
    else # compute nodes
        slurm_pam_adopt
        systemctl --no-pager restart slurmd
        systemctl --no-pager status slurmd
    fi
    
    pecho "scontrol show config | grep -i cgroup:"
    scontrol show config | grep -i cgroup
    sinfo
    
    pecho "leave ${FUNCNAME}."
}

main $@

exit


# code to test pam_slurm_adopt on all nodes
# ssh-keygen -t rsa -q -f "$HOME/.ssh/id_rsa" -N ""
# cat ~/.ssh/id_rsa.pub >> ~/.ssh/authorized_keys
user="ubuntu" # a non admin user
nodes="$(sinfo -t idle -N -h -o "%n" | sort -u)"
squeue -u ${user}
for node in $nodes ;do
    echo "Testing node $node :"
    ssh -o StrictHostKeyChecking=no ${user}@${node} "hostname"
    jobid="$(sbatch -w "${node}" --wrap "sleep 10" -N 1 | grep -Po "[0-9]+")"
    ssh -o StrictHostKeyChecking=no ${user}@${node} "hostname"
    scancel ${jobid}
    echo
done
squeue -u ${user}

    # output expected:
    #   Testing node ip-10-1-28-64 :
    #   Access denied by pam_slurm_adopt: you have no active jobs on this node
    #   Connection closed by 10.1.28.64 port 22
    #   ip-10-1-28-64



