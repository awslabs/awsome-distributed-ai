#!/bin/bash

sudo cp utils/motd.txt /etc/motd

# Grab instance type
token=$(curl -X PUT "http://169.254.169.254/latest/api/token" -H "X-aws-ec2-metadata-token-ttl-seconds: 21600")
instance_type=$(curl -H "X-aws-ec2-metadata-token: $token" -v http://169.254.169.254/latest/meta-data/instance-type)

GREEN="\e[32m"
ENDCOLOR="\e[0m"
echo -e "You're on the ${GREEN}$1${ENDCOLOR}" | sudo tee -a /etc/motd
echo -e "Controller Node IP: ${GREEN}$2${ENDCOLOR}" | sudo tee -a /etc/motd
[ ! -z "$3" ] && echo -e "Login Node IP: ${GREEN}$3${ENDCOLOR}" | sudo tee -a /etc/motd
echo -e "Instance Type: ${GREEN}ml.${instance_type}${ENDCOLOR}" | sudo tee -a /etc/motd