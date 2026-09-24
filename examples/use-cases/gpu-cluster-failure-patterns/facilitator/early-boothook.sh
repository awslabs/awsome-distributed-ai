#cloud-boothook
#!/bin/bash
set -Eeuo pipefail
trap 'trap "" TERM INT HUP; printf "AIM344 early metadata setup failed; boot held before user access\n" >&2; while :; do sleep 3600; done' ERR TERM INT HUP
install -d -m 0755 /opt/aim344/.prolog /usr/local/sbin
# Root-only metadata, on host OUTPUT and forwarded/container traffic.
iptables -C OUTPUT -d 169.254.169.254 -m owner ! --uid-owner 0 -j REJECT || iptables -I OUTPUT 1 -d 169.254.169.254 -m owner ! --uid-owner 0 -j REJECT
iptables -C FORWARD -d 169.254.169.254 -j REJECT || iptables -I FORWARD 1 -d 169.254.169.254 -j REJECT
ip6tables -C OUTPUT -d fd00:ec2::254 -m owner ! --uid-owner 0 -j REJECT || ip6tables -I OUTPUT 1 -d fd00:ec2::254 -m owner ! --uid-owner 0 -j REJECT
ip6tables -C FORWARD -d fd00:ec2::254 -j REJECT || ip6tables -I FORWARD 1 -d fd00:ec2::254 -j REJECT
test ! -L /usr/local/sbin/aim344-early-imds
printf '%s' 'H4sIAAAAAAACA1NW1E/KzNNPSizO4CpOLVHQTS3NVyjILEhNS8zM4cosKElMykktVtB1VvAPDQkIDVHQTVEwNLPUMzI10YPSCrq5CvnlealFCooKurqlmSm6EJ6Bgm6WQpCrl6tziEJNjQLCLE+YWYbkmYbiLDf/oHDHIBesJuGyHqbHEK8uoDVm2LyflmJgYJWabGRlRYrnzbD7nlTDUB2F5Hk0g3BajuJ3HJq4AAhiuWgWAgAA' | base64 -d | gzip -d > /usr/local/sbin/aim344-early-imds
chown root:root /usr/local/sbin/aim344-early-imds
chmod 0700 /usr/local/sbin/aim344-early-imds
test ! -L /etc/systemd/system/aim344-early-imds.service
printf '%s' 'H4sIAAAAAAACA12PMXICMQxFe5/CF1i2CK0LCBQUaSCpGApja4MmXmkjaQO+fRySNJTS/3rzdHwjtJPbgCbByZAprHYvT8ulF2brmEr1I1jM0aI/80w5SvVnGFjAzwriY0qg2ghDnIttYALKQAlBA7FbDQYStKrBmDuBsSGsG3TRTr8wgVvfUT8NbCYLi/IO5gnsyvLRTQL/K73MlvlKf7N7ZhoKJtPwmBwPv+yTe60TBCbQC5vb3iAdWsVCP6v0hVMsvZ6R+ohj+7iDKKV2OGZ1exgj0l1+e0MLFdQdd6QWSzm19HNGgbyuD97uG0VJ0ghPAQAA' | base64 -d | gzip -d > /etc/systemd/system/aim344-early-imds.service
chown root:root /etc/systemd/system/aim344-early-imds.service
chmod 0644 /etc/systemd/system/aim344-early-imds.service
systemctl daemon-reload
systemctl enable aim344-early-imds.service
/bin/bash /usr/local/sbin/aim344-early-imds
