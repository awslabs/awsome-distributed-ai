# get ec2 metadata

Walk through ec2 imds v2 meta-data tree path in depth and export corresponding variables including json values flattening.
It replaces ec2-metadata CLI:
from "ec2-metadata --help":
    << Use to retrieve EC2 instance metadata from within a running EC2 instance.
    For more information on Amazon EC2 instance meta-data, refer to the documentation at
    https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-instance-metadata.html >>


```
./getmetadata.sh --help                                                                                            
ec2md - 0.1 (2024 May 30th)

Walk through ec2 imds v2 meta-data tree path in depth and export corresponding variables including json values flattening.
It replaces ec2-metadata CLI:
from "ec2-metadata --help":
    << Use to retrieve EC2 instance metadata from within a running EC2 instance.
    For more information on Amazon EC2 instance meta-data, refer to the documentation at
    https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-instance-metadata.html >>

Usage:
   ec2md [OPTION]|[OPTION <FILE>]... [VARIABLE] [[VARIABLE]...]

Arguments:
   [VARIABLE]...                   Output the specific variable only (no file written). It must start with "ec2" or "ec2".
   --nokey             -n          Remove the key and print only its value, option for "[VARIABLE]" only.
   --vars-export       -e <FILE>   Write "dynamic/ meta-data/" variables export in FILE. Do not forget to "source FILE" in your local shell environment.
   --vars-all          -a          Output all "dynamic/ meta-data/" variables export on stdout.
   --user-data         -u <FILE>   Write "user-data" in FILE.
   --user-data-import  -i <FILE>   Import and swap the "user-data" content in the local instance IMDS by the FILE content.
   
   --user-data-delete  -r          Delete the "user-data" on the local instance IMDS.
   --on                -o          Enable the metadata on the local instance.
   --off               -f          Disable the metadata on the local instance.
   
   --verbose           -v          Output extra information.
   --quiet             -q          Remove output of variables export while exporting to FILE, option for "--vars-export" only.
   --help              -h          Display this help and exit.
   --version           -V          Print version information and exit.
   
 Examples:
    ec2md user-data
    ec2md -u ./user-data.sh
    
    ec2md -a > ec2-metadata.sh && source ec2-metadata.sh
    ec2md -e   /etc/profile.d/ec2-metadata.sh
    
    ec2md -n metadata_placement_availability_zone_id
    ec2md -n meta-data/placement/availability-zone-id
```

    


