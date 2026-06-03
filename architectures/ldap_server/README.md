# Deploy an OpenLDAP server on Amazon EC2.

This sample setup a OpenLDAP server on an Amazon EC2 instance for HPC and ML cluster.
It contains two groups:

1. clusteradmins: for people who will administer the cluster
1. clusterusers: for the personas who will use the cluster

It provides a UI based on `phpldapadmin` that enable you to create groups and users.

## Security groups

This solution creates two security groups with the following rules:

1. ldap-ui-external to allow communication to LDAP UI and from internet.

Inbound rules

| Type  | Protocol | Port Range | Source                                                                                    | Description                          |
| ----- | -------- | ---------- | ----------------------------------------------------------------------------------------- | ------------------------------------ |
| HTTPS | TCP      | 443        | Enter IP range or security group Id from which you want to access the UI from             | Allows access to the UI              |

Outbound rules

| Type     | Protocol | Port Range | Source                                                                                    | Description                       |
| -------- | -------- | ---------- | ----------------------------------------------------------------------------------------- | ----------------------------------|
| Internet | All      | All        | 0.0.0.0/0                                                                                 | Allows access to the internet     |


2. ldap-cluster to allow communication to LDAP server in the cluster.

Inbound rules

| Type | Protocol | Port Range | Source                                                                                    | Description                          |
| ---- | -------- | ---------- | ----------------------------------------------------------------------------------------- | ------------------------------------ |
| LDAP | TCP      | 389        | Choose Custom and enter the security group ID of the security group that you just created | Allows access to the OpenLDAP server |

## Deploy

1. Download the `cf_ldap_server.yaml` file
1. Run the following command
  ```bash
  aws cloudformation deploy --stack-name ldap-server \
  --template-file cf_ldap_server.yaml \
  --capabilities CAPABILITY_IAM \
  --parameter-overrides SubnetId=XXX SecurityGroupIds=XXX,XXX VpcId=XXX IpCidrUIAccess=XXX
  ```

## Connect to the UI

1. Retrieve the `LdapUIUrl` to connect to the LDAP User Interface.
  ```bash
  aws cloudformation describe-stacks --stack-name ldap-server \
  --query 'Stacks[0].Outputs[?OutputKey==`LdapUIUrl`].OutputValue' \
  --output text
  ```
  Copy URL into a Web Browser.


## Get the LDAP Password
The password to access the LDAP was generated randomly and stored in AWS Secret Manager under `LdapPassword` output of the cloudformation stack.

1. Get the Secret ARN
	```bash
	 SECRET_ARN=$(aws cloudformation describe-stacks --stack-name ldap-server \
	 --query 'Stacks[0].Outputs[?OutputKey==`LdapPassword`].OutputValue' \
	 --output text)
	```

1. Get the password that you will use to login
	```bash
	aws secretsmanager get-secret-value --secret-id $SECRET_ARN\
	  --query SecretString \
	  --output text
	```
