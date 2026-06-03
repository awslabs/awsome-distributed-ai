
## Troubleshooting

A common issue we see customer face is a problem with the post install scripts or issue to access capacity due to a mis-configuration. This can manifest itself through a `HeadNodeWaitCondition` that'll cause the ParallelCluster to fail a cluster deployment.

To solve that, you can look at the cluster logs in CloudWatch in the cluster log group, otherwise use the option `--rollback-on-failure false` to keep resources up upon failure for further troubleshooting.


#### Specification of reserved capacity in ParallelCluster config file

AWS ParallelCluster supports specifying the [CapacityReservationId](https://docs.aws.amazon.com/parallelcluster/latest/ug/Scheduling-v3.html#yaml-Scheduling-SlurmQueues-CapacityReservationTarget) in the cluster's config file. If using a capacity reservation put the ID i.e. `cr-12356790abcd` in your config file by substituting the variable `PLACEHOLDER_CAPACITY_RESERVATION_ID`. It should look like the following:

  ```yaml
  CapacityReservationTarget:
      CapacityReservationId: cr-12356790abcd
  ```

If you have multiple ODCR's you can group them together into a [*Capacity Reservation Group*](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/create-cr-group.html), this allows you to launch instances from multiple ODCR's as part of the **same queue** of the cluster.

1. First create a group, this will return a group arn like: `arn:aws:resource-groups:us-east-2:123456789012:group/MyCRGroup`. Save that for later.

    ```bash
    aws resource-groups create-group --name MyCRGroup --configuration '{"Type":"AWS::EC2::CapacityReservationPool"}' '{"Type":"AWS::ResourceGroups::Generic", "Parameters": [{"Name": "allowed-resource-types", "Values": ["AWS::EC2::CapacityReservation"]}]}'
    ```

2. Next add your capacity reservations to that group:

    ```bash
    aws resource-groups group-resources --group MyCRGroup --resource-arns arn:aws:ec2:sa-east-1:123456789012:capacity-reservation/cr-1234567890abcdef1 arn:aws:ec2:sa-east-1:123456789012:capacity-reservation/cr-54321abcdef567890
    ```

3. Then add the group to your cluster's config like so:

    ```yaml
        CapacityReservationTarget:
            CapacityReservationResourceGroupArn: arn:aws:resource-groups:us-east-2:123456789012:group/MyCRGroup
    ```