"""Minimal cfnresponse shim for S3-packaged custom-resource Lambdas.

CloudFormation auto-injects a `cfnresponse` module into inline (ZipFile)
Lambdas, but NOT into S3-packaged ones. The skill uploader is packaged with a
bundled current boto3 (its Asset API is newer than the Lambda runtime's boto3),
so it ships this module explicitly. API-compatible with the AWS-provided one.
"""
import json
import urllib.request

SUCCESS = "SUCCESS"
FAILED = "FAILED"


def send(event, context, response_status, response_data=None, physical_resource_id=None, no_echo=False, reason=None):
    response_url = event["ResponseURL"]
    body = {
        "Status": response_status,
        "Reason": reason
        or f"See the details in CloudWatch Log Stream: {getattr(context, 'log_stream_name', 'n/a')}",
        "PhysicalResourceId": physical_resource_id or getattr(context, "log_stream_name", "n/a"),
        "StackId": event["StackId"],
        "RequestId": event["RequestId"],
        "LogicalResourceId": event["LogicalResourceId"],
        "NoEcho": no_echo,
        "Data": response_data or {},
    }
    encoded = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        response_url,
        data=encoded,
        method="PUT",
        headers={"content-type": "", "content-length": str(len(encoded))},
    )
    try:
        with urllib.request.urlopen(req) as resp:
            print(f"cfnresponse status={resp.status}")
    except Exception as e:  # noqa: BLE001 - CFN will time out if we can't respond
        print(f"cfnresponse send failed: {e!r}")
        raise
