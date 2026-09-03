"""CloudFormation custom resource: upload DevOps Agent skill assets from S3.

CloudFormation has no `AWS::DevOpsAgent::Asset` (or skill) resource type, so
skills are uploaded imperatively here — the same create-asset / update-asset
flow the earlier per-skill upload used, but driven from a manifest + zips that
`make sync-skills` staged into an S3 bucket (the SKILL.md + the ~85 KB
mental-model doc are far too large for an inline Lambda ZipFile).

Resource properties (from the template):
  AgentSpaceId  - the Agent Space to upload into
  AssetsBucket  - S3 bucket holding the staged skill zips
  SkillsVersion - content hash; a change forces CloudFormation to re-run Update
                  so edited skills / mental-model doc get re-uploaded
  Manifest      - list of {"name", "zipKey", "agentTypes": [...]}

Lifecycle:
  Create/Update - for each manifest entry, look up an existing asset by name and
                  update-asset (preserving its id) or create-asset otherwise.
  Delete        - delete-asset for each managed skill so the AgentSpace (deleted
                  after this resource, via DependsOn) has no leftover assets.

The DevOps Agent API blob member `content.zip.zipFile` takes raw bytes; boto3
handles the base64 wire-encoding, so we pass the downloaded zip bytes directly.
"""
import json

import boto3
import cfnresponse


_devops = boto3.client("devops-agent")
_s3 = boto3.client("s3")


def _parse_manifest(raw) -> list:
    """Manifest may arrive as a JSON string (CFN parameter) or a list."""
    if isinstance(raw, str):
        return json.loads(raw) if raw.strip() else []
    return raw or []


def _find_asset_id(agent_space_id: str, name: str) -> str | None:
    """Return the assetId of the skill asset named `name`, or None."""
    kwargs = {"agentSpaceId": agent_space_id, "maxResults": 100}
    while True:
        resp = _devops.list_assets(**kwargs)
        for item in resp.get("items", []) or []:
            meta = item.get("metadata", {}) or {}
            if item.get("assetType") == "skill" and meta.get("name") == name:
                return item.get("assetId")
        token = resp.get("nextToken")
        if not token:
            return None
        kwargs["nextToken"] = token


def _fetch_zip(bucket: str, key: str) -> bytes:
    obj = _s3.get_object(Bucket=bucket, Key=key)
    blob = obj["Body"].read()
    print(f"fetched s3://{bucket}/{key} ({len(blob)} bytes)")
    return blob


def _upload_one(agent_space_id: str, bucket: str, entry: dict) -> dict:
    name = entry["name"]
    zip_key = entry["zipKey"]
    agent_types = entry.get("agentTypes") or ["GENERIC"]
    blob = _fetch_zip(bucket, zip_key)

    existing_id = _find_asset_id(agent_space_id, name)
    if existing_id:
        print(f"updating existing asset {existing_id} for skill {name!r}")
        resp = _devops.update_asset(
            agentSpaceId=agent_space_id,
            assetId=existing_id,
            metadata={"agent_types": agent_types, "status": "ACTIVE"},
            content={"zip": {"zipFile": blob}},
        )
    else:
        print(f"creating new asset for skill {name!r}")
        resp = _devops.create_asset(
            agentSpaceId=agent_space_id,
            assetType="skill",
            metadata={"agent_types": agent_types, "status": "ACTIVE"},
            content={"zip": {"zipFile": blob}},
        )
    asset_id = resp.get("asset", {}).get("assetId", existing_id or "")
    return {"name": name, "assetId": asset_id}


def _delete_one(agent_space_id: str, name: str) -> None:
    asset_id = _find_asset_id(agent_space_id, name)
    if not asset_id:
        print(f"no asset to delete for skill {name!r}")
        return
    try:
        _devops.delete_asset(agentSpaceId=agent_space_id, assetId=asset_id)
        print(f"deleted asset {asset_id} for skill {name!r}")
    except Exception as e:  # noqa: BLE001 - best-effort teardown
        print(f"delete_asset failed for {name!r} ({asset_id}): {e!r}")


def handler(event, context):
    request_type = event.get("RequestType")
    props = event.get("ResourceProperties", {}) or {}
    agent_space_id = props.get("AgentSpaceId", "")
    bucket = props.get("AssetsBucket", "")
    manifest = _parse_manifest(props.get("Manifest"))
    physical_id = event.get("PhysicalResourceId") or f"skills-{agent_space_id}"

    print(
        f"RequestType={request_type} agentSpaceId={agent_space_id!r} "
        f"bucket={bucket!r} skills={[e.get('name') for e in manifest]}"
    )

    # Delete must NEVER fail the stack: any leftover skill assets are deleted
    # along with the AgentSpace anyway. Swallow all errors and report SUCCESS.
    if request_type == "Delete":
        try:
            if agent_space_id:
                for entry in manifest:
                    _delete_one(agent_space_id, entry.get("name", ""))
        except Exception as e:  # noqa: BLE001 - best-effort cleanup only
            print(f"delete cleanup error (ignored so stack can delete): {e!r}")
        cfnresponse.send(event, context, cfnresponse.SUCCESS, {}, physical_id)
        return

    try:
        # Create or Update
        if not agent_space_id or not bucket:
            raise ValueError("AgentSpaceId and AssetsBucket are required properties")
        results = [_upload_one(agent_space_id, bucket, entry) for entry in manifest]
        cfnresponse.send(
            event,
            context,
            cfnresponse.SUCCESS,
            {"Uploaded": ",".join(r["name"] for r in results)},
            physical_id,
        )
    except Exception as e:  # noqa: BLE001 - must always signal CFN
        print(f"ERROR: {e!r}")
        cfnresponse.send(
            event, context, cfnresponse.FAILED, {"Error": str(e)}, physical_id
        )
