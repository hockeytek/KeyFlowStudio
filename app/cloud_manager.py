"""AWS EC2 instance management for KeyFlow Studio Cloud.

Requires boto3 to be installed: pip install boto3
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)

_REMOTE_WORKER_ROOT = "~/keyflow-worker"
_REMOTE_WORKER_ENTRY = "~/keyflow-worker/ec2_worker/worker.py"
_REMOTE_WORKER_LOG = "~/keyflow-worker/worker.log"
_REMOTE_WORKER_MANIFEST = "~/keyflow-worker/.keyflow_bundle_manifest.json"
_LOCAL_CLOUD_BUNDLE_PATHS = (
    "ec2_worker",
    "app/services",
    "app/utils",
    "app/__init__.py",
)


class InstanceState(Enum):
    UNKNOWN = "unknown"
    PENDING = "pending"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    TERMINATED = "terminated"
    ERROR = "error"


# Default timeouts for all AWS API calls (connect, read) in seconds
_AWS_CONNECT_TIMEOUT = 6
_AWS_READ_TIMEOUT = 15


def _get_ec2(profile: str, region: str):
    """Return boto3 EC2 client with explicit timeouts."""
    import boto3
    from botocore.config import Config
    cfg = Config(
        connect_timeout=_AWS_CONNECT_TIMEOUT,
        read_timeout=_AWS_READ_TIMEOUT,
        retries={"max_attempts": 1},
    )
    session = boto3.Session(profile_name=profile or None, region_name=region)
    return session.client("ec2", config=cfg)


def _get_sts(profile: str | None, region: str):
    """Return boto3 STS client with explicit timeouts."""
    import boto3
    from botocore.config import Config
    cfg = Config(
        connect_timeout=_AWS_CONNECT_TIMEOUT,
        read_timeout=_AWS_READ_TIMEOUT,
        retries={"max_attempts": 1},
    )
    session = boto3.Session(profile_name=profile or None, region_name=region)
    return session.client("sts", config=cfg)


# Human-readable city/location names for AWS regions (AWS Console labels)
AWS_REGION_NAMES: dict[str, str] = {
    "af-south-1":      "Cape Town",
    "ap-east-1":       "Hong Kong",
    "ap-east-2":       "Taiwan",
    "ap-northeast-1":  "Tokyo",
    "ap-northeast-2":  "Seoul",
    "ap-northeast-3":  "Osaka",
    "ap-south-1":      "Mumbai",
    "ap-south-2":      "Hyderabad",
    "ap-southeast-1":  "Singapore",
    "ap-southeast-2":  "Sydney",
    "ap-southeast-3":  "Jakarta",
    "ap-southeast-4":  "Melbourne",
    "ap-southeast-5":  "Malaysia",
    "ap-southeast-6":  "Thailand",
    "ap-southeast-7":  "Bangkok",
    "ca-central-1":    "Canada Central",
    "ca-west-1":       "Calgary",
    "eu-central-1":    "Frankfurt",
    "eu-central-2":    "Zurich",
    "eu-north-1":      "Stockholm",
    "eu-south-1":      "Milan",
    "eu-south-2":      "Spain",
    "eu-west-1":       "Ireland",
    "eu-west-2":       "London",
    "eu-west-3":       "Paris",
    "il-central-1":    "Tel Aviv",
    "me-central-1":    "UAE",
    "me-south-1":      "Bahrain",
    "mx-central-1":    "Mexico",
    "sa-east-1":       "São Paulo",
    "us-east-1":       "N. Virginia",
    "us-east-2":       "Ohio",
    "us-west-1":       "N. California",
    "us-west-2":       "Oregon",
}


def get_available_regions(
    profile: str | None = None,
    region: str = "eu-west-1",
) -> tuple[list[str], str]:
    """Return the AWS EC2 regions available to the current account/profile."""
    try:
        ec2 = _get_ec2(profile or "", region)
        response = ec2.describe_regions(AllRegions=False)
        region_names = sorted(
            {
                str(item.get("RegionName") or "").strip()
                for item in response.get("Regions", [])
                if str(item.get("RegionName") or "").strip()
            }
        )
        return region_names, ""
    except ImportError:
        return [], "boto3 not installed. Run: pip install boto3"
    except Exception as e:
        logger.warning("get_available_regions error: %s", e)
        return [], str(e)


def get_regions_with_gpu_quota(
    profile: str | None = None,
    fallback_region: str = "eu-west-1",
) -> tuple[list[str], str]:
    """Return EC2 regions where the account has non-zero GPU (G/VT) vCPU quotas.

    For each enabled region the function checks both the Spot and On-Demand
    G/VT instance quotas via the *service-quotas* API concurrently.

    Filtering rules:
    - Included  — quota > 0 for Spot OR On-Demand G/VT vCPUs.
    - Excluded  — both quotas are explicitly 0 (no capacity granted).
    - Included  — quota API failed or timed out (fail-open, better than
      accidentally hiding a region the user can actually use).
    - ``fallback_region`` is always included regardless of quota result.

    All checks run in parallel (≤ 20 threads); overall timeout is 10 s.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from concurrent.futures import TimeoutError as FuturesTimeout

    regions, err = get_available_regions(profile=profile, region=fallback_region)
    if not regions:
        return [fallback_region], err

    all_regions: list[str] = sorted(set(regions) | {fallback_region})

    def _check(region: str) -> tuple[str, bool]:
        """Return (region, has_quota).  Fails open on any error."""
        try:
            spot_q = get_spot_vcpu_quota(region, profile)
            ondemand_q = get_ondemand_vcpu_quota(region, profile)
            if spot_q is not None and ondemand_q is not None:
                return region, (spot_q + ondemand_q) > 0
        except Exception:
            pass
        return region, True  # API unavailable → include (fail-open)

    checked: set[str] = set()
    included: list[str] = []

    with ThreadPoolExecutor(max_workers=20) as pool:
        futures = {pool.submit(_check, r): r for r in all_regions}
        try:
            for fut in as_completed(futures, timeout=10):
                r, has_q = fut.result()
                checked.add(r)
                if has_q:
                    included.append(r)
        except FuturesTimeout:
            logger.debug("get_regions_with_gpu_quota: timed out, some regions unchecked")

    # Fail-open for regions whose quota check did not complete in time
    for r in all_regions:
        if r not in checked:
            included.append(r)

    if not included:
        included = list(all_regions)

    # Always keep the current/fallback region even if quota is 0
    if fallback_region not in included:
        included.append(fallback_region)

    return sorted(set(included)), err


def get_instance_state(instance_id: str, region: str, profile: str) -> tuple[InstanceState, str]:
    """Return (state, public_ip). public_ip is '' when stopped."""
    try:
        ec2 = _get_ec2(profile, region)
        resp = ec2.describe_instances(InstanceIds=[instance_id])
        inst = resp["Reservations"][0]["Instances"][0]
        state_name = inst["State"]["Name"]
        public_ip = inst.get("PublicIpAddress", "")
        state = InstanceState(state_name) if state_name in InstanceState._value2member_map_ else InstanceState.UNKNOWN
        return state, public_ip
    except ImportError:
        return InstanceState.ERROR, ""
    except Exception as e:
        logger.warning("get_instance_state error: %s", e)
        return InstanceState.ERROR, str(e)


def get_instance_launch_config(instance_id: str, region: str, profile: str | None = None) -> tuple[dict | None, str]:
    """Return instance type/market details needed to compute the real launch price."""
    try:
        ec2 = _get_ec2(profile, region)
        resp = ec2.describe_instances(InstanceIds=[instance_id])
        reservations = resp.get("Reservations", [])
        if not reservations or not reservations[0].get("Instances"):
            return None, "Instance not found"
        inst = reservations[0]["Instances"][0]
        return {
            "id": inst.get("InstanceId", instance_id),
            "instance_type": str(inst.get("InstanceType") or ""),
            "market": str(inst.get("InstanceLifecycle") or "on-demand"),
            "state": str((inst.get("State") or {}).get("Name") or "unknown"),
            "public_ip": str(inst.get("PublicIpAddress") or ""),
        }, ""
    except ImportError:
        return None, "boto3 not installed. Run: pip install boto3"
    except Exception as e:
        logger.warning("get_instance_launch_config error: %s", e)
        return None, str(e)


def _workspace_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _iter_cloud_bundle_files() -> list[Path]:
    root = _workspace_root()
    collected: list[Path] = []
    for rel_path in _LOCAL_CLOUD_BUNDLE_PATHS:
        path = root / rel_path
        if not path.exists():
            continue
        if path.is_file():
            collected.append(path)
            continue
        for file_path in sorted(path.rglob("*")):
            if not file_path.is_file():
                continue
            if "__pycache__" in file_path.parts or file_path.suffix in {".pyc", ".pyo"}:
                continue
            collected.append(file_path)
    return collected


def build_cloud_worker_bundle_manifest() -> dict:
    """Return a manifest describing the local cloud worker bundle revision."""
    root = _workspace_root()
    revision_hasher = hashlib.sha256()
    files: list[dict] = []
    for file_path in _iter_cloud_bundle_files():
        rel_path = file_path.relative_to(root).as_posix()
        payload = file_path.read_bytes()
        file_sha = hashlib.sha256(payload).hexdigest()
        revision_hasher.update(rel_path.encode("utf-8"))
        revision_hasher.update(file_sha.encode("ascii"))
        files.append({
            "path": rel_path,
            "sha256": file_sha,
            "size": len(payload),
        })
    return {
        "revision": revision_hasher.hexdigest()[:16],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "file_count": len(files),
        "files": files,
    }


def _find_local_gvm_core_dir() -> Path | None:
    candidates = [
        _workspace_root() / ".venv" / "lib",
        Path(sys_prefix) if (sys_prefix := os.environ.get("VIRTUAL_ENV")) else Path(),
    ]
    for base in candidates:
        if not base:
            continue
        if base.name == "lib":
            for site_packages in sorted(base.glob("python*/site-packages")):
                pkg = site_packages / "gvm_core"
                if pkg.is_dir():
                    return pkg
        else:
            pkg = base / "lib"
            for site_packages in sorted(pkg.glob("python*/site-packages")):
                candidate = site_packages / "gvm_core"
                if candidate.is_dir():
                    return candidate
    return None


def _run_local_command(args: list[str], *, timeout: int = 300) -> tuple[bool, str]:
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return False, f"Timeout after {timeout}s: {' '.join(args)}"
    except Exception as exc:
        return False, str(exc)
    output = (result.stdout or "") + (result.stderr or "")
    return result.returncode == 0, output


def _run_remote_command(public_ip: str, ssh_user: str, ssh_key_path: str, script: str, *, timeout: int = 300) -> tuple[bool, str]:
    """Execute a bash script on remote host via SSH. Returns (success, output)."""
    key_path = os.path.expanduser(ssh_key_path)
    try:
        result = subprocess.run(
            ["ssh", "-i", key_path, "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=15",
             f"{ssh_user}@{public_ip}", "bash", "-s"],
            input=script,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return False, f"Remote command timed out after {timeout}s"
    except Exception as exc:
        return False, str(exc)
    output = (result.stdout or "") + (result.stderr or "")
    return result.returncode == 0, output


def _rsync_to_remote(local_path: Path, remote_target: str, ssh_user: str, public_ip: str, ssh_key_path: str, *, delete: bool = True) -> tuple[bool, str]:
    rsync = shutil.which("rsync")
    if not rsync:
        return False, "rsync is not installed locally"
    source = str(local_path)
    if local_path.is_dir() and not source.endswith("/"):
        source += "/"
    cmd = [
        rsync,
        "-az",
        "--force",
        "--exclude=__pycache__/",
        "--exclude=*.pyc",
        "-e",
        f"ssh -i {ssh_key_path} -o StrictHostKeyChecking=no -o ConnectTimeout=15",
        source,
        f"{ssh_user}@{public_ip}:{remote_target}",
    ]
    if delete:
        cmd.insert(2, "--delete")
    return _run_local_command(cmd, timeout=600)


def start_instance(instance_id: str, region: str, profile: str) -> tuple[bool, str]:
    """Start EC2 instance. Returns (success, message)."""
    try:
        ec2 = _get_ec2(profile, region)
        ec2.start_instances(InstanceIds=[instance_id])
        return True, "starting"
    except ImportError:
        return False, "boto3 not installed. Run: pip install boto3"
    except Exception as e:
        response = getattr(e, "response", None)
        code = (response or {}).get("Error", {}).get("Code", "") if isinstance(response, dict) else ""
        if code == "InsufficientInstanceCapacity":
            logger.warning("start_instance: InsufficientInstanceCapacity for %s in %s", instance_id, region)
            return False, "INSUFFICIENT_CAPACITY"
        logger.warning("start_instance error: %s", e)
        return False, str(e)


def stop_instance(instance_id: str, region: str, profile: str) -> tuple[bool, str]:
    """Stop EC2 instance. Returns (success, message)."""
    try:
        ec2 = _get_ec2(profile, region)
        ec2.stop_instances(InstanceIds=[instance_id])
        return True, "stopping"
    except ImportError:
        return False, "boto3 not installed. Run: pip install boto3"
    except Exception as e:
        logger.warning("stop_instance error: %s", e)
        return False, str(e)


def terminate_instance(instance_id: str, region: str, profile: str) -> tuple[bool, str]:
    """Terminate EC2 instance. Returns (success, message)."""
    try:
        ec2 = _get_ec2(profile, region)
        ec2.terminate_instances(InstanceIds=[instance_id])
        return True, "terminating"
    except ImportError:
        return False, "boto3 not installed. Run: pip install boto3"
    except Exception as e:
        logger.warning("terminate_instance error: %s", e)
        return False, str(e)


def create_ami_from_instance(
    instance_id: str,
    region: str,
    profile: str,
    name: str,
    description: str = "",
    no_reboot: bool = True,
) -> tuple[bool, str]:
    """Create an AMI from an EC2 instance.

    Returns (success, ami_id_or_error).
    """
    try:
        ec2 = _get_ec2(profile, region)
        resp = ec2.create_image(
            InstanceId=instance_id,
            Name=name,
            Description=description or name,
            NoReboot=no_reboot,
            TagSpecifications=[{
                "ResourceType": "image",
                "Tags": [
                    {"Key": "Name", "Value": name},
                    {"Key": "App", "Value": "KeyFlowStudio"},
                    {"Key": "SourceInstanceId", "Value": instance_id},
                ],
            }],
        )
        return True, resp["ImageId"]
    except ImportError:
        return False, "boto3 not installed. Run: pip install boto3"
    except Exception as e:
        logger.warning("create_ami_from_instance error: %s", e)
        return False, str(e)


def change_instance_type(
    instance_id: str,
    region: str,
    profile: str,
    instance_type: str,
) -> tuple[bool, str]:
    """Change EC2 instance type. The instance must be stopped first.

    Returns (success, message).
    """
    try:
        ec2 = _get_ec2(profile, region)
        ec2.modify_instance_attribute(
            InstanceId=instance_id,
            InstanceType={"Value": instance_type},
        )
        return True, instance_type
    except ImportError:
        return False, "boto3 not installed. Run: pip install boto3"
    except Exception as e:
        logger.warning("change_instance_type error: %s", e)
        return False, str(e)


def find_keyflow_ami(region: str, profile: str | None = None) -> tuple[str, str, str]:
    """Search for a KeyFlow Studio AMI in the user's own EC2 account.

    Looks first by tag ``App=KeyFlowStudio``, then by name containing "KeyFlow".
    Returns (ami_id, ami_name, error_message). If nothing is found, returns ('', '', '').
    """
    try:
        ec2 = _get_ec2(profile, region)
        # Primary: tag-based search
        resp = ec2.describe_images(
            Owners=["self"],
            Filters=[
                {"Name": "tag:App", "Values": ["KeyFlowStudio"]},
                {"Name": "state", "Values": ["available"]},
            ],
        )
        images = sorted(
            resp.get("Images", []),
            key=lambda x: x.get("CreationDate", ""),
            reverse=True,
        )
        if images:
            img = images[0]
            return img["ImageId"], img.get("Name", ""), ""
        # Fallback: name-based search
        resp2 = ec2.describe_images(
            Owners=["self"],
            Filters=[
                {"Name": "name", "Values": ["*KeyFlow*", "*keyflow*"]},
                {"Name": "state", "Values": ["available"]},
            ],
        )
        images2 = sorted(
            resp2.get("Images", []),
            key=lambda x: x.get("CreationDate", ""),
            reverse=True,
        )
        if images2:
            img = images2[0]
            return img["ImageId"], img.get("Name", ""), ""
        return "", "", ""
    except ImportError:
        return "", "", "boto3 not installed. Run: pip install boto3"
    except Exception as e:
        logger.warning("find_keyflow_ami error: %s", e)
        return "", "", str(e)


def find_public_gpu_ami(region: str, profile: str | None = None) -> tuple[str, str, str]:
    """Find a public AWS Deep Learning GPU AMI suitable for a fresh instance.

    Prefers recent Ubuntu 22.04 GPU images published by AWS/Amazon.
    Returns (ami_id, ami_name, error_message). If nothing is found, returns ('', '', '').
    """
    name_patterns = [
        "Deep Learning Base OSS Nvidia Driver GPU AMI*Ubuntu 22.04*",
        "Deep Learning OSS Nvidia Driver AMI GPU PyTorch*Ubuntu 22.04*",
        "*GPU*Ubuntu 22.04*Deep Learning*",
    ]
    owner_sets = [
        ["amazon"],
        ["aws-marketplace"],
    ]
    try:
        ec2 = _get_ec2(profile, region)
        for owners in owner_sets:
            for pattern in name_patterns:
                resp = ec2.describe_images(
                    Owners=owners,
                    Filters=[
                        {"Name": "name", "Values": [pattern]},
                        {"Name": "state", "Values": ["available"]},
                        {"Name": "architecture", "Values": ["x86_64"]},
                    ],
                )
                images = sorted(
                    resp.get("Images", []),
                    key=lambda x: x.get("CreationDate", ""),
                    reverse=True,
                )
                if images:
                    img = images[0]
                    return img["ImageId"], img.get("Name", ""), ""
        return "", "", ""
    except ImportError:
        return "", "", "boto3 not installed. Run: pip install boto3"
    except Exception as e:
        logger.warning("find_public_gpu_ami error: %s", e)
        return "", "", str(e)


def list_user_amis(region: str, profile: str | None = None) -> tuple[list[dict], str]:
    """List all available AMIs owned by the current AWS account in the region.

    Returns (amis, error_message).  Each dict: {id, name, description, creation_date}.
    """
    try:
        ec2 = _get_ec2(profile, region)
        resp = ec2.describe_images(
            Owners=["self"],
            Filters=[{"Name": "state", "Values": ["available"]}],
        )
        images = sorted(
            resp.get("Images", []),
            key=lambda x: x.get("CreationDate", ""),
            reverse=True,
        )
        result = [
            {
                "id": img["ImageId"],
                "name": img.get("Name", ""),
                "description": img.get("Description", ""),
                "creation_date": img.get("CreationDate", "")[:10],
            }
            for img in images
        ]
        return result, ""
    except ImportError:
        return [], "boto3 not installed — run: pip install boto3"
    except Exception as e:
        logger.warning("list_user_amis error: %s", e)
        return [], str(e)


def get_key_pairs(region: str, profile: str | None = None) -> tuple[list[str], str]:
    """List EC2 key pair names available in the region.
    Returns (names, error_message).
    """
    try:
        ec2 = _get_ec2(profile, region)
        resp = ec2.describe_key_pairs()
        names = sorted(kp["KeyName"] for kp in resp.get("KeyPairs", []))
        return names, ""
    except ImportError:
        return [], "boto3 not installed"
    except Exception as e:
        logger.warning("get_key_pairs error: %s", e)
        return [], str(e)


def get_security_groups(region: str, profile: str | None = None) -> tuple[list[dict], str]:
    """List EC2 security groups available in the region.
    Returns (sgs, error_message).  Each dict: {id, name, description}.
    """
    try:
        ec2 = _get_ec2(profile, region)
        resp = ec2.describe_security_groups()
        result = sorted(
            [
                {
                    "id": sg["GroupId"],
                    "name": sg.get("GroupName", ""),
                    "description": sg.get("Description", ""),
                }
                for sg in resp.get("SecurityGroups", [])
            ],
            key=lambda x: x["name"],
        )
        return result, ""
    except ImportError:
        return [], "boto3 not installed"
    except Exception as e:
        logger.warning("get_security_groups error: %s", e)
        return [], str(e)


def list_instances(region: str, profile: str) -> tuple[list[dict], str]:
    """Return (instances, error_message). error_message is '' on success.

    Each dict: {
        id, name, state, instance_type, public_ip, launch_time, market,
        key_name, security_group_ids
    }
    """
    try:
        ec2 = _get_ec2(profile, region)
        resp = ec2.describe_instances(
            Filters=[{"Name": "instance-state-name",
                       "Values": ["pending", "running", "stopping", "stopped"]}]
        )
        result = []
        for reservation in resp.get("Reservations", []):
            for inst in reservation.get("Instances", []):
                name = ""
                for tag in inst.get("Tags", []):
                    if tag["Key"] == "Name":
                        name = tag["Value"]
                        break
                result.append({
                    "id": inst["InstanceId"],
                    "name": name,
                    "state": inst["State"]["Name"],
                    "instance_type": inst.get("InstanceType", ""),
                    "market": inst.get("InstanceLifecycle", "on-demand"),
                    "public_ip": inst.get("PublicIpAddress", ""),
                    "launch_time": str(inst.get("LaunchTime", "")),
                    "key_name": inst.get("KeyName", ""),
                    "security_group_ids": [
                        sg.get("GroupId", "")
                        for sg in inst.get("SecurityGroups", [])
                        if sg.get("GroupId")
                    ],
                })
        return result, ""
    except ImportError:
        return [], "boto3 not installed — run: pip install boto3"
    except Exception as e:
        logger.warning("list_instances error: %s", e)
        return [], str(e)


def _get_ami_root_volume_gb(ec2_client, ami_id: str) -> tuple[int, str]:
    """Return the minimum root volume size (GB) required by the AMI snapshot.

    Calls describe_images and inspects the BlockDeviceMappings of the AMI
    to find the root device volume size so launch_instance can respect it.
    Returns (size_gb, device_name); falls back to (0, "/dev/sda1") on error.
    """
    try:
        resp = ec2_client.describe_images(ImageIds=[ami_id])
        images = resp.get("Images", [])
        if not images:
            return 0, "/dev/sda1"
        image = images[0]
        root_device = image.get("RootDeviceName", "/dev/sda1")
        for mapping in image.get("BlockDeviceMappings", []):
            if mapping.get("DeviceName") == root_device:
                size = int((mapping.get("Ebs") or {}).get("VolumeSize") or 0)
                return size, root_device
        return 0, root_device
    except Exception as e:
        logger.warning("_get_ami_root_volume_gb error: %s", e)
        return 0, "/dev/sda1"


# ── Disk size budget for a fresh KeyFlow worker instance ─────────────────────
# OS + CUDA + PyTorch (already baked into AMI snapshot, reported by describe_images)
#   → read dynamically from AMI via _get_ami_root_volume_gb(); typically ~75 GB
#
# Additional space needed on top of the AMI snapshot:
_MODELS_GB = (
    0.5   # MatAnyone2 checkpoint (.pth, GitHub release)
  + 1.0   # BiRefNet General preset (HuggingFace safetensors)
  + 1.0   # BiRefNet Matting preset
  + 2.0   # GVM (HuggingFace geyongtao/gvm)
)  # ≈ 4.5 GB total model weights

_WORKSPACE_GB = 15   # uploads + inference outputs + temp files during processing

# Minimum requested volume = models + workspace; AMI snapshot floor is enforced
# separately via max(requested, ami_min) in launch_instance.
_VOLUME_REQUEST_GB = round(_MODELS_GB + _WORKSPACE_GB) + 10  # +10 safety margin → ~30 GB


def launch_instance(
    region: str,
    profile: str,
    instance_type: str = "g5.xlarge",
    ami_id: str = "",
    key_name: str = "",
    security_group_id: str = "",
    volume_gb: int = _VOLUME_REQUEST_GB,
    use_spot: bool = True,
) -> tuple[bool, str]:
    """Launch a new EC2 GPU instance. Returns (success, instance_id_or_error).

    The actual EBS volume size is max(volume_gb, ami_snapshot_size) so the disk
    always fits both the OS image and the model weights + working space.

    Args:
        ami_id: AMI ID to use. Required — returns error if empty.
        key_name: EC2 key pair name. Optional (instance will launch without SSH if omitted).
        security_group_id: Security group ID. Optional (uses VPC default if omitted).
        use_spot: If True, launch as Spot (cheaper). If False, launch as On-Demand.
    """
    if not ami_id:
        return False, "AMI ID is required. Detect or set an AMI in the Instance Manager first."
    try:
        ec2 = _get_ec2(profile, region)
        ami_min_gb, root_device = _get_ami_root_volume_gb(ec2, ami_id)
        # Ensure the volume fits: AMI snapshot floor OR models+workspace, whichever is larger
        actual_volume_gb = max(volume_gb + ami_min_gb, ami_min_gb)
        kwargs: dict = {
            "ImageId": ami_id,
            "InstanceType": instance_type,
            "MinCount": 1,
            "MaxCount": 1,
            "BlockDeviceMappings": [{
                "DeviceName": root_device,
                "Ebs": {"VolumeSize": actual_volume_gb, "VolumeType": "gp3", "DeleteOnTermination": True},
            }],
            "TagSpecifications": [{
                "ResourceType": "instance",
                "Tags": [{"Key": "Name", "Value": "KeyFlow-GPU"}],
            }],
        }
        if key_name:
            kwargs["KeyName"] = key_name
        if security_group_id:
            kwargs["SecurityGroupIds"] = [security_group_id]
        if use_spot:
            kwargs["InstanceMarketOptions"] = {
                "MarketType": "spot",
                "SpotOptions": {
                    "SpotInstanceType": "one-time",
                    "InstanceInterruptionBehavior": "terminate",
                },
            }
        resp = ec2.run_instances(**kwargs)
        instance_id = resp["Instances"][0]["InstanceId"]
        return True, instance_id
    except ImportError:
        return False, "boto3 not installed. Run: pip install boto3"
    except Exception as e:
        logger.warning("launch_instance error: %s", e)
        return False, str(e)


def check_environment_ssh(
    public_ip: str,
    ssh_key_path: str,
    ssh_user: str = "ubuntu",
    expected_revision: str | None = None,
) -> dict:
    """SSH into instance and inspect cloud-worker readiness and bundle revision."""
    key_path = os.path.expanduser(ssh_key_path)
    remote_script = r'''
python3 - <<'PY'
import importlib
import json
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

home = Path.home()
worker_root = home / "keyflow-worker"
venv_python = home / "keyflow-venv" / "bin" / "python3"
manifest_path = worker_root / ".keyflow_bundle_manifest.json"
required_files = [
    worker_root / "ec2_worker" / "worker.py",
    worker_root / "ec2_worker" / "download_models.py",
    worker_root / "app" / "__init__.py",
    worker_root / "app" / "services" / "model_service.py",
    worker_root / "app" / "services" / "birefnet_service.py",
    worker_root / "app" / "utils" / "__init__.py",
]
summary = {
    "python": sys.version.split()[0],
    "venv": venv_python.exists(),
    "cuda": False,
    "gpu": "",
    "torch_version": "",
    "fastapi": False,
    "deps_ok": False,
    "worker_files": all(path.exists() for path in required_files),
    "worker_running": False,
    "health_ok": False,
    "corridorkey_module": False,
    "health": {},
    "bundle_revision": "",
    "bundle_file_count": 0,
    "errors": [],
}

if manifest_path.is_file():
    try:
        manifest = json.loads(manifest_path.read_text())
        summary["bundle_revision"] = str(manifest.get("revision") or "")
        summary["bundle_file_count"] = int(manifest.get("file_count") or 0)
    except Exception as exc:
        summary["errors"].append(f"manifest: {exc}")

def run_venv(code: str):
    if not venv_python.exists():
        return 127, "venv missing"
    result = subprocess.run([str(venv_python), "-c", code], capture_output=True, text=True, timeout=90)
    return result.returncode, (result.stdout or "") + (result.stderr or "")

rc, out = run_venv(
    "import json, torch; "
    "print(json.dumps({'cuda': bool(torch.cuda.is_available()), 'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else '', 'torch': str(torch.__version__)}))"
)
if rc == 0:
    try:
        torch_info = json.loads(out.strip().splitlines()[-1])
        summary["cuda"] = bool(torch_info.get("cuda"))
        summary["gpu"] = str(torch_info.get("gpu") or "")
        summary["torch_version"] = str(torch_info.get("torch") or "")
    except Exception as exc:
        summary["errors"].append(f"torch-parse: {exc}")
else:
    summary["errors"].append(out.strip() or "torch probe failed")

rc, out = run_venv(
    "import importlib, json; mods=['fastapi','uvicorn','huggingface_hub','boto3','cv2','numpy','PIL','CorridorKeyModule']; "
    "status={m: True for m in mods}; "
    "[importlib.import_module(m) for m in mods]; "
    "print(json.dumps(status))"
)
if rc == 0:
    try:
        dep_info = json.loads(out.strip().splitlines()[-1])
        summary["fastapi"] = bool(dep_info.get("fastapi"))
        summary["corridorkey_module"] = bool(dep_info.get("CorridorKeyModule"))
        summary["deps_ok"] = all(bool(v) for v in dep_info.values())
    except Exception as exc:
        summary["errors"].append(f"deps-parse: {exc}")
else:
    summary["errors"].append(out.strip() or "dependency probe failed")

running = subprocess.run("pgrep -f 'python3 worker.py|uvicorn worker:app' >/dev/null", shell=True)
summary["worker_running"] = running.returncode == 0
if summary["worker_running"]:
    try:
        with urllib.request.urlopen("http://127.0.0.1:8080/health", timeout=4) as resp:
            payload = resp.read().decode("utf-8")
        health = json.loads(payload) if payload else {}
        summary["health_ok"] = True
        summary["health"] = health if isinstance(health, dict) else {}
    except Exception as exc:
        summary["errors"].append(f"health: {exc}")

# Get CorridorKey version on server (git revision)
try:
    ck_root = home / "CorridorKey"
    if ck_root.exists():
        result = subprocess.run(
            ["git", "-C", str(ck_root), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            summary["corridorkey_revision"] = result.stdout.strip()
        else:
            summary["corridorkey_revision"] = "git-error"
    else:
        summary["corridorkey_revision"] = "not-found"
except Exception as exc:
    summary["corridorkey_revision"] = f"error: {exc}"

print("__KEYFLOW_JSON_START__")
print(json.dumps(summary))
print("__KEYFLOW_JSON_END__")
PY
'''
    try:
        result = subprocess.run(
            ["ssh", "-i", key_path, "-o", "StrictHostKeyChecking=no",
             "-o", "ConnectTimeout=10", f"{ssh_user}@{public_ip}", "bash", "-s"],
            input=remote_script, capture_output=True, text=True, timeout=60,
        )
        output = result.stdout + result.stderr
        start_marker = "__KEYFLOW_JSON_START__"
        end_marker = "__KEYFLOW_JSON_END__"
        start = output.find(start_marker)
        end = output.find(end_marker)
        summary: dict = {}
        if start >= 0 and end > start:
            payload = output[start + len(start_marker):end].strip()
            summary = json.loads(payload)
        revision = str(summary.get("bundle_revision") or "")
        summary["revision_match"] = bool(expected_revision) and revision == expected_revision
        output_lines = [
            f"Python: {summary.get('python') or 'unknown'}",
            f"VENV: {'ok' if summary.get('venv') else 'missing'}",
            f"CUDA: {summary.get('cuda')}",
            f"GPU: {summary.get('gpu') or 'none'}",
            f"TORCH: {summary.get('torch_version') or 'unknown'}",
            f"FastAPI: {'ok' if summary.get('fastapi') else 'missing'}",
            f"CorridorKeyModule: {'ok' if summary.get('corridorkey_module') else 'missing'}",
            f"CorridorKey revision: {summary.get('corridorkey_revision') or 'unknown'}",
            f"Deps: {'ok' if summary.get('deps_ok') else 'missing'}",
            f"Worker files: {'ok' if summary.get('worker_files') else 'missing'}",
            f"Worker process: {'ok' if summary.get('worker_running') else 'not running'}",
            f"Health endpoint: {'ok' if summary.get('health_ok') else 'unavailable'}",
            f"Bundle revision: {revision or 'missing'}",
        ]
        if expected_revision:
            output_lines.append(f"Expected revision: {expected_revision}")
            output_lines.append(f"Revision match: {summary.get('revision_match')}")
        if summary.get("errors"):
            output_lines.append("Errors: " + "; ".join(str(x) for x in summary.get("errors") or []))
        return {
            "ok": result.returncode == 0 and bool(summary),
            "output": "\n".join(output_lines),
            "cuda": bool(summary.get("cuda")),
            "venv": bool(summary.get("venv")),
            "fastapi": bool(summary.get("fastapi")),
            "corridorkey_module": bool(summary.get("corridorkey_module")),
            "deps_ok": bool(summary.get("deps_ok")),
            "worker_files": bool(summary.get("worker_files")),
            "worker_running": bool(summary.get("worker_running")),
            "health_ok": bool(summary.get("health_ok")),
            "bundle_revision": revision,
            "revision_match": bool(summary.get("revision_match")),
            "health": summary.get("health") or {},
            "errors": summary.get("errors") or [],
        }
    except subprocess.TimeoutExpired:
        return {
            "ok": False,
            "output": "SSH timeout (60s)",
            "cuda": False,
            "venv": False,
            "fastapi": False,
            "corridorkey_module": False,
            "deps_ok": False,
            "worker_files": False,
            "worker_running": False,
            "health_ok": False,
            "bundle_revision": "",
            "revision_match": False,
            "health": {},
            "errors": ["SSH timeout (60s)"],
        }
    except Exception as e:
        return {
            "ok": False,
            "output": str(e),
            "cuda": False,
            "venv": False,
            "fastapi": False,
            "corridorkey_module": False,
            "deps_ok": False,
            "worker_files": False,
            "worker_running": False,
            "health_ok": False,
            "bundle_revision": "",
            "revision_match": False,
            "health": {},
            "errors": [str(e)],
        }


def prepare_cloud_worker_ssh(
    public_ip: str,
    ssh_key_path: str,
    ssh_user: str = "ubuntu",
    progress_callback=None,
) -> tuple[bool, str]:
    """Prepare a clean EC2 instance for KeyFlow cloud work and sync local worker files."""
    def _emit(message: str) -> None:
        if progress_callback is not None:
            try:
                progress_callback(str(message))
            except Exception:
                logger.exception("cloud bootstrap progress callback failed")

    key_path = os.path.expanduser(ssh_key_path)
    manifest = build_cloud_worker_bundle_manifest()
    root = _workspace_root()
    log_parts = [f"Local bundle revision: {manifest['revision']} ({manifest['file_count']} files)"]
    _emit(f"Local bundle revision: {manifest['revision']} ({manifest['file_count']} files)")

    with tempfile.TemporaryDirectory(prefix="keyflow-cloud-") as temp_dir:
        manifest_path = Path(temp_dir) / ".keyflow_bundle_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        _emit("Creating remote worker directories...")
        mkdir_cmd = [
            "ssh", "-i", key_path, "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=15",
            f"{ssh_user}@{public_ip}",
            "mkdir -p ~/keyflow-worker/ec2_worker ~/keyflow-worker/app/services ~/keyflow-worker/app/utils",
        ]
        ok, out = _run_local_command(mkdir_cmd, timeout=60)
        log_parts.append(out.strip())
        if not ok:
            return False, "\n".join(part for part in log_parts if part)

        sync_jobs = [
            (root / "ec2_worker", f"{_REMOTE_WORKER_ROOT}/ec2_worker/"),
            (root / "app" / "services", f"{_REMOTE_WORKER_ROOT}/app/services/"),
            (root / "app" / "utils", f"{_REMOTE_WORKER_ROOT}/app/utils/"),
            (root / "app" / "__init__.py", f"{_REMOTE_WORKER_ROOT}/app/"),
            (manifest_path, _REMOTE_WORKER_MANIFEST),
        ]
        for local_path, remote_path in sync_jobs:
            _emit(f"Syncing {Path(local_path).name} → {remote_path}")
            ok, out = _rsync_to_remote(Path(local_path), remote_path, ssh_user, public_ip, key_path)
            log_parts.append(out.strip())
            if not ok:
                return False, "\n".join(part for part in log_parts if part)

        local_gvm_core = _find_local_gvm_core_dir()
        if local_gvm_core is not None:
            _emit("Syncing local gvm_core package...")
            ok, out = _rsync_to_remote(local_gvm_core, f"{_REMOTE_WORKER_ROOT}/gvm_core/", ssh_user, public_ip, key_path)
            log_parts.append(out.strip())
            if not ok:
                return False, "\n".join(part for part in log_parts if part)
        else:
            log_parts.append("WARN: local gvm_core package not found; GVM jobs may stay unavailable")
            _emit("WARN: local gvm_core package not found; GVM jobs may stay unavailable")

        # Apply local patches on top of gvm_core (e.g. VAE encode chunking fix).
        patch_dir = root / "ec2_worker" / "_patch"
        if patch_dir.is_dir():
            _emit("Applying ec2_worker/_patch/ fixes...")
            ok, out = _rsync_to_remote(patch_dir, f"{_REMOTE_WORKER_ROOT}/", ssh_user, public_ip, key_path, delete=False)
            log_parts.append(out.strip())
            if ok:
                # Clear stale .pyc so Python picks up patched source files.
                _run_remote_command(public_ip, ssh_user, key_path,
                    f"find {_REMOTE_WORKER_ROOT}/gvm_core -name '*.pyc' -delete 2>/dev/null || true")
                _emit("✓ Patches applied")

    _emit("Running remote bootstrap: apt, venv, Python deps, original CorridorKey, worker start...")
    remote_script = f'''
set -e
export DEBIAN_FRONTEND=noninteractive
sudo apt-get update -qq
sudo apt-get install -y -qq git ffmpeg python3-pip python3-venv

[ -d ~/keyflow-venv ] || python3 -m venv ~/keyflow-venv
source ~/keyflow-venv/bin/activate
python3 -m pip install -q --upgrade pip setuptools wheel
python3 -m pip install -q torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
python3 -m pip install -q \
  opencv-python-headless Pillow imageio imageio-ffmpeg \
  numpy scipy tqdm timm==1.0.24 transformers kornia \
  huggingface-hub easydict diffusers peft accelerate \
  matplotlib av pims requests psutil fastapi uvicorn python-multipart boto3

if [ ! -d ~/CorridorKey/.git ]; then
    git clone --depth 1 https://github.com/nikopueringer/CorridorKey.git ~/CorridorKey
else
    git -C ~/CorridorKey fetch --depth 1 origin main
    git -C ~/CorridorKey checkout -f main
    git -C ~/CorridorKey reset --hard origin/main
fi
python3 -m pip install -q -e ~/CorridorKey --no-deps

cd {_REMOTE_WORKER_ROOT}/ec2_worker
pkill -f "python3 worker.py|uvicorn worker:app" 2>/dev/null || true
nohup ~/keyflow-venv/bin/python3 worker.py > {_REMOTE_WORKER_LOG} 2>&1 &
echo "Worker started (PID $!)"
sleep 8

set +e
python3 - <<'PY'
import json
import time
import urllib.request

last_error = ""
for _ in range(15):
    try:
        with urllib.request.urlopen("http://127.0.0.1:8080/health", timeout=4) as resp:
            payload = resp.read().decode("utf-8")
        print(payload)
        raise SystemExit(0)
    except Exception as exc:
        last_error = str(exc)
        time.sleep(2)
print(json.dumps({{"status": "error", "detail": last_error}}))
raise SystemExit(1)
PY
HEALTH_EXIT=$?
if [ $HEALTH_EXIT -ne 0 ]; then
    echo "--- worker.log (last 60 lines) ---"
    tail -60 {_REMOTE_WORKER_LOG} 2>/dev/null || echo "(worker.log not found)"
    exit 1
fi
'''
    try:
        result = subprocess.run(
            ["ssh", "-i", key_path, "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=15", f"{ssh_user}@{public_ip}", "bash", "-s"],
            input=remote_script,
            capture_output=True,
            text=True,
            timeout=1800,
        )
        output = (result.stdout or "") + (result.stderr or "")
        log_parts.append(output.strip())
        if result.returncode != 0:
            return False, "\n".join(part for part in log_parts if part)
    except subprocess.TimeoutExpired:
        return False, "\n".join(log_parts + ["Remote bootstrap timed out after 1800s"])
    except Exception as exc:
        return False, "\n".join(log_parts + [str(exc)])

    _emit("Verifying remote worker health and bundle revision...")
    verify = check_environment_ssh(public_ip, ssh_key_path, ssh_user, expected_revision=manifest["revision"])
    log_parts.append(verify.get("output", "").strip())
    
    # Auto-sync CorridorKey from GitHub to server
    _emit("Syncing CorridorKey from GitHub...")
    sync_ck_cmd = '''
set -e
source ~/keyflow-venv/bin/activate
git -C ~/CorridorKey fetch --depth 1 origin main
git -C ~/CorridorKey reset --hard origin/main
python3 -m pip install -q -e ~/CorridorKey --no-deps
echo "CorridorKey synced from GitHub"
'''
    ok, out = _run_remote_command(public_ip, ssh_user, key_path, sync_ck_cmd)
    if ok:
        _emit("✓ CorridorKey synced from GitHub")
        log_parts.append(out.strip())
    else:
        _emit(f"⚠ CorridorKey sync warning: {out}")
        log_parts.append(out.strip())
    
    ready = all([
        verify.get("venv"),
        verify.get("cuda"),
        verify.get("fastapi"),
        verify.get("corridorkey_module"),
        verify.get("deps_ok"),
        verify.get("worker_files"),
        verify.get("worker_running"),
        verify.get("health_ok"),
        verify.get("revision_match"),
    ])
    return ready, "\n".join(part for part in log_parts if part)


def install_environment_ssh(
    public_ip: str,
    ssh_key_path: str,
    ssh_user: str = "ubuntu",
    progress_callback=None,
) -> tuple[bool, str]:
    """Backward-compatible alias for the full cloud worker bootstrap routine."""
    return prepare_cloud_worker_ssh(public_ip, ssh_key_path, ssh_user, progress_callback=progress_callback)


# ── Watchdog: автоотключение при простое ───────────────────────────────────

_WATCHDOG_SCRIPT = r"""#!/usr/bin/env python3
"""
_WATCHDOG_SCRIPT += r'''"""KeyFlow GPU idle watchdog.
Checks GPU utilization every minute. If below threshold for IDLE_MINUTES → shutdown.
Installed as: /usr/local/bin/keyflow-watchdog.py
Service: /etc/systemd/system/keyflow-watchdog.service
"""
import subprocess, time, sys, logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [watchdog] %(message)s",
    handlers=[
        logging.FileHandler("/var/log/keyflow-watchdog.log"),
        logging.StreamHandler(sys.stdout),
    ],
    force=True,
)
log = logging.getLogger(__name__)

IDLE_THRESHOLD_PCT = int("__THRESHOLD__")   # GPU util % below which = idle
IDLE_MINUTES       = int("__MINUTES__")     # consecutive idle minutes before shutdown
CHECK_INTERVAL_SEC = 60

idle_count = 0

log.info("Watchdog started. threshold=%d%% idle_minutes=%d", IDLE_THRESHOLD_PCT, IDLE_MINUTES)

while True:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            timeout=10,
        ).decode().strip()
        util = int(out.split("\n")[0].strip())
        if util < IDLE_THRESHOLD_PCT:
            idle_count += 1
            log.info("GPU idle %d%% — idle streak: %d/%d min", util, idle_count, IDLE_MINUTES)
        else:
            if idle_count > 0:
                log.info("GPU active %d%% — resetting idle counter", util)
            idle_count = 0
    except Exception as e:
        log.warning("nvidia-smi error: %s — skipping", e)

    if idle_count >= IDLE_MINUTES:
        log.info("Idle limit reached (%d min). Shutting down.", IDLE_MINUTES)
        subprocess.run(["sudo", "shutdown", "-h", "now"])
        break

    time.sleep(CHECK_INTERVAL_SEC)
'''

_WATCHDOG_SERVICE = """[Unit]
Description=KeyFlow GPU idle watchdog
After=multi-user.target

[Service]
Type=simple
ExecStart=/usr/bin/python3 /usr/local/bin/keyflow-watchdog.py
Restart=on-failure
RestartSec=30
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
"""

def get_system_logs_http(
    base_url: str,
    after_seq: int = 0,
    lines: int = 200,
    timeout: int = 5,
) -> tuple[list[str], int]:
    """Fetch worker log lines from /system/logs.

    Returns (new_lines, next_seq). On error returns ([], after_seq).
    """
    import urllib.request, json as _json
    url = f"{base_url.rstrip('/')}/system/logs?after_seq={after_seq}&lines={lines}"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            data = _json.loads(r.read())
        return data.get("lines", []), int(data.get("next_seq", after_seq))
    except Exception:
        return [], after_seq


def install_watchdog_ssh(
    public_ip: str,
    ssh_key_path: str,
    ssh_user: str = "ubuntu",
    idle_minutes: int = 15,
    gpu_threshold_pct: int = 5,
) -> tuple[bool, str]:
    """Install idle-shutdown watchdog on EC2 instance via SSH.

    The watchdog runs as a systemd service and shuts down the instance
    if GPU utilization stays below gpu_threshold_pct for idle_minutes.
    """
    import subprocess
    import os

    key_path = os.path.expanduser(ssh_key_path)
    script = _WATCHDOG_SCRIPT.replace("__THRESHOLD__", str(gpu_threshold_pct)).replace(
        "__MINUTES__", str(idle_minutes)
    )
    service = _WATCHDOG_SERVICE

    install_cmd = rf"""
cat > /tmp/keyflow-watchdog.py << 'PYEOF'
{script}
PYEOF
sudo mv /tmp/keyflow-watchdog.py /usr/local/bin/keyflow-watchdog.py
sudo chmod +x /usr/local/bin/keyflow-watchdog.py

cat > /tmp/keyflow-watchdog.service << 'SVCEOF'
{service}
SVCEOF
sudo mv /tmp/keyflow-watchdog.service /etc/systemd/system/keyflow-watchdog.service

sudo systemd-analyze verify /etc/systemd/system/keyflow-watchdog.service 2>&1 || true
sudo systemctl daemon-reload
sudo systemctl enable keyflow-watchdog
sudo systemctl restart keyflow-watchdog
sudo systemctl status keyflow-watchdog --no-pager
sudo systemctl is-active keyflow-watchdog
"""
    try:
        result = subprocess.run(
            ["ssh", "-i", key_path,
             "-o", "StrictHostKeyChecking=no",
             "-o", "ConnectTimeout=15",
             "-o", "ServerAliveInterval=10",
             f"{ssh_user}@{public_ip}", install_cmd],
            capture_output=True, text=True, timeout=90,
        )
        output = result.stdout + result.stderr
        # Определяем, что SSH ещё не готов (connection refused / no route / timeout)
        ssh_not_ready = any(s in output.lower() for s in (
            "connection refused", "no route to host", "connection timed out",
            "network is unreachable",
        ))
        if ssh_not_ready:
            return False, f"SSH not ready yet: {output[:300]}"
        return result.returncode == 0, output
    except subprocess.TimeoutExpired:
        return False, "SSH timeout (90s)"
    except Exception as e:
        return False, str(e)


def get_watchdog_status_ssh(
    public_ip: str,
    ssh_key_path: str,
    ssh_user: str = "ubuntu",
) -> tuple[bool, str]:
    """Return (installed, status_output) of watchdog service."""
    import subprocess
    import os

    key_path = os.path.expanduser(ssh_key_path)
    try:
        result = subprocess.run(
            ["ssh", "-i", key_path, "-o", "StrictHostKeyChecking=no",
             "-o", "ConnectTimeout=10", f"{ssh_user}@{public_ip}",
             "systemctl is-active keyflow-watchdog 2>&1; "
             "systemctl status keyflow-watchdog --no-pager -n 10 2>&1 || true; "
             "tail -20 /var/log/keyflow-watchdog.log 2>/dev/null || echo '(no log yet)'"],
            capture_output=True, text=True, timeout=20,
        )
        output = result.stdout + result.stderr
        installed = "keyflow-watchdog" in output and "not-found" not in output
        return installed, output
    except subprocess.TimeoutExpired:
        return False, "SSH timeout"
    except Exception as e:
        return False, str(e)


def get_monthly_costs(profile: str | None = None) -> dict:
    """Fetch AWS spend data via Cost Explorer API.

    Returns dict with keys:
        month     – month-to-date cost string e.g. "$12.34", or None on error
        yesterday – yesterday's cost string, or None
        forecast  – end-of-month forecast string, or None
        error     – error message string if fetch failed, else None
    Cost Explorer endpoint is always us-east-1 regardless of instance region.
    """
    from datetime import date, timedelta
    import calendar

    try:
        import boto3
    except ImportError:
        return {"month": None, "yesterday": None, "forecast": None, "error": "boto3 not installed"}

    try:
        session = boto3.Session(profile_name=profile or None)
        ce = session.client("ce", region_name="us-east-1")

        today = date.today()
        first_of_month = today.replace(day=1)
        yesterday = today - timedelta(days=1)
        last_day_of_month = date(
            today.year, today.month, calendar.monthrange(today.year, today.month)[1]
        )
        month_end_excl = last_day_of_month + timedelta(days=1)

        # Month-to-date
        resp_month = ce.get_cost_and_usage(
            TimePeriod={"Start": first_of_month.isoformat(), "End": today.isoformat()},
            Granularity="MONTHLY",
            Metrics=["UnblendedCost"],
        )
        month_amount = float(
            resp_month["ResultsByTime"][0]["Total"]["UnblendedCost"]["Amount"]
        ) if resp_month.get("ResultsByTime") else 0.0

        # Yesterday's daily cost
        resp_yday = ce.get_cost_and_usage(
            TimePeriod={"Start": yesterday.isoformat(), "End": today.isoformat()},
            Granularity="DAILY",
            Metrics=["UnblendedCost"],
        )
        yesterday_amount = float(
            resp_yday["ResultsByTime"][0]["Total"]["UnblendedCost"]["Amount"]
        ) if resp_yday.get("ResultsByTime") else 0.0

        # Forecast to end of month (only if we're not already on the last day)
        forecast_amount = None
        if today < last_day_of_month:
            try:
                resp_fc = ce.get_cost_forecast(
                    TimePeriod={"Start": today.isoformat(), "End": month_end_excl.isoformat()},
                    Granularity="MONTHLY",
                    Metric="UNBLENDED_COST",
                )
                forecast_amount = float(resp_fc["Total"]["Amount"])
            except Exception:
                pass

        return {
            "month": f"${month_amount:.2f}",
            "yesterday": f"${yesterday_amount:.2f}",
            "forecast": f"${forecast_amount:.2f}" if forecast_amount is not None else None,
            "error": None,
        }
    except Exception as exc:
        logger.debug("get_monthly_costs error: %s", exc)
        return {"month": None, "yesterday": None, "forecast": None, "error": str(exc)}


# vCPU counts for known GPU instance types
_INSTANCE_VCPUS: dict[str, int] = {
    "g4dn.xlarge":   4,
    "g4dn.2xlarge":  8,
    "g4dn.4xlarge":  16,
    "g5.xlarge":     4,
    "g5.2xlarge":    8,
    "g5.4xlarge":    16,
    "g5.12xlarge":   48,
    "g6.xlarge":     4,
    "g6.2xlarge":    8,
    "g6.4xlarge":    16,
    "p3.2xlarge":    8,
    "p3.8xlarge":    32,
}

_INSTANCE_GPUS: dict[str, int] = {
    "g4dn.xlarge":   1,
    "g4dn.2xlarge":  1,
    "g4dn.4xlarge":  1,
    "g5.xlarge":     1,
    "g5.2xlarge":    1,
    "g5.4xlarge":    1,
    "g5.12xlarge":   4,
    "g6.xlarge":     1,
    "g6.2xlarge":    1,
    "g6.4xlarge":    1,
    "p3.2xlarge":    1,
    "p3.8xlarge":    4,
}

_INSTANCE_GPU_SPECS: dict[str, tuple[str, int]] = {
    "g4dn.xlarge":   ("T4", 16),
    "g4dn.2xlarge":  ("T4", 16),
    "g4dn.4xlarge":  ("T4", 16),
    "g5.xlarge":     ("A10G", 24),
    "g5.2xlarge":    ("A10G", 24),
    "g5.4xlarge":    ("A10G", 24),
    "g5.12xlarge":   ("A10G", 24),
    "g6.xlarge":     ("L4", 24),
    "g6.2xlarge":    ("L4", 24),
    "g6.4xlarge":    ("L4", 24),
    "p3.2xlarge":    ("V100", 16),
    "p3.8xlarge":    ("V100", 16),
}

_ALL_GPU_TYPES = sorted(_INSTANCE_VCPUS.keys())

_GPU_VT_SPOT_QUOTA_CODE = "L-3819A6DF"
_GPU_VT_ONDEMAND_QUOTA_CODE = "L-DB2E81BA"
_KEYFLOW_MIN_VRAM_GB = 24
_KEYFLOW_MAX_GPUS = 1
_KEYFLOW_PREFERRED_FAMILIES: dict[str, int] = {
    "g5": 10,
    "g6": 20,
    "g6e": 30,
    "g6f": 40,
    "g7e": 50,
    "p4": 60,
    "p5": 70,
}


def _mib_to_marketing_gb(memory_mib: int) -> int:
    """Convert AWS MiB memory values to marketing-style decimal GB labels."""
    bytes_total = memory_mib * 1024 * 1024
    return max(1, round(bytes_total / 1_000_000_000))


def _instance_type_sort_key(instance_type: str) -> tuple[int, int, str]:
    family_order = {
        "g4dn": 10,
        "g5": 20,
        "g6": 30,
        "g6e": 40,
        "g6f": 50,
        "p3": 60,
        "p4": 70,
        "p5": 80,
    }
    family = instance_type.split(".", 1)[0]
    return (family_order.get(family, 999), _INSTANCE_VCPUS.get(instance_type, 9999), instance_type)


def _is_keyflow_suitable_instance_type(instance_type: str) -> bool:
    """Return True for instance types that are a pragmatic fit for KeyFlow Studio.

    The cloud worker runs a single inference process, so multi-GPU variants are
    poor value. We also prefer modern NVIDIA inference families with at least
    24 GB VRAM, which matches the current g5/g6 baseline used by the app.
    """
    family = instance_type.split(".", 1)[0]
    if family not in _KEYFLOW_PREFERRED_FAMILIES:
        return False

    gpu_count = _INSTANCE_GPUS.get(instance_type)
    if gpu_count is not None and gpu_count > _KEYFLOW_MAX_GPUS:
        return False

    gpu_spec = _INSTANCE_GPU_SPECS.get(instance_type)
    if gpu_spec is not None and gpu_spec[1] < _KEYFLOW_MIN_VRAM_GB:
        return False

    return True


def _keyflow_instance_type_sort_key(instance_type: str) -> tuple[int, int, str]:
    family = instance_type.split(".", 1)[0]
    return (
        _KEYFLOW_PREFERRED_FAMILIES.get(family, 999),
        _INSTANCE_VCPUS.get(instance_type, 9999),
        instance_type,
    )


def _filter_keyflow_supported_instance_types(instance_types: list[str]) -> list[str]:
    """Return only KeyFlow-suitable types when the region offers them.

    If AWS returns no types that match the current suitability profile, keep the
    original ordered list so the UI does not become empty for niche regions.
    """
    ordered = sorted(instance_types, key=_instance_type_sort_key)
    suitable = [instance_type for instance_type in ordered if _is_keyflow_suitable_instance_type(instance_type)]
    if suitable:
        return sorted(suitable, key=_keyflow_instance_type_sort_key)
    return ordered


def _update_instance_type_metadata(ec2, instance_types: list[str]) -> None:
    if not instance_types:
        return
    for start in range(0, len(instance_types), 100):
        chunk = instance_types[start:start + 100]
        resp = ec2.describe_instance_types(InstanceTypes=chunk)
        for info in resp.get("InstanceTypes", []):
            instance_type = info.get("InstanceType")
            if not instance_type:
                continue

            default_vcpus = (info.get("VCpuInfo") or {}).get("DefaultVCpus")
            if isinstance(default_vcpus, int):
                _INSTANCE_VCPUS[instance_type] = default_vcpus

            gpus = (info.get("GpuInfo") or {}).get("Gpus") or []
            if not gpus:
                continue

            _INSTANCE_GPUS[instance_type] = len(gpus)
            first_gpu = gpus[0]
            gpu_name = str(first_gpu.get("Name") or "GPU")
            memory_mib = ((first_gpu.get("MemoryInfo") or {}).get("SizeInMiB"))
            if isinstance(memory_mib, int) and memory_mib > 0:
                _INSTANCE_GPU_SPECS[instance_type] = (gpu_name, _mib_to_marketing_gb(memory_mib))


def get_available_gpu_types(
    region: str,
    profile: str | None = None,
    use_spot: bool = True,
) -> list[str]:
    """Return GPU instance types actually offered in the region.

    For Spot, additionally filters by the account's Spot G/VT vCPU quota.
    Falls back to the built-in catalog if the AWS metadata query fails.
    """
    try:
        ec2 = _get_ec2(profile, region)
        paginator = ec2.get_paginator("describe_instance_type_offerings")
        offered_types: set[str] = set()
        for page in paginator.paginate(LocationType="region", Filters=[{"Name": "location", "Values": [region]}]):
            for offering in page.get("InstanceTypeOfferings", []):
                instance_type = str(offering.get("InstanceType") or "")
                if instance_type.startswith(("g", "p")):
                    offered_types.add(instance_type)

        if not offered_types:
            fallback_types = get_available_spot_types(region, profile) if use_spot else _ALL_GPU_TYPES
            return _filter_keyflow_supported_instance_types(fallback_types)

        ordered_types = sorted(offered_types, key=_instance_type_sort_key)
        _update_instance_type_metadata(ec2, ordered_types)
        filtered_types = _filter_keyflow_supported_instance_types(ordered_types)

        quota = get_gpu_vcpu_quota(region, profile, use_spot=use_spot)
        if quota is None:
            return filtered_types
        return [instance_type for instance_type in filtered_types if _INSTANCE_VCPUS.get(instance_type, 9999) <= quota]
    except Exception as exc:
        logger.debug("get_available_gpu_types error: %s", exc)
        fallback_types = get_available_spot_types(region, profile) if use_spot else _ALL_GPU_TYPES
        return _filter_keyflow_supported_instance_types(fallback_types)


def format_instance_type_label(instance_type: str) -> str:
    """Return a user-facing label like 'g5.xlarge (4 vCPU, A10G, 24 GB VRAM)'."""
    vcpus = _INSTANCE_VCPUS.get(instance_type)
    gpus = _INSTANCE_GPUS.get(instance_type)
    gpu_spec = _INSTANCE_GPU_SPECS.get(instance_type)
    parts: list[str] = []
    if vcpus is not None:
        parts.append(f"{vcpus} vCPU")

    if gpu_spec is not None:
        gpu_name, gpu_vram_gb = gpu_spec
        if gpus is not None and gpus > 1:
            parts.append(f"{gpus}x {gpu_name}")
            parts.append(f"{gpu_vram_gb} GB VRAM each")
        else:
            parts.append(gpu_name)
            parts.append(f"{gpu_vram_gb} GB VRAM")
    elif gpus is not None:
        parts.append(f"{gpus} GPU")

    if not parts:
        return instance_type
    return f"{instance_type} ({', '.join(parts)})"


def _get_ec2_service_quota(region: str, profile: str | None, quota_code: str) -> int | None:
    """Return an EC2 service quota value as int, or None on error."""
    try:
        import boto3
        session = boto3.Session(profile_name=profile or None, region_name=region)
        sq = session.client("service-quotas")
        resp = sq.get_service_quota(ServiceCode="ec2", QuotaCode=quota_code)
        return int(resp["Quota"]["Value"])
    except ImportError:
        return None
    except Exception as exc:
        logger.debug("_get_ec2_service_quota error for %s: %s", quota_code, exc)
        return None


def get_spot_vcpu_quota(region: str, profile: str | None = None) -> int | None:
    """Return the Spot vCPU quota for G/VT instances.

    Returns the quota value as int, or None on error.
    AWS Service Quotas: 'All G and VT Spot Instance Requests' = L-3819A6DF
    """
    return _get_ec2_service_quota(region, profile, _GPU_VT_SPOT_QUOTA_CODE)


def get_ondemand_vcpu_quota(region: str, profile: str | None = None) -> int | None:
    """Return the On-Demand vCPU quota for G/VT instances.

    AWS Service Quotas: 'Running On-Demand G and VT instances' = L-DB2E81BA
    """
    return _get_ec2_service_quota(region, profile, _GPU_VT_ONDEMAND_QUOTA_CODE)


def get_gpu_vcpu_quota(region: str, profile: str | None = None, use_spot: bool = True) -> int | None:
    """Return the relevant G/VT quota for the selected market type."""
    if use_spot:
        return get_spot_vcpu_quota(region, profile)
    return get_ondemand_vcpu_quota(region, profile)


def check_credentials(profile: str | None = None, region: str = "us-east-1") -> tuple[bool, str]:
    """Check whether AWS credentials for the given profile are valid.

    Makes a lightweight STS get-caller-identity call.
    Returns (ok, message). message is the account/ARN on success or an error description.
    """
    try:
        sts = _get_sts(profile, region)
        identity = sts.get_caller_identity()
        account = identity.get("Account", "")
        arn = identity.get("Arn", "")
        return True, f"{arn}  (account: {account})"
    except ImportError:
        return False, "boto3 not installed — run: pip install boto3"
    except Exception as e:
        msg = str(e)
        # Provide user-friendly summaries for common errors
        if "NoCredentialsError" in type(e).__name__ or "Unable to locate credentials" in msg:
            return False, "no_credentials"
        if "InvalidClientTokenId" in msg or "AuthFailure" in msg:
            return False, "invalid_credentials"
        if "ProfileNotFound" in type(e).__name__ or "could not be found" in msg.lower():
            return False, "profile_not_found"
        return False, msg


def save_aws_credentials_to_file(
    access_key_id: str,
    secret_access_key: str,
    profile: str = "keyflow",
) -> tuple[bool, str]:
    """Write credentials to ~/.aws/credentials under the given profile.

    Creates the file (and ~/.aws directory) if they do not exist.
    Returns (ok, message).
    """
    import configparser
    import os
    import stat

    creds_path = os.path.expanduser("~/.aws/credentials")
    aws_dir = os.path.dirname(creds_path)

    try:
        os.makedirs(aws_dir, mode=0o700, exist_ok=True)

        cfg = configparser.ConfigParser()
        if os.path.exists(creds_path):
            cfg.read(creds_path)

        section = profile or "default"
        if not cfg.has_section(section):
            cfg.add_section(section)
        cfg.set(section, "aws_access_key_id", access_key_id.strip())
        cfg.set(section, "aws_secret_access_key", secret_access_key.strip())

        with open(creds_path, "w") as f:
            cfg.write(f)
        # Restrict permissions: owner read/write only
        os.chmod(creds_path, stat.S_IRUSR | stat.S_IWUSR)
        return True, creds_path
    except Exception as e:
        logger.warning("save_aws_credentials_to_file error: %s", e)
        return False, str(e)


def get_available_spot_types(region: str, profile: str | None = None) -> list[str]:
    """Return list of GPU instance types available within the account's Spot vCPU quota.

    Falls back to the full list if quota cannot be fetched.
    """
    quota = get_spot_vcpu_quota(region, profile)
    if quota is None:
        return _filter_keyflow_supported_instance_types(_ALL_GPU_TYPES)
    return _filter_keyflow_supported_instance_types(
        [t for t, vcpus in _INSTANCE_VCPUS.items() if vcpus <= quota]
    )


def get_spot_price(instance_type: str, region: str, profile: str | None = None) -> dict:
    """Return current Spot price for the given instance type.

    Returns dict with keys:
        price  – float price per hour, or None on error
        az     – availability zone with the cheapest price
        error  – error string or None
    """
    try:
        from datetime import datetime, timezone
        ec2 = _get_ec2(profile, region)
        resp = ec2.describe_spot_price_history(
            InstanceTypes=[instance_type],
            ProductDescriptions=["Linux/UNIX"],
            StartTime=datetime.now(timezone.utc),
            MaxResults=10,
        )
        history = resp.get("SpotPriceHistory", [])
        if not history:
            return {"price": None, "az": None, "error": "No Spot price data"}
        cheapest = min(history, key=lambda x: float(x["SpotPrice"]))
        return {
            "price": float(cheapest["SpotPrice"]),
            "az": cheapest["AvailabilityZone"],
            "error": None,
        }
    except ImportError:
        return {"price": None, "az": None, "error": "boto3 not installed"}
    except Exception as exc:
        logger.debug("get_spot_price error: %s", exc)
        return {"price": None, "az": None, "error": str(exc)}


def get_ondemand_price(instance_type: str, region: str, profile: str | None = None) -> dict:
    """Return current On-Demand hourly price for the given instance type.

    Returns dict with keys:
        price  – float price per hour, or None on error
        error  – error string or None
    """
    try:
        import boto3

        session = boto3.Session(profile_name=profile or None, region_name=region)
        pricing = session.client("pricing", region_name="us-east-1")
        resp = pricing.get_products(
            ServiceCode="AmazonEC2",
            Filters=[
                {"Type": "TERM_MATCH", "Field": "instanceType", "Value": instance_type},
                {"Type": "TERM_MATCH", "Field": "operatingSystem", "Value": "Linux"},
                {"Type": "TERM_MATCH", "Field": "preInstalledSw", "Value": "NA"},
                {"Type": "TERM_MATCH", "Field": "tenancy", "Value": "Shared"},
                {"Type": "TERM_MATCH", "Field": "capacitystatus", "Value": "Used"},
                {"Type": "TERM_MATCH", "Field": "regionCode", "Value": region},
            ],
            MaxResults=10,
        )
        for raw in resp.get("PriceList", []):
            data = json.loads(raw)
            ondemand_terms = data.get("terms", {}).get("OnDemand", {})
            for term in ondemand_terms.values():
                for dim in term.get("priceDimensions", {}).values():
                    if dim.get("unit") != "Hrs":
                        continue
                    price_str = dim.get("pricePerUnit", {}).get("USD")
                    if price_str:
                        return {"price": float(price_str), "error": None}
        return {"price": None, "error": "No On-Demand price data"}
    except ImportError:
        return {"price": None, "error": "boto3 not installed"}
    except Exception as exc:
        logger.debug("get_ondemand_price error: %s", exc)
        return {"price": None, "error": str(exc)}

