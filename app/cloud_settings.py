"""AWS Cloud settings helpers for KeyFlow Studio."""

from __future__ import annotations

from app.settings import get_app_settings


DEFAULTS = {
    "cloud/instance_id": "",
    "cloud/api_host": "",          # last-known public IP/host — auto-updated on each poll
    "cloud/region": "eu-west-1",
    "cloud/ssh_key_path": "~/.ssh/keyflow-gpu.pem",
    "cloud/ssh_user": "ubuntu",
    "cloud/aws_profile": "keyflow",
    "cloud/enabled": False,
    "cloud/watchdog_idle_min": 15,
    "cloud/watchdog_gpu_pct": 5,
}


def get_cloud_setting(key: str):
    settings = get_app_settings()
    default = DEFAULTS.get(key)
    if isinstance(default, bool):
        return settings.value(key, default, type=bool)
    return settings.value(key, default)


def set_cloud_setting(key: str, value) -> None:
    settings = get_app_settings()
    settings.setValue(key, value)
    settings.sync()


def save_cloud_settings(
    instance_id: str,
    region: str,
    ssh_key_path: str,
    ssh_user: str,
    aws_profile: str,
    enabled: bool,
    watchdog_idle_min: int = 15,
    watchdog_gpu_pct: int = 5,
    api_host: str = "",
) -> None:
    settings = get_app_settings()
    settings.setValue("cloud/instance_id", instance_id.strip())
    settings.setValue("cloud/api_host", api_host.strip())
    settings.setValue("cloud/region", region.strip())
    settings.setValue("cloud/ssh_key_path", ssh_key_path.strip())
    settings.setValue("cloud/ssh_user", ssh_user.strip())
    settings.setValue("cloud/aws_profile", aws_profile.strip())
    settings.setValue("cloud/enabled", enabled)
    settings.setValue("cloud/watchdog_idle_min", watchdog_idle_min)
    settings.setValue("cloud/watchdog_gpu_pct", watchdog_gpu_pct)
    settings.sync()


# ── Per-profile/region launch config (AMI, key pair, security group) ─────────

def _launch_cfg_key(prefix: str, profile: str, region: str) -> str:
    safe_p = (profile or "default").replace("/", "_").replace("\\", "_")
    safe_r = (region or "us-east-1").replace("-", "_")
    return f"cloud/{prefix}/{safe_p}/{safe_r}"


def get_ami_for_profile_region(profile: str, region: str) -> str:
    return str(get_app_settings().value(_launch_cfg_key("ami", profile, region), "") or "")


def set_ami_for_profile_region(profile: str, region: str, ami_id: str) -> None:
    s = get_app_settings()
    s.setValue(_launch_cfg_key("ami", profile, region), ami_id.strip())
    s.sync()


def get_key_name_for_profile_region(profile: str, region: str) -> str:
    return str(get_app_settings().value(_launch_cfg_key("key_name", profile, region), "") or "")


def set_key_name_for_profile_region(profile: str, region: str, key_name: str) -> None:
    s = get_app_settings()
    s.setValue(_launch_cfg_key("key_name", profile, region), key_name.strip())
    s.sync()


def get_sg_for_profile_region(profile: str, region: str) -> str:
    return str(get_app_settings().value(_launch_cfg_key("sg", profile, region), "") or "")


def set_sg_for_profile_region(profile: str, region: str, sg_id: str) -> None:
    s = get_app_settings()
    s.setValue(_launch_cfg_key("sg", profile, region), sg_id.strip())
    s.sync()
