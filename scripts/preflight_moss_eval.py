#!/usr/bin/env python3
"""Read-only readiness check for the MOSS/AISHELL-4 evaluation path."""
from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import requests
import soundfile as sf


@dataclass(frozen=True)
class VllmProfile:
    max_upload_mb: float
    max_duration_sec: float
    max_num_batched_tokens: int


def _flag(argv: list[str], name: str, default: str) -> str:
    for index, value in enumerate(argv):
        if value == name and index + 1 < len(argv):
            return argv[index + 1]
        if value.startswith(f"{name}="):
            return value.split("=", 1)[1]
    return default


def parse_vllm_profile(*, environ: dict[str, str], argv: list[str]) -> VllmProfile:
    return VllmProfile(
        max_upload_mb=float(environ.get("VLLM_MAX_AUDIO_CLIP_FILESIZE_MB", "25")),
        max_duration_sec=float(environ.get("VLLM_MAX_AUDIO_DECODE_DURATION_S", "600")),
        max_num_batched_tokens=int(_flag(argv, "--max-num-batched-tokens", "2048")),
    )


def validate_sample(sample: dict[str, Any], profile: VllmProfile) -> list[dict[str, Any]]:
    checks = (
        (float(sample["size_mb"]), profile.max_upload_mb, "audio_upload_limit_insufficient"),
        (float(sample["duration_sec"]), profile.max_duration_sec, "audio_duration_limit_insufficient"),
        (int(sample["estimated_audio_tokens"]), profile.max_num_batched_tokens, "encoder_capacity_insufficient"),
    )
    return [
        {"code": code, "required": required, "configured": configured}
        for required, configured, code in checks
        if required > configured
    ]


def _read_process(pid: int) -> tuple[dict[str, str], list[str]]:
    proc = Path("/proc") / str(pid)
    environment = {}
    for raw in (proc / "environ").read_bytes().split(b"\0"):
        if b"=" in raw:
            key, value = raw.split(b"=", 1)
            environment[key.decode(errors="replace")] = value.decode(errors="replace")
    argv = [item.decode(errors="replace") for item in (proc / "cmdline").read_bytes().split(b"\0") if item]
    return environment, argv


def _sample_info(path: Path) -> dict[str, Any]:
    info = sf.info(path)
    duration = float(info.duration)
    return {
        "path": str(path),
        "size_mb": path.stat().st_size / (1024 * 1024),
        "duration_sec": duration,
        "estimated_audio_tokens": math.ceil(duration * 12.5),
    }


def _writable(path: Path) -> bool:
    path.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=".retrace-preflight-", dir=path)
    os.close(descriptor)
    Path(name).unlink()
    return True


def _family(identity: dict[str, Any]) -> str:
    backend = str(identity.get("backend") or "").lower()
    if backend.startswith("moss"):
        return "moss"
    if backend.startswith(("qwen-omni", "qwen3-omni")):
        return "qwen-omni"
    return backend


def run_preflight(args: argparse.Namespace) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    failures: list[dict[str, Any]] = []
    try:
        environment, argv = _read_process(args.vllm_pid)
        profile = parse_vllm_profile(environ=environment, argv=argv)
        checks["vllm_profile"] = asdict(profile)
    except Exception as exc:  # noqa: BLE001
        profile = VllmProfile(0, 0, 0)
        failures.append({"code": "vllm_process_unreadable", "detail": str(exc)})

    samples = [_sample_info(path) for suffix in ("*.wav", "*.flac") for path in sorted(args.wav_dir.glob(suffix))]
    checks["samples"] = samples
    if not samples:
        failures.append({"code": "no_audio_samples", "detail": str(args.wav_dir)})
    for sample in samples:
        failures.extend({**failure, "sample": sample["path"]} for failure in validate_sample(sample, profile))

    try:
        response = requests.get(f"{args.base_url.rstrip('/')}/v1/models", timeout=10)
        response.raise_for_status()
        models = response.json().get("data") or []
        served = [str(item.get("id")) for item in models if isinstance(item, dict)]
        checks["served_models"] = served
        if "MOSS-Transcribe-Diarize" not in served:
            failures.append({"code": "moss_model_not_served", "served_models": served})
    except Exception as exc:  # noqa: BLE001
        failures.append({"code": "moss_endpoint_unready", "detail": str(exc)})

    try:
        response = requests.get(f"{args.retrace_base_url.rstrip('/')}/api/integrations/status", timeout=10)
        response.raise_for_status()
        status = response.json()
        checks["retrace_integrations"] = status
        if not (status.get("deepseek") or {}).get("ready"):
            failures.append({"code": "deepseek_not_ready"})
        tools = status.get("audio_tools") or {}
        families = {_family(tools.get(name) or {}) for name in ("first_pass", "targeted_verifier", "open_relistener")}
        if families != {"moss"}:
            failures.append({"code": "audio_tool_family_mismatch", "families": sorted(families)})
    except Exception as exc:  # noqa: BLE001
        failures.append({"code": "retrace_endpoint_unready", "detail": str(exc)})

    for label, path in {"output_dir": args.output_dir, "temp_dir": args.temp_dir}.items():
        try:
            checks[label] = {"path": str(path), "writable": _writable(path)}
        except Exception as exc:  # noqa: BLE001
            failures.append({"code": f"{label}_not_writable", "detail": str(exc)})
    checks["soundfile_available"] = True
    return {"ready": not failures, "checks": checks, "failures": failures}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vllm-pid", type=int, required=True)
    parser.add_argument("--wav-dir", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8010")
    parser.add_argument("--retrace-base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--temp-dir", type=Path, default=Path("/tmp/retrace-moss"))
    args = parser.parse_args()
    result = run_preflight(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["ready"] else 1)


if __name__ == "__main__":
    main()
