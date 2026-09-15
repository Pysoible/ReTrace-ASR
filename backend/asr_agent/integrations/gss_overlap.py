"""Optional guided source separation for MOSS overlap candidates.

The worker is deliberately isolated behind ``MOSS_GSS_ENABLED``.  It never
changes the immutable first-pass transcript: it only attaches a separated
decode to selected overlapping chunks so the normal ReTrace gates can audit a
bounded local substitution.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any

from asr_agent.integrations import moss_asr


def _truthy(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def status() -> dict[str, Any]:
    python = Path(os.getenv("MOSS_GSS_PYTHON", "/home/panyibo/envs/retrace-moss/bin/python"))
    package = Path(os.getenv("MOSS_GSS_PACKAGE", "/home/panyibo/tools/gss"))
    site_packages = Path(os.getenv("MOSS_GSS_SITE_PACKAGES", "/home/panyibo/envs/retrace-gss/py312-site-packages"))
    return {
        "enabled": _truthy("MOSS_GSS_ENABLED"),
        "ready": python.exists() and package.exists() and site_packages.exists(),
        "python": str(python),
        "package": str(package),
        "site_packages": str(site_packages),
        "cuda_visible_devices": os.getenv("MOSS_GSS_CUDA_VISIBLE_DEVICES", "2"),
        "max_components": int(os.getenv("MOSS_GSS_MAX_COMPONENTS", "10")),
    }


def _overlap_components(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_index = {int(chunk.get("index", position)): position for position, chunk in enumerate(chunks)}
    graph: dict[int, set[int]] = {}
    for position, chunk in enumerate(chunks):
        if not chunk.get("overlap"):
            continue
        graph.setdefault(position, set())
        for peer_index in chunk.get("overlap_with_indices") or []:
            peer = by_index.get(int(peer_index))
            if peer is None:
                continue
            graph[position].add(peer)
            graph.setdefault(peer, set()).add(position)
    output: list[dict[str, Any]] = []
    unseen = set(graph)
    while unseen:
        seed = min(unseen)
        stack, component = [seed], set()
        while stack:
            current = stack.pop()
            if current in component:
                continue
            component.add(current)
            unseen.discard(current)
            stack.extend(graph.get(current, set()) - component)
        score = sum(float(chunks[item].get("overlap_duration_sec") or 0.0) for item in component) / 2.0
        output.append({"positions": sorted(component), "overlap_duration_sec": round(score, 3)})
    return sorted(output, key=lambda item: (-item["overlap_duration_sec"], item["positions"]))


def _selected_positions(chunks: list[dict[str, Any]], max_components: int) -> tuple[list[int], list[dict[str, Any]]]:
    components = _overlap_components(chunks)[:max(0, max_components)]
    positions = sorted({position for item in components for position in item["positions"]})
    return positions, components


def _audio_info(audio_path: Path) -> dict[str, Any]:
    import soundfile as sf

    with sf.SoundFile(audio_path) as handle:
        return {
            "sampling_rate": int(handle.samplerate),
            "num_samples": int(len(handle)),
            "duration": float(len(handle) / handle.samplerate),
            "channels": list(range(int(handle.channels))),
        }


def _write_jsonl_gz(path: Path, rows: list[dict[str, Any]]) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def _manifests(
    audio_path: Path,
    chunks: list[dict[str, Any]],
    positions: list[int],
    workdir: Path,
) -> tuple[Path, Path, list[dict[str, Any]]]:
    info = _audio_info(audio_path)
    if len(info["channels"]) < 2:
        raise ValueError("GSS requires multichannel audio")
    recording_id = audio_path.stem
    recording = {
        "id": recording_id,
        "sources": [{"type": "file", "channels": info["channels"], "source": str(audio_path)}],
        "sampling_rate": info["sampling_rate"],
        "num_samples": info["num_samples"],
        "duration": info["duration"],
        "channel_ids": info["channels"],
    }
    targets: list[dict[str, Any]] = []
    supervisions: list[dict[str, Any]] = []
    for position, chunk in enumerate(chunks):
        start, end = chunk.get("start_sec"), chunk.get("end_sec")
        speaker = str(chunk.get("speaker") or "unknown")
        if start is None or end is None or float(end) <= float(start):
            continue
        supervision_id = f"moss-{position:04d}"
        supervision = {
            "id": supervision_id,
            "recording_id": recording_id,
            "start": float(start),
            "duration": float(end) - float(start),
            "channel": info["channels"],
            "text": str(chunk.get("text") or ""),
            "speaker": speaker,
        }
        supervisions.append(supervision)
        if position not in positions:
            continue
        local_supervision = dict(supervision)
        local_supervision["start"] = 0.0
        targets.append({
            "id": supervision_id,
            "start": float(start),
            "duration": float(end) - float(start),
            "channel": info["channels"],
            "supervisions": [local_supervision],
            "recording": recording,
            "type": "MultiCut",
        })
    recording_cut = {
        "id": recording_id,
        "start": 0.0,
        "duration": info["duration"],
        "channel": info["channels"],
        "supervisions": supervisions,
        "recording": recording,
        "type": "MultiCut",
    }
    recording_path, target_path = workdir / "recording.jsonl.gz", workdir / "targets.jsonl.gz"
    _write_jsonl_gz(recording_path, [recording_cut])
    _write_jsonl_gz(target_path, targets)
    return recording_path, target_path, targets


def _worker_env() -> dict[str, str]:
    env = dict(os.environ)
    package = os.getenv("MOSS_GSS_PACKAGE", "/home/panyibo/tools/gss")
    site_packages = os.getenv("MOSS_GSS_SITE_PACKAGES", "/home/panyibo/envs/retrace-gss/py312-site-packages")
    env["PYTHONPATH"] = os.pathsep.join(item for item in (package, site_packages, env.get("PYTHONPATH", "")) if item)
    runtime = Path(os.getenv("MOSS_GSS_CUDA_RUNTIME", "/home/panyibo/envs/retrace-gss/cuda12-runtime"))
    libraries = sorted(str(path) for path in runtime.rglob("lib") if path.is_dir()) if runtime.exists() else []
    env["LD_LIBRARY_PATH"] = os.pathsep.join([*libraries, env.get("LD_LIBRARY_PATH", "")]).rstrip(os.pathsep)
    env["CUDA_PATH"] = os.getenv("MOSS_GSS_CUDA_PATH", str(runtime / "nvidia/cuda_runtime"))
    env["CUDA_VISIBLE_DEVICES"] = os.getenv("MOSS_GSS_CUDA_VISIBLE_DEVICES", "2")
    return env


def _enhance(recording: Path, targets: Path, output_dir: Path, manifest: Path) -> tuple[float, str]:
    python = os.getenv("MOSS_GSS_PYTHON", "/home/panyibo/envs/retrace-moss/bin/python")
    command = [
        python, "-c", "from gss.bin.gss import cli; cli()", "enhance", "cuts",
        str(recording), str(targets), str(output_dir),
        "--bss-iterations", os.getenv("MOSS_GSS_BSS_ITERATIONS", "5"),
        "--no-wpe", "--context-duration", os.getenv("MOSS_GSS_CONTEXT_DURATION", "8"),
        "--use-garbage-class", "--min-segment-length", "0.25", "--max-segment-length", "15",
        "--max-batch-duration", "5", "--max-batch-cuts", "1", "--num-workers", "1",
        "--num-buckets", "1", "--enhanced-manifest", str(manifest), "--force-overwrite",
    ]
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        env=_worker_env(),
        capture_output=True,
        text=True,
        timeout=float(os.getenv("MOSS_GSS_TIMEOUT", "900")),
        check=False,
    )
    elapsed = time.perf_counter() - started
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "unknown GSS error")[-4000:]
        raise RuntimeError(f"GSS exited {completed.returncode}: {detail}")
    return elapsed, (completed.stderr or completed.stdout or "")[-4000:]


_OUTPUT_KEY_RE = re.compile(r"-(?P<speaker>[^-]+)-(?P<start>\d+)_(?P<end>\d+)(?:-\d+)?$")


def _target_key(speaker: str, start: float, end: float) -> tuple[str, int, int]:
    return speaker, round(start * 100), round(end * 100)


def _decode_outputs(manifest: Path, targets: list[dict[str, Any]]) -> tuple[dict[int, str], float]:
    lookup: dict[tuple[str, int, int], int] = {}
    for target in targets:
        supervision = target["supervisions"][0]
        original_position = int(str(target["id"]).split("-")[-1])
        lookup[_target_key(supervision["speaker"], float(target["start"]), float(target["start"] + target["duration"]))] = original_position
    decoded: dict[int, str] = {}
    started = time.perf_counter()
    with gzip.open(manifest, "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            source = Path(row["recording"]["sources"][0]["source"])
            match = _OUTPUT_KEY_RE.search(row["recording"]["id"])
            if match is None:
                continue
            key = (match.group("speaker"), int(match.group("start")), int(match.group("end")))
            position = lookup.get(key)
            if position is None:
                continue
            payload = moss_asr._post_transcription(source)
            raw = str(payload.get("text") or payload.get("transcript") or "")
            text = str(moss_asr.parse_moss_transcript(raw).get("text") or "").strip()
            if text:
                decoded[position] = text
    return decoded, time.perf_counter() - started


def _cache_key(audio_path: Path, chunks: list[dict[str, Any]], positions: list[int]) -> str:
    stat = audio_path.stat()
    contract = {
        "path": str(audio_path.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
        "positions": positions,
        "chunks": [
            [chunk.get("speaker"), chunk.get("start_sec"), chunk.get("end_sec")]
            for chunk in chunks
        ],
        "bss_iterations": os.getenv("MOSS_GSS_BSS_ITERATIONS", "5"),
        "context_duration": os.getenv("MOSS_GSS_CONTEXT_DURATION", "8"),
    }
    return hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()


def enrich(audio_path: str | Path, asr: dict[str, Any]) -> dict[str, Any]:
    """Return copied ASR evidence with cached GSS decodes attached to chunks."""
    output = dict(asr)
    chunks = [dict(item) for item in (asr.get("chunks") or [])]
    output["chunks"] = chunks
    if not _truthy("MOSS_GSS_ENABLED"):
        output["gss_overlap_audit"] = {"enabled": False, "status": "disabled"}
        return output
    ready = status()
    if not ready["ready"]:
        output["gss_overlap_audit"] = {**ready, "status": "unavailable"}
        return output
    positions, components = _selected_positions(chunks, ready["max_components"])
    if not positions:
        output["gss_overlap_audit"] = {**ready, "status": "no_overlap", "selected_components": 0}
        return output
    source = Path(audio_path).expanduser().resolve()
    cache_root = Path(os.getenv("MOSS_GSS_CACHE_DIR", "retrace_state/gss_cache")).expanduser()
    cache_dir = cache_root / _cache_key(source, chunks, positions)
    cache_file = cache_dir / "result.json"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_hit = cache_file.exists()
    if cache_hit:
        cached = json.loads(cache_file.read_text(encoding="utf-8"))
        decoded = {int(key): str(value) for key, value in (cached.get("decoded") or {}).items()}
        gss_elapsed = decode_elapsed = 0.0
    else:
        try:
            recording, targets_path, targets = _manifests(source, chunks, positions, cache_dir)
            manifest = cache_dir / "enhanced.jsonl.gz"
            gss_elapsed, log_tail = _enhance(recording, targets_path, cache_dir / "enhanced", manifest)
            decoded, decode_elapsed = _decode_outputs(manifest, targets)
            cache_file.write_text(json.dumps({
                "decoded": decoded,
                "gss_elapsed_sec": gss_elapsed,
                "decode_elapsed_sec": decode_elapsed,
                "log_tail": log_tail,
            }, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as exc:  # noqa: BLE001
            output["gss_overlap_audit"] = {
                **ready, "status": "failed", "error": f"{type(exc).__name__}: {exc}",
                "selected_components": len(components), "selected_chunks": len(positions),
            }
            return output
    for position, text in decoded.items():
        if 0 <= position < len(chunks):
            chunks[position]["gss_text"] = text
    output["gss_overlap_audit"] = {
        **ready,
        "status": "ok",
        "cache_hit": cache_hit,
        "selected_components": len(components),
        "selected_chunks": len(positions),
        "decoded_chunks": len(decoded),
        "components": components,
        "gss_elapsed_sec": round(gss_elapsed, 3),
        "decode_elapsed_sec": round(decode_elapsed, 3),
    }
    return output
