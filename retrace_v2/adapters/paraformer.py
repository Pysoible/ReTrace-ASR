"""Lazy FunASR/Paraformer adapter for bounded local audio windows."""
from __future__ import annotations

import hashlib
from pathlib import Path
import tempfile
from typing import Any

import soundfile as sf

from ..schemas import EvidenceHypothesis
from ..tools import ToolFailure


class ParaformerAdapter:
    def __init__(self, model_id: str, *, runtime: Any | None = None) -> None:
        self.model_id = model_id
        if runtime is None:
            try:
                from funasr import AutoModel
            except ImportError as error:
                raise ToolFailure(
                    "FunASR is unavailable; install the v2-acoustic extra"
                ) from error
            runtime = AutoModel(model=model_id, disable_update=True)
        self.runtime = runtime

    def transcribe(
        self,
        audio_path: Path,
        start_sec: float,
        end_sec: float,
    ) -> tuple[EvidenceHypothesis, ...]:
        info = sf.info(audio_path)
        duration = info.frames / info.samplerate
        if start_sec < 0 or end_sec <= start_sec or end_sec > duration + 1e-6:
            raise ToolFailure(
                f"audio window [{start_sec}, {end_sec}] is outside [0, {duration}]"
            )
        start_frame = round(start_sec * info.samplerate)
        end_frame = min(info.frames, round(end_sec * info.samplerate))
        audio, sample_rate = sf.read(
            audio_path,
            start=start_frame,
            stop=end_frame,
            dtype="float32",
            always_2d=False,
        )
        temporary = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        temporary_path = Path(temporary.name)
        temporary.close()
        try:
            sf.write(temporary_path, audio, sample_rate)
            result = self.runtime.generate(input=str(temporary_path), batch_size_s=60)
            return self._normalize(result, start_sec=start_sec, end_sec=end_sec)
        except ToolFailure:
            raise
        except Exception as error:
            raise ToolFailure(f"Paraformer inference failed: {error}") from error
        finally:
            temporary_path.unlink(missing_ok=True)

    def _normalize(
        self,
        result: object,
        *,
        start_sec: float,
        end_sec: float,
    ) -> tuple[EvidenceHypothesis, ...]:
        rows = result if isinstance(result, list) else [result]
        hypotheses: list[EvidenceHypothesis] = []
        for index, value in enumerate(rows):
            if not isinstance(value, dict):
                continue
            text = str(value.get("text") or "").strip()
            if not text:
                continue
            timestamps = value.get("timestamp")
            hypothesis_end = end_sec
            if isinstance(timestamps, list) and timestamps:
                last = timestamps[-1]
                if isinstance(last, (list, tuple)) and len(last) >= 2:
                    hypothesis_end = min(end_sec, start_sec + float(last[1]) / 1000.0)
            score = float(value.get("score") or 0.0)
            digest = hashlib.sha256(
                f"{self.model_id}|{start_sec}|{end_sec}|{index}|{text}".encode("utf-8")
            ).hexdigest()[:16]
            hypotheses.append(
                EvidenceHypothesis(
                    hypothesis_id=f"paraformer:{digest}",
                    model_id=self.model_id,
                    view="original",
                    text=text,
                    start_sec=start_sec,
                    end_sec=max(start_sec, hypothesis_end),
                    acoustic_score=score,
                )
            )
        if not hypotheses:
            raise ToolFailure("Paraformer returned no usable hypothesis")
        return tuple(hypotheses)
