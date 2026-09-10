from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from retrace_v2.adapters.paraformer import ParaformerAdapter
from retrace_v2.adapters.separation import CommandSeparator
from retrace_v2.tools import ToolFailure


class FakeParaformerRuntime:
    def __init__(self) -> None:
        self.generated_path: Path | None = None
        self.input_existed_during_call = False

    def generate(self, *, input: str, batch_size_s: int):
        self.generated_path = Path(input)
        self.input_existed_during_call = self.generated_path.is_file()
        return [{"text": "方案", "timestamp": [[0, 200], [200, 400]], "score": 0.92}]


def test_paraformer_preserves_identity_and_window_time(tmp_path: Path) -> None:
    audio = tmp_path / "audio.wav"
    sf.write(audio, np.zeros(32000, dtype=np.float32), 16000)
    runtime = FakeParaformerRuntime()

    rows = ParaformerAdapter("paraformer-zh", runtime=runtime).transcribe(
        audio, 1.0, 2.0
    )

    assert rows[0].model_id == "paraformer-zh"
    assert rows[0].text == "方案"
    assert rows[0].start_sec == pytest.approx(1.0)
    assert rows[0].end_sec == pytest.approx(1.4)
    assert rows[0].acoustic_score == pytest.approx(0.92)
    assert runtime.input_existed_during_call
    assert runtime.generated_path is not None
    assert not runtime.generated_path.exists()


def test_paraformer_rejects_out_of_range_window(tmp_path: Path) -> None:
    audio = tmp_path / "audio.wav"
    sf.write(audio, np.zeros(16000, dtype=np.float32), 16000)

    with pytest.raises(ToolFailure, match="window"):
        ParaformerAdapter("paraformer-zh", runtime=FakeParaformerRuntime()).transcribe(
            audio, 1.0, 2.0
        )


def test_separation_command_failure_is_explicit(tmp_path: Path) -> None:
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"not-used")
    adapter = CommandSeparator(
        ["false"],
        backend_id="gss",
        output_root=tmp_path / "separated",
    )

    with pytest.raises(ToolFailure, match="gss"):
        adapter.separate(audio, 0.0, 1.0)
