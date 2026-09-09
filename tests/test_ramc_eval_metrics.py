import importlib.util
from pathlib import Path


_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_ramc_dataset_eval.py"
_SPEC = importlib.util.spec_from_file_location("run_ramc_dataset_eval", _SCRIPT)
assert _SPEC and _SPEC.loader
ramc_eval = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(ramc_eval)


def test_revision_metrics_report_historical_and_current_routes_separately():
    session = {
        "turns": [
            {"turn_id": "t1", "raw_text": "泰信方案", "current_text": "泰康方案", "meta": {}},
            {"turn_id": "t2", "raw_text": "当前错词", "current_text": "当前错词", "meta": {"uncertainty": {"acoustic_disagreement": [{"span_a": "错词"}]}}},
            {"turn_id": "t3", "raw_text": "后文提到泰康", "current_text": "后文提到泰康", "meta": {}},
        ],
        "revision_events": [
            {
                "event_id": "history",
                "active": True,
                "action": "REVISE_HISTORY",
                "target_turn_id": "t1",
                "source_turn_id": "t3",
                "span": "泰信",
                "replacement": "泰康",
                "before_text": "泰信方案",
                "after_text": "泰康方案",
                "evidence": ["audio:/tmp/a.wav:0-2"],
                "resolver": "audio-llm-confirm",
            },
            {
                "event_id": "current",
                "active": True,
                "action": "REVISE_CURRENT",
                "target_turn_id": "t2",
                "source_turn_id": "t2",
                "span": "错词",
                "replacement": "错字",
                "before_text": "当前错词",
                "after_text": "当前错字",
                "evidence": ["acoustic_disagreement:错词"],
                "resolver": "audio-uncertainty-relisten",
            },
        ],
    }

    report = ramc_eval.revision_metrics(
        session,
        reference="泰康方案当前正确后文提到泰康",
        duration=10.0,
        turn_references={"t1": "泰康方案", "t2": "当前正确", "t3": "后文提到泰康"},
    )

    detail = report["_detail"]
    assert detail["historical_revision_count"] == 1
    assert detail["current_revision_count"] == 1
    assert detail["historical_revision_precision"] == 1.0
    assert detail["current_revision_precision"] == 0.0
    assert detail["historical_resolution_latency_turns"] == 2.0
    assert detail["acoustic_sentinel_signal_turns"] == 1


def test_aggregate_exposes_route_metrics_from_per_sample_details():
    summary = ramc_eval.aggregate([
        {
            "status": "ok",
            "raw_asr": {"cer": 0.4},
            "final": {"cer": 0.3},
            "cer_absolute_gain": 0.1,
            "cer_relative_gain": 0.25,
            "revision_events": 2,
            "audio_verified_events": 1,
            "relisten_seconds_proxy": 3.0,
            "rollback_count": 0,
            "acoustic_sentinel_observed_turns": 4,
            "acoustic_sentinel_signal_turns": 1,
            "rtf": 0.8,
            "requested_metrics": {
                "HRA": 1.0,
                "_detail": {
                    "historical_revision_precision": 1.0,
                    "current_revision_precision": 0.0,
                    "historical_overcorrection_rate": 0.0,
                    "current_overcorrection_rate": 1.0,
                    "historical_resolution_latency_turns": 2.0,
                    "rollback_success_rate": None,
                },
            },
        }
    ])

    assert summary["route_metrics"]["historical_revision_precision"] == 1.0
    assert summary["route_metrics"]["current_overcorrection_rate"] == 1.0


def test_metric_transcript_join_removes_adjacent_chunk_overlap():
    session = {
        "turns": [
            {"turn_id": "t1", "raw_text": "abcdef", "current_text": "abcdef", "meta": {"start_sec": 0.0, "end_sec": 1.0}},
            {"turn_id": "t2", "raw_text": "defghi", "current_text": "defghi", "meta": {"start_sec": 0.8, "end_sec": 1.8}},
        ],
        "revision_events": [],
    }

    assert ramc_eval.join_turns_for_metrics(session, "raw_text") == "abcdefghi"


def test_reference_segments_are_assigned_once_across_overlapping_turns():
    turns = [
        {"turn_id": "t1", "meta": {"start_sec": 0.0, "end_sec": 1.0}},
        {"turn_id": "t2", "meta": {"start_sec": 0.8, "end_sec": 1.8}},
    ]
    segments = [
        {"start": 0.85, "end": 1.2, "text": "重复", "clean": "重复"},
    ]

    references, diagnostics = ramc_eval.reference_by_assigned_turn(turns, segments)

    assert list(references.values()).count("重复") == 1
    assert diagnostics["ambiguous_turn_count"] == 2


def test_evaluate_one_reports_diagnostic_cer_not_fake_wer(monkeypatch, tmp_path):
    wav = tmp_path / "sample.wav"
    txt = tmp_path / "sample.txt"
    with __import__("wave").open(str(wav), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16000)
        stream.writeframes(b"\x00\x00" * 16000)
    txt.write_text("[0.0,1.0] spk utt abcdefghi\n", encoding="utf-8")

    monkeypatch.setattr(
        ramc_eval,
        "post_json",
        lambda *_args, **_kwargs: {
            "final_text": "abcdefdefghi",
            "session": {
                "turns": [
                    {"turn_id": "t1", "raw_text": "abcdef", "current_text": "abcdef", "meta": {"start_sec": 0.0, "end_sec": 1.0}},
                    {"turn_id": "t2", "raw_text": "defghi", "current_text": "defghi", "meta": {"start_sec": 0.8, "end_sec": 1.8}},
                ],
                "revision_events": [],
            },
        },
    )
    monkeypatch.setattr(ramc_eval, "embedding_similarity", lambda *_args: None)
    monkeypatch.setattr(ramc_eval, "official_menli", lambda *_args: None)

    row = ramc_eval.evaluate_one("http://example", wav, txt, tmp_path, 30.0, "s")

    assert row["raw_asr"]["cer"] == 0.0
    assert row["final"]["cer"] == 0.0
    assert row["final_wer"] is None
    assert row["metric_kind"] == "diagnostic_cer"
    assert row["experiment_mode"] == "retrace"
    assert row["primary_metrics"]["lecr"] is None
    assert row["primary_metrics"]["entity_consistency_error_rate"] is None
    assert row["eligible_turn_count"] == 1
    assert row["ambiguous_turn_count"] == 2
    assert row["unscorable_turn_count"] == 1
