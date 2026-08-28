from asr_agent.degeneration import assess_transcript, should_replace_degenerate
from asr_agent.degeneration import assess_repeated_tail


def test_detects_generic_repetition_without_hard_coding_filler_words():
    repeated_filler = assess_transcript("啊啊啊啊啊啊啊啊", duration_sec=6.0)
    repeated_phrase = assess_transcript("项目延期项目延期项目延期项目延期", duration_sec=8.0)

    assert repeated_filler.degenerate is True
    assert repeated_phrase.degenerate is True
    assert "repetition" in repeated_filler.reasons
    assert "periodic_loop" in repeated_phrase.reasons


def test_keeps_normal_sentences_that_contain_fillers():
    assessment = assess_transcript("啊，我觉得这个方案可以继续推进", duration_sec=4.0)

    assert assessment.degenerate is False


def test_ignores_display_timestamp_when_scoring_repetition():
    assessment = assess_transcript("[0.0-7.0] 啊啊啊啊啊啊啊啊", duration_sec=7.0)

    assert assessment.degenerate is True
    assert "repetition" in assessment.reasons
    assert "format_leak" not in assessment.reasons


def test_replacement_requires_a_clear_quality_improvement():
    original = assess_transcript("嗯嗯嗯嗯嗯嗯嗯嗯", duration_sec=7.0)
    improved = assess_transcript("我们下周一开始进行项目验收", duration_sec=7.0)
    another_loop = assess_transcript("然后然后然后然后然后", duration_sec=7.0)

    assert should_replace_degenerate(original, improved) is True
    assert should_replace_degenerate(original, another_loop) is False

def test_assess_repeated_tail_finds_loop_after_valid_prefix():
    assessment = assess_repeated_tail("这是一段正常内容，然后嗯嗯嗯嗯嗯嗯嗯嗯嗯嗯嗯嗯")

    assert assessment.degenerate is True
    assert "repeated_tail" in assessment.reasons
