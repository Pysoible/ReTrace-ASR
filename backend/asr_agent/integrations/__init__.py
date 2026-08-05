"""Bridges to ASR_domainterms (DeepSeek) and Qwen-Omni two-pass ASR."""

from asr_agent.integrations.deepseek import deepseek_status, revise_with_deepseek, correct_text_with_deepseek
from asr_agent.integrations.qwen_asr import asr_status, preload_engine, transcribe_audio

__all__ = [
    "asr_status",
    "correct_text_with_deepseek",
    "deepseek_status",
    "preload_engine",
    "revise_with_deepseek",
    "transcribe_audio",
]
