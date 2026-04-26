"""
Common prompt utilities shared across all prompt modules.
Extracted from HunyuanService._extract_json().
"""
import re
import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def extract_json(content: str) -> dict:
    """
    Robustly extract JSON from LLM response content.
    Handles markdown code blocks, raw JSON, and mixed content.

    Extracted from HunyuanService._extract_json().
    """
    if not content:
        return {}

    # Pre-process: strip stray % symbols from numbers (e.g. "accuracy_rate": 95.5% -> 95.5)
    # This handles LLM outputs that include % in numeric values
    content = re.sub(r'(\d+\.?\d*)\s*%', r'\1', content)

    # Strategy 1: Extract from markdown code block
    json_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?\s*```', content, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Strategy 2: Extract JSON object via regex
    json_match = re.search(r'\{[\s\S]*\}', content)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass

    # Strategy 3: Extract JSON array via regex
    json_match = re.search(r'\[[\s\S]*\]', content)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass

    # Strategy 4: Parse entire content as JSON
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    logger.error(f"Failed to extract JSON from content: {content[:200]}")
    return {}


def build_word_info_table(word_info_list: list[dict]) -> str:
    """Build a markdown table from word info list."""
    if not word_info_list:
        return ""

    table = "| 字词 | 开始时间(ms) | 结束时间(ms) | 时长(ms) |\n"
    table += "|------|-------------|-------------|----------|\n"
    for w in word_info_list[:50]:  # Limit to 50 words
        table += f"| {w.get('word', '')} | {w.get('begin_time', 0)} | {w.get('end_time', 0)} | {w.get('duration', 0)} |\n"
    return table


def build_low_score_words_table(low_score_words: list[dict]) -> str:
    """Build a markdown table from low score words."""
    if not low_score_words:
        return ""

    table = "| 字词 | 准确度 | 流利度 |\n"
    table += "|------|--------|--------|\n"
    for word in low_score_words[:20]:
        table += f"| {word.get('word', '')} | {word.get('accuracy', 0)} | {word.get('fluency', 0)} |\n"
    return table
