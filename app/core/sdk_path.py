"""
Shared SDK path setup for Tencent Cloud Speech SDK.
Import this module once to add the SDK to sys.path.
"""
import sys
import os

SDK_PATH = os.path.join(
    os.path.dirname(__file__), "util", "tencentcloud-speech-sdk-python"
)

if SDK_PATH not in sys.path:
    sys.path.insert(0, SDK_PATH)
