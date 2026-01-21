"""
Audio processing utilities using ffmpeg.
"""
import asyncio
import tempfile
import os
import shutil
from typing import Optional


def find_executable(name: str) -> str:
    """
    Find executable path, checking common locations.

    Args:
        name: Executable name (e.g., 'ffmpeg', 'ffprobe')

    Returns:
        Full path to executable, or just the name if not found in common paths
    """
    # Try using shutil.which first (checks PATH)
    path = shutil.which(name)
    if path:
        return path

    # Common installation paths on Linux
    common_paths = [
        f"/usr/bin/{name}",
        f"/usr/local/bin/{name}",
        f"/opt/homebrew/bin/{name}",  # macOS Homebrew
        f"/snap/bin/{name}",
    ]

    for p in common_paths:
        if os.path.isfile(p) and os.access(p, os.X_OK):
            return p

    # Return just the name, let the system try to find it
    return name


# Find ffmpeg and ffprobe paths
FFMPEG_PATH = find_executable("ffmpeg")
FFPROBE_PATH = find_executable("ffprobe")


async def convert_audio_to_wav(
    audio_data: bytes,
    sample_rate: int = 16000,
    channels: int = 1,
    bit_depth: int = 16
) -> bytes:
    """
    Convert audio to WAV format using ffmpeg.

    Args:
        audio_data: Input audio bytes (any format supported by ffmpeg)
        sample_rate: Output sample rate in Hz (default: 16000)
        channels: Number of output channels (default: 1 for mono)
        bit_depth: Output bit depth (default: 16)

    Returns:
        Converted WAV audio bytes

    Raises:
        RuntimeError: If ffmpeg conversion fails
    """
    # Create temporary files for input and output
    with tempfile.NamedTemporaryFile(delete=False, suffix=".input") as input_file:
        input_path = input_file.name
        input_file.write(audio_data)

    output_path = input_path + ".wav"

    try:
        # Build ffmpeg command
        # -y: overwrite output file
        # -i: input file
        # -ar: sample rate
        # -ac: number of channels
        # -sample_fmt: sample format (s16 = signed 16-bit)
        # -f wav: output format
        cmd = [
            FFMPEG_PATH,
            "-y",
            "-i", input_path,
            "-ar", str(sample_rate),
            "-ac", str(channels),
            "-sample_fmt", f"s{bit_depth}",
            "-f", "wav",
            output_path
        ]

        # Run ffmpeg asynchronously
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

        stdout, stderr = await process.communicate()

        if process.returncode != 0:
            error_msg = stderr.decode("utf-8", errors="ignore")
            raise RuntimeError(f"ffmpeg conversion failed: {error_msg}")

        # Read converted audio
        with open(output_path, "rb") as f:
            converted_audio = f.read()

        return converted_audio

    finally:
        # Clean up temporary files
        if os.path.exists(input_path):
            os.unlink(input_path)
        if os.path.exists(output_path):
            os.unlink(output_path)


async def get_audio_duration(audio_data: bytes) -> Optional[float]:
    """
    Get audio duration in seconds using ffprobe.

    Args:
        audio_data: Audio bytes

    Returns:
        Duration in seconds, or None if failed
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".audio") as f:
        temp_path = f.name
        f.write(audio_data)

    try:
        cmd = [
            FFPROBE_PATH,
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            temp_path
        ]

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

        stdout, stderr = await process.communicate()

        if process.returncode == 0:
            try:
                return float(stdout.decode().strip())
            except ValueError:
                return None
        return None

    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)
