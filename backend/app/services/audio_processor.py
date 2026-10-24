from fractions import Fraction
from io import BytesIO
import subprocess

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


TARGET_SAMPLE_RATE = 16000


class AudioProcessingError(ValueError):
    pass


def _decode_with_ffmpeg(
    content: bytes,
    max_duration_seconds: int,
):
    try:
        probe = subprocess.run(
            [
                "ffprobe",
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                "pipe:0",
            ],
            input=content,
            capture_output=True,
            timeout=10,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        raise AudioProcessingError(
            "Audio decoding failed. Check that FFmpeg is installed."
        ) from exc

    if probe.returncode != 0:
        raise AudioProcessingError(
            "Could not decode this audio. Try a WAV, FLAC, or MP3 file."
        )

    try:
        duration = float(probe.stdout.decode("utf-8").strip())
    except (ValueError, UnicodeDecodeError) as exc:
        raise AudioProcessingError(
            "Could not determine the audio duration."
        ) from exc

    if not np.isfinite(duration) or duration <= 0:
        raise AudioProcessingError("The audio duration is invalid.")

    if duration > max_duration_seconds:
        raise AudioProcessingError(
            f"Audio must be {max_duration_seconds} seconds or shorter."
        )

    try:
        decoded = subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel", "error",
                "-i", "pipe:0",
                "-vn",
                "-ac", "1",
                "-ar", str(TARGET_SAMPLE_RATE),
                "-f", "f32le",
                "pipe:1",
            ],
            input=content,
            capture_output=True,
            timeout=20,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        raise AudioProcessingError(
            "FFmpeg could not decode the uploaded audio."
        ) from exc

    if decoded.returncode != 0 or not decoded.stdout:
        raise AudioProcessingError(
            "FFmpeg could not decode the uploaded audio."
        )

    audio = np.frombuffer(decoded.stdout, dtype="<f4").copy()
    return audio, TARGET_SAMPLE_RATE, duration


def load_audio(
    content: bytes,
    max_duration_seconds: int = 10,
):
    if not content:
        raise AudioProcessingError("The uploaded audio file is empty.")

    try:
        audio, sample_rate = sf.read(
            BytesIO(content),
            dtype="float32",
            always_2d=False,
        )
        if audio.ndim == 2:
            audio = audio.mean(axis=1)

        if audio.ndim != 1 or audio.size == 0:
            raise AudioProcessingError("No usable audio was found.")

        if sample_rate <= 0:
            raise AudioProcessingError("The sample rate is invalid.")

        duration = len(audio) / sample_rate

        if duration > max_duration_seconds:
            raise AudioProcessingError(
                f"Audio must be {max_duration_seconds} seconds or shorter."
            )

        if sample_rate != TARGET_SAMPLE_RATE:
            ratio = Fraction(
                TARGET_SAMPLE_RATE,
                sample_rate,
            ).limit_denominator()

            audio = resample_poly(
                audio,
                ratio.numerator,
                ratio.denominator,
            ).astype(np.float32)

        sample_rate = TARGET_SAMPLE_RATE

    except AudioProcessingError:
        raise
    except Exception:
        audio, sample_rate, duration = _decode_with_ffmpeg(
            content,
            max_duration_seconds,
        )

    if audio.ndim != 1 or audio.size == 0:
        raise AudioProcessingError("No usable audio was found.")

    if not np.isfinite(audio).all():
        raise AudioProcessingError("The audio contains invalid values.")

    if np.max(np.abs(audio)) == 0:
        raise AudioProcessingError("The recording is silent.")

    if "duration" not in locals():
        duration = len(audio) / sample_rate

    if duration < 2.0:
        raise AudioProcessingError(
            "Audio must be at least 2 seconds long."
        )

    if duration > max_duration_seconds:
        raise AudioProcessingError(
            f"Audio must be {max_duration_seconds} seconds or shorter."
        )

    return audio.astype(np.float32), sample_rate, float(duration)
