import numpy as np


def assess_audio_quality(audio: np.ndarray, sample_rate: int) -> dict:
    audio = np.asarray(audio, dtype=np.float64).reshape(-1)

    if audio.size == 0 or sample_rate <= 0:
        return {
            "status": "insufficient_data",
            "reasons": ["No usable audio data."],
            "duration_seconds": 0.0,
            "rms_level": 0.0,
            "clipping_fraction": 0.0,
            "near_silent_frame_fraction": 1.0,
        }

    duration = len(audio) / sample_rate
    rms = float(np.sqrt(np.mean(audio ** 2)))
    peak = float(np.max(np.abs(audio)))
    clipping_fraction = float(np.mean(np.abs(audio) >= 0.999))

    # Use 20 ms frames to inspect quiet portions of the recording.
    # This is a quality diagnostic, not a synthetic-speech feature.
    frame_length = max(1, int(sample_rate * 0.020))
    padding = (-len(audio)) % frame_length

    if padding:
        framed_audio = np.pad(audio, (0, padding))
    else:
        framed_audio = audio

    frames = framed_audio.reshape(-1, frame_length)
    frame_rms = np.sqrt(np.mean(frames ** 2, axis=1))
    near_silent_frame_fraction = float(np.mean(frame_rms < 0.002))

    reasons = []

    if duration < 2.0:
        reasons.append(
            "Recording is shorter than 2 seconds; evidence may be limited."
        )

    if rms < 0.005:
        reasons.append(
            "Low overall signal level. Check that speech is clearly audible."
        )

    if clipping_fraction > 0.01:
        reasons.append(
            "More than 1% of samples are near full scale; clipping may be present."
        )

    if near_silent_frame_fraction > 0.80:
        reasons.append(
            "Most 20 ms frames have very low energy; check for silence or an "
            "incorrect recording."
        )

    return {
        "status": "limited" if reasons else "experimental",
        "reasons": reasons,
        "duration_seconds": round(duration, 3),
        "rms_level": round(rms, 6),
        "peak_amplitude": round(peak, 6),
        "clipping_fraction": round(clipping_fraction, 6),
        "near_silent_frame_fraction": round(
            near_silent_frame_fraction, 6
        ),
    }
