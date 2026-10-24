from fastapi import APIRouter, File, HTTPException, UploadFile
import numpy as np

from app.config import settings
from app.schemas import DetectionResponse
from app.services.audio_processor import AudioProcessingError, load_audio
from app.services.reliability import assess_audio_quality
from app.services.spectra_detector import (
    BONAFIDE_THRESHOLD,
    SpectraDetector,
)


router = APIRouter()
spectra_detector = SpectraDetector()


@router.get("/spectra-health")
def spectra_health():
    return {
        "status": "ok" if spectra_detector.model_file_exists else "unavailable",
        "model_name": spectra_detector.model_name,
        "model_file_exists": spectra_detector.model_file_exists,
        "model_loaded": spectra_detector.ready,
        "threshold_bona_fide_logit": BONAFIDE_THRESHOLD,
    }


@router.post("/analyze-spectra", response_model=DetectionResponse)
async def analyze_audio_spectra(file: UploadFile = File(...)):
    content = await file.read(
        settings.max_audio_size_mb * 1024 * 1024 + 1
    )

    if len(content) > settings.max_audio_size_mb * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail="Audio file is too large.",
        )

    try:
        audio, sample_rate, duration = load_audio(
            content,
            settings.max_audio_duration_seconds,
        )
    except AudioProcessingError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    try:
        result = spectra_detector.predict(audio, sample_rate)
    except (FileNotFoundError, RuntimeError) as exc:
        print("VOICEMATRIX Spectra-0 loading/inference error:", repr(exc))
        raise HTTPException(
            status_code=503,
            detail=(
                "Spectra-0 is unavailable. The existing AASIST-L "
                "endpoint remains separate and unchanged."
            ),
        ) from exc
    except Exception as exc:
        print("VOICEMATRIX Spectra-0 inference error:", repr(exc))
        raise HTTPException(
            status_code=500,
            detail="Spectra-0 audio analysis failed.",
        ) from exc

    quality = assess_audio_quality(audio, sample_rate)

    # Use Spectra-0's raw genuine-class logit, not AASIST-L thresholds.
    if result.bona_fide_logit > BONAFIDE_THRESHOLD:
        verdict = "likely_genuine"
    else:
        verdict = "potentially_synthetic"

    warnings = [
        "Spectra-0 is experimental in this application.",
        "The displayed score is not a calibrated probability.",
        "A model prediction alone does not establish identity or fraud.",
    ]

    if quality["reasons"]:
        warnings.append(
            "Audio-quality checks: " + "; ".join(quality["reasons"])
        )

    if result.used_repeated_padding:
        warnings.append(
            "The short recording was repeated to fill the model window. "
            "Interpret this result cautiously."
        )

    return DetectionResponse(
        verdict=verdict,
        score=result.score,
        reliability=quality["status"],
        explanation=result.explanation,
        duration_seconds=duration,
        model_name=result.model_name,
        inference_ms=result.inference_ms,
        window_count=1,
        window_score_range=None,
        bona_fide_logit=result.bona_fide_logit,
        window_bona_fide_logits=[result.bona_fide_logit],
        quality_reasons=quality["reasons"],
        audio_rms=quality["rms_level"],
        peak_amplitude=quality["peak_amplitude"],
        clipping_fraction=quality["clipping_fraction"],
        near_silent_frame_fraction=quality["near_silent_frame_fraction"],
        warning=" ".join(warnings),
    )


# Experimental robustness scan for Spectra-0.
# This endpoint intentionally uses Spectra-0's raw genuine-class
# logit and its own decision threshold, not the AASIST-L thresholds.

def _spectra_verdict(genuine_logit: float) -> str:
    if genuine_logit >= BONAFIDE_THRESHOLD:
        return "likely_genuine"
    return "potentially_synthetic"


@router.post("/spectra-robustness")
async def spectra_robustness_scan(file: UploadFile = File(...)):
    content = await file.read(
        settings.max_audio_size_mb * 1024 * 1024 + 1
    )

    if len(content) > settings.max_audio_size_mb * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail="Audio file is too large.",
        )

    try:
        audio, sample_rate, duration = load_audio(
            content,
            settings.max_audio_duration_seconds,
        )
    except AudioProcessingError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    if not 2.0 <= duration <= 10.0:
        raise HTTPException(
            status_code=400,
            detail=(
                "Robustness testing requires audio "
                "between 2 and 10 seconds."
            ),
        )

    try:
        # The first call loads Spectra-0 if it is not already loaded.
        baseline = spectra_detector.predict(audio, sample_rate)
        baseline_verdict = _spectra_verdict(
            baseline.bona_fide_logit
        )

        # Reproducible white-noise transformation at approximately 20 dB SNR.
        rng = np.random.default_rng(20261024)
        signal_rms = float(
            np.sqrt(np.mean(audio.astype(np.float64) ** 2))
        )

        if signal_rms > 1e-8:
            noise_rms = signal_rms / (10.0 ** (20.0 / 20.0))
            noise = rng.normal(
                0.0,
                noise_rms,
                size=audio.shape,
            ).astype(np.float32)
            noisy_audio = np.clip(
                audio + noise,
                -1.0,
                1.0,
            ).astype(np.float32)
        else:
            noisy_audio = audio.copy()

        # Test transformations independently from the original audio.
        gain_minus_6db = (
            audio * (10.0 ** (-6.0 / 20.0))
        ).astype(np.float32)

        leading_silence = np.concatenate((
            np.zeros(int(0.250 * sample_rate), dtype=np.float32),
            audio,
        ))

        transformations = [
            ("volume_minus_6db", gain_minus_6db),
            ("noise_20db_snr", noisy_audio),
            ("leading_silence_250ms", leading_silence),
        ]

        results = [{
            "condition": "original",
            "score": float(baseline.score),
            "score_change": None,
            "verdict": baseline_verdict,
            "classification_changed": False,
            "genuine_logit": float(baseline.bona_fide_logit),
        }]

        score_changes = []
        changed_predictions = 0

        for condition, modified_audio in transformations:
            result = spectra_detector.predict(
                modified_audio,
                sample_rate,
            )

            verdict = _spectra_verdict(
                result.bona_fide_logit
            )

            score_change = float(result.score - baseline.score)
            classification_changed = verdict != baseline_verdict

            score_changes.append(abs(score_change))

            if classification_changed:
                changed_predictions += 1

            results.append({
                "condition": condition,
                "score": float(result.score),
                "score_change": score_change,
                "verdict": verdict,
                "classification_changed": classification_changed,
                "genuine_logit": float(result.bona_fide_logit),
            })

        maximum_change = (
            max(score_changes) if score_changes else 0.0
        )

        # This is a diagnostic heuristic, not a calibrated guarantee.
        is_sensitive = (
            changed_predictions > 0
            or maximum_change >= 0.15
        )

        return {
            "model_name": spectra_detector.model_name,
            "robustness_status": (
                "sensitive_to_tested_transformations"
                if is_sensitive
                else "stable_under_tested_transformations"
            ),
            "baseline_score": float(baseline.score),
            "baseline_verdict": baseline_verdict,
            "max_absolute_score_change": float(maximum_change),
            "changed_prediction_count": changed_predictions,
            "results": results,
            "warning": (
                "Experimental diagnostic using the original recording "
                "and three controlled transformations. White noise is "
                "synthetic; these tests do not establish general "
                "real-world accuracy or prove authenticity."
            ),
        }

    except (FileNotFoundError, RuntimeError) as exc:
        print("VOICEMATRIX Spectra robustness error:", repr(exc))
        raise HTTPException(
            status_code=503,
            detail="Spectra-0 is unavailable for robustness testing.",
        ) from exc
    except Exception as exc:
        print("VOICEMATRIX Spectra robustness error:", repr(exc))
        raise HTTPException(
            status_code=500,
            detail="Spectra-0 robustness analysis failed.",
        ) from exc

