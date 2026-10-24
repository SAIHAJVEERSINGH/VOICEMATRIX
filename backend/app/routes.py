from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

from app.config import settings
from app.schemas import DetectionResponse, HealthResponse
from app.services.audio_processor import AudioProcessingError, load_audio
from app.services.detector import VoiceDetector
from app.services.reliability import assess_audio_quality
from app.services.robustness import run_robustness_scan
from app.services.scam_guidance import get_scam_guidance


router = APIRouter()
detector = VoiceDetector()


class GuidanceRequest(BaseModel):
    urgent_money_request: bool = False
    asks_for_otp_or_pin: bool = False
    caller_claims_emergency: bool = False


@router.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        app=settings.app_name,
        detector_ready=detector.ready,
    )


@router.post("/analyze", response_model=DetectionResponse)
async def analyze_audio(file: UploadFile = File(...)):
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

    if not detector.ready:
        raise HTTPException(
            status_code=503,
            detail="AASIST-L is unavailable. Check the model file and server logs.",
        )

    try:
        result = detector.predict(audio, sample_rate)
    except Exception as exc:
        print("VOICEMATRIX inference error:", repr(exc))
        raise HTTPException(
            status_code=500,
            detail="Audio analysis failed.",
        ) from exc

    quality = assess_audio_quality(audio, sample_rate)

    # These thresholds remain provisional until calibrated on development data.
    if result.score >= 0.70:
        verdict = "likely_genuine"
    elif result.score <= 0.30:
        verdict = "potentially_synthetic"
    else:
        verdict = "inconclusive"

    warnings = [
        "Model scores are not calibrated probabilities.",
        "A model result alone does not establish identity or fraud.",
    ]

    if quality["reasons"]:
        warnings.append(
            "Audio-quality checks: " + "; ".join(quality["reasons"])
        )

    if result.used_repeated_padding:
        warnings.append(
            "The recording was repeated to fill the model window. "
            "Interpret short-clip results cautiously."
        )

    # Conservative Decision Guard:
    # downgrade a decisive result when existing diagnostics indicate
    # that the available evidence may not support a reliable decision.
    score_disagreement = (
        result.window_score_range is not None
        and result.window_score_range >= 0.25
    )

    if score_disagreement:
        warnings.append(
            "Scores vary substantially across windows. "
            "Treat the classification with extra caution."
        )

    guard_reasons = []

    if quality.get("status") in {"limited", "insufficient_data"}:
        guard_reasons.append(
            "audio-quality checks indicate limited or insufficient evidence"
        )

    if score_disagreement:
        guard_reasons.append(
            "the genuine-speech scores vary substantially across windows"
        )

    if guard_reasons:
        previous_verdict = verdict

        if previous_verdict == "inconclusive":
            warnings.append(
                "Conservative Decision Guard confirmed that the result "
                "should remain inconclusive because: "
                + "; ".join(guard_reasons)
                + ". This safeguard identifies warning signs; it cannot "
                  "detect every model generalization error."
            )
        else:
            verdict = "inconclusive"

            warnings.append(
                "Conservative Decision Guard downgraded the result from "
                f"{previous_verdict.replace('_', ' ')} to inconclusive because: "
                + "; ".join(guard_reasons)
                + ". This safeguard identifies warning signs; it cannot "
                  "detect every model generalization error."
            )

    return DetectionResponse(
        verdict=verdict,
        score=result.score,
        reliability=quality["status"],
        explanation=result.explanation,
        duration_seconds=duration,
        model_name=result.model_name,
        inference_ms=result.inference_ms,
        window_count=result.window_count,
        window_score_range=result.window_score_range,
        bona_fide_logit=result.bona_fide_logit,
        window_bona_fide_logits=result.window_bona_fide_logits,
        quality_reasons=quality["reasons"],
        audio_rms=quality["rms_level"],
        peak_amplitude=quality["peak_amplitude"],
        clipping_fraction=quality["clipping_fraction"],
        near_silent_frame_fraction=quality["near_silent_frame_fraction"],
        warning=" ".join(warnings),
    )


@router.post("/robustness")
async def robustness_scan(file: UploadFile = File(...)):
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
            detail="Robustness testing requires audio between 2 and 10 seconds.",
        )

    if not detector.ready:
        raise HTTPException(
            status_code=503,
            detail="AASIST-L is unavailable.",
        )

    try:
        return run_robustness_scan(
            audio=audio,
            sample_rate=sample_rate,
            detector=detector,
        )
    except Exception as exc:
        print("VOICEMATRIX robustness scan error:", repr(exc))
        raise HTTPException(
            status_code=500,
            detail="Robustness analysis failed.",
        ) from exc


@router.post("/guidance")
def scam_guidance(request: GuidanceRequest):
    return {
        "recommendations": get_scam_guidance(
            urgent_money_request=request.urgent_money_request,
            asks_for_otp_or_pin=request.asks_for_otp_or_pin,
            caller_claims_emergency=request.caller_claims_emergency,
        )
    }
