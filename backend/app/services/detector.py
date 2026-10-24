from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import onnxruntime as ort


TARGET_SAMPLES = 64600
TARGET_SAMPLE_RATE = 16000
HOP_SAMPLES = TARGET_SAMPLES // 2


@dataclass
class DetectorResult:
    score: float
    model_name: str
    explanation: str
    inference_ms: float
    used_repeated_padding: bool
    window_count: int
    window_score_range: float | None
    bona_fide_logit: float
    window_bona_fide_logits: list[float]


class VoiceDetector:
    def __init__(self):
        self.model_name = "AASIST-L ONNX"
        self._session = None
        self._load_error = None

        model_path = (
            Path(__file__).resolve().parents[1]
            / "models"
            / "aasist-l.onnx"
        )

        try:
            if not model_path.exists():
                raise FileNotFoundError(str(model_path))

            options = ort.SessionOptions()
            options.intra_op_num_threads = 2
            options.inter_op_num_threads = 1

            self._session = ort.InferenceSession(
                str(model_path),
                sess_options=options,
                providers=["CPUExecutionProvider"],
            )

            inputs = self._session.get_inputs()
            outputs = self._session.get_outputs()

            if len(inputs) != 1 or len(outputs) != 1:
                raise RuntimeError(
                    "Unexpected model input/output configuration."
                )

            self._input_name = inputs[0].name
            self._output_name = outputs[0].name

            if inputs[0].shape[-1] != TARGET_SAMPLES:
                raise RuntimeError("Unexpected model input length.")

        except Exception as exc:
            self._session = None
            self._load_error = str(exc)

    @property
    def ready(self) -> bool:
        return self._session is not None

    def predict(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> DetectorResult:
        if not self.ready:
            raise RuntimeError(
                "AASIST-L unavailable: " + str(self._load_error)
            )

        if sample_rate != TARGET_SAMPLE_RATE:
            raise ValueError("Audio must be sampled at 16000 Hz.")

        audio = np.asarray(audio, dtype=np.float32).reshape(-1)

        if audio.size == 0 or not np.isfinite(audio).all():
            raise ValueError("Audio is empty or invalid.")

        used_repeated_padding = audio.size < TARGET_SAMPLES

        if used_repeated_padding:
            repeats = TARGET_SAMPLES // audio.size + 1
            windows = [
                np.tile(audio, repeats)[:TARGET_SAMPLES]
            ]
        else:
            last_start = audio.size - TARGET_SAMPLES
            starts = list(range(0, last_start + 1, HOP_SAMPLES))

            if not starts or starts[-1] != last_start:
                starts.append(last_start)

            windows = [
                audio[start:start + TARGET_SAMPLES]
                for start in starts
            ]

        model_input = np.ascontiguousarray(
            np.stack(windows),
            dtype=np.float32,
        )

        started = perf_counter()

        output = self._session.run(
            [self._output_name],
            {self._input_name: model_input},
        )[0]

        inference_ms = (perf_counter() - started) * 1000

        logits = np.asarray(output, dtype=np.float64)

        if (
            logits.ndim != 2
            or logits.shape != (len(windows), 2)
            or not np.isfinite(logits).all()
        ):
            raise RuntimeError("The model returned unexpected output.")

        # Class index 1 is bona-fide speech.
        # Preserve the raw logit for analysis.
        raw_bona_fide_logits = logits[:, 1]

        # Retain the previous normalized score for backward compatibility.
        # Softmax values are not calibrated probabilities.
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exponentials = np.exp(shifted)
        normalized = exponentials / exponentials.sum(
            axis=1, keepdims=True
        )
        genuine_scores = normalized[:, 1]

        score_range = (
            float(np.max(genuine_scores) - np.min(genuine_scores))
            if len(genuine_scores) > 1
            else None
        )

        return DetectorResult(
            score=float(np.mean(genuine_scores)),
            model_name=self.model_name,
            explanation=(
                "AASIST-L analyzes overlapping audio windows. The existing "
                "normalized score is retained for compatibility; raw "
                "bona-fide logits are also provided for evaluation. Neither "
                "output is a calibrated probability."
            ),
            inference_ms=round(inference_ms, 2),
            used_repeated_padding=used_repeated_padding,
            window_count=len(windows),
            window_score_range=(
                round(score_range, 6)
                if score_range is not None
                else None
            ),
            bona_fide_logit=float(np.mean(raw_bona_fide_logits)),
            window_bona_fide_logits=[
                float(value) for value in raw_bona_fide_logits
            ],
        )
