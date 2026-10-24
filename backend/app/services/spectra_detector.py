from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from time import perf_counter

import numpy as np
import onnxruntime as ort
from scipy.signal import lfilter


TARGET_SAMPLES = 64600
TARGET_SAMPLE_RATE = 16000
BONAFIDE_THRESHOLD = -1.0625009


@dataclass
class SpectraResult:
    score: float
    bona_fide_logit: float
    spoof_logit: float
    inference_ms: float
    used_repeated_padding: bool
    model_name: str
    explanation: str


class SpectraDetector:
    def __init__(self):
        # backend/app/services/spectra_detector.py -> project root
        self.model_path = (
            Path(__file__).resolve().parents[3]
            / "models"
            / "spectra0"
            / "spectra-0.onnx"
        )

        self.model_name = "Spectra-0 ONNX"
        self._session = None
        self._input_name = None
        self._output_name = None
        self._load_lock = Lock()
        self._load_error = None

    @property
    def model_file_exists(self):
        return self.model_path.is_file()

    @property
    def ready(self):
        return self._session is not None

    def _ensure_loaded(self):
        if self._session is not None:
            return

        with self._load_lock:
            if self._session is not None:
                return

            if not self.model_file_exists:
                raise FileNotFoundError(
                    f"Spectra-0 model not found: {self.model_path}"
                )

            try:
                options = ort.SessionOptions()
                options.intra_op_num_threads = 2
                options.inter_op_num_threads = 1

                session = ort.InferenceSession(
                    str(self.model_path),
                    sess_options=options,
                    providers=["CPUExecutionProvider"],
                )

                inputs = session.get_inputs()
                outputs = session.get_outputs()

                if len(inputs) != 1 or len(outputs) != 1:
                    raise RuntimeError(
                        "Unexpected Spectra-0 input/output configuration."
                    )

                if inputs[0].shape[-1] != TARGET_SAMPLES:
                    raise RuntimeError(
                        "Unexpected Spectra-0 input length."
                    )

                if outputs[0].shape[-1] != 2:
                    raise RuntimeError(
                        "Spectra-0 must return two class logits."
                    )

                self._input_name = inputs[0].name
                self._output_name = outputs[0].name
                self._session = session

            except Exception as exc:
                self._load_error = str(exc)
                raise RuntimeError(
                    "Spectra-0 could not be loaded: " + str(exc)
                ) from exc

    def predict(self, audio: np.ndarray, sample_rate: int):
        self._ensure_loaded()

        if sample_rate != TARGET_SAMPLE_RATE:
            raise ValueError(
                "Spectra-0 requires audio sampled at 16000 Hz."
            )

        audio = np.asarray(
            audio,
            dtype=np.float32,
        ).reshape(-1)

        if audio.size == 0 or not np.isfinite(audio).all():
            raise ValueError("Audio is empty or contains invalid values.")

        # Match the preprocessing used in our Spectra-0 evaluation.
        # y[n] = x[n] - 0.97 * x[n-1]
        audio = lfilter(
            [1.0, -0.97],
            [1.0],
            audio,
        ).astype(np.float32)

        used_repeated_padding = audio.size < TARGET_SAMPLES

        if used_repeated_padding:
            repeats = int(np.ceil(TARGET_SAMPLES / audio.size))
            audio = np.tile(audio, repeats)[:TARGET_SAMPLES]
        else:
            # Match the tested deterministic first-window policy.
            audio = audio[:TARGET_SAMPLES]

        model_input = np.ascontiguousarray(
            audio[np.newaxis, :],
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
            logits.shape != (1, 2)
            or not np.isfinite(logits).all()
        ):
            raise RuntimeError(
                "Spectra-0 returned an unexpected output."
            )

        # Class index 0 = spoof, index 1 = bona-fide.
        spoof_logit = float(logits[0, 0])
        bona_fide_logit = float(logits[0, 1])

        # The softmax score is for display compatibility only.
        # It is not a calibrated probability.
        shifted = logits[0] - np.max(logits[0])
        probabilities = np.exp(shifted)
        probabilities /= probabilities.sum()

        return SpectraResult(
            score=float(probabilities[1]),
            bona_fide_logit=bona_fide_logit,
            spoof_logit=spoof_logit,
            inference_ms=round(inference_ms, 2),
            used_repeated_padding=used_repeated_padding,
            model_name=self.model_name,
            explanation=(
                "Spectra-0 evaluates the first 4.0375 seconds of the "
                "prepared waveform. The verdict uses its genuine-class "
                "logit and the model-specific threshold. The displayed "
                "softmax score is not a calibrated probability."
            ),
        )
