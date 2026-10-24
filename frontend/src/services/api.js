const API_BASE =
  import.meta.env.VITE_API_BASE_URL || "http://localhost:8000/api";

async function readResponse(response) {
  const data = await response.json().catch(() => ({}));

  if (!response.ok) {
    throw new Error(
      data.detail || `Request failed with status ${response.status}`
    );
  }

  return data;
}

export async function analyzeAudio(file, model = "aasist") {
  const formData = new FormData();
  formData.append("file", file);

  const endpoint = model === "spectra"
    ? "analyze-spectra"
    : "analyze";

  const response = await fetch(`${API_BASE}/${endpoint}`, {
    method: "POST",
    body: formData
  });

  return readResponse(response);
}

export async function runRobustnessScan(file) {
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch(`${API_BASE}/robustness`, {
    method: "POST",
    body: formData
  });

  return readResponse(response);
}

export async function runSpectraRobustnessScan(file) {
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch(`${API_BASE}/spectra-robustness`, {
    method: "POST",
    body: formData
  });

  return readResponse(response);
}

export async function getScamGuidance(answers) {
  const response = await fetch(`${API_BASE}/guidance`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify(answers)
  });

  return readResponse(response);
}

export async function checkHealth() {
  const response = await fetch(`${API_BASE}/health`);
  return readResponse(response);
}
