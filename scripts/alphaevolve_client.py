"""Standalone REST client for the Google Cloud AlphaEvolve API (Discovery Engine v1alpha).

Matches the exact wire contract of Google Cloud's official AlphaEvolve framework:
  - Session creation: .../assistants/default_assistant:streamAssist
  - Experiments:      .../{session}/alphaEvolveExperiments
  - Seed program:     .../{experiment}/alphaEvolvePrograms
  - Lifecycle:        .../{experiment}:start | :resume
  - Candidate loop:   .../{experiment}:acquirePrograms
                      .../{experiment}:submitProgramsEvaluations

Auth works automatically via `google.auth.default()` if installed, or falls back
to `gcloud auth print-access-token` so it runs cleanly in Cloud Shell with zero
extra pip installs.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional


def get_gcp_access_token() -> str:
    """Acquire an OAuth2 access token via ADC or gcloud CLI fallback."""
    env_token = os.environ.get("GCP_ACCESS_TOKEN")
    if env_token:
        return env_token.strip()

    try:
        import google.auth  # type: ignore
        import google.auth.transport.requests  # type: ignore

        creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
        req = google.auth.transport.requests.Request()
        if not creds.valid:
            creds.refresh(req)
        if creds.token:
            return str(creds.token)
    except Exception:
        pass

    # Clean fallback for Cloud Shell / local gcloud login
    for cmd in (
        ["gcloud", "auth", "application-default", "print-access-token"],
        ["gcloud", "auth", "print-access-token"],
    ):
        try:
            out = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL).strip()
            if out:
                return out
        except Exception:
            continue

    raise RuntimeError(
        "Unable to acquire GCP access token. Run `gcloud auth login` and "
        "`gcloud auth application-default login` (or run inside Cloud Shell)."
    )


def sanitize_relative_path(file_path: str) -> str:
    """Reject directory traversal in model-returned file paths."""
    cleaned = file_path.strip().lstrip("/")
    normalized = os.path.normpath(cleaned)
    if normalized.startswith("..") or "/../" in normalized or normalized == "..":
        raise ValueError(f"Unsafe relative path in candidate program: {file_path}")
    return normalized


def unflatten_program_content(flat_content: str) -> List[Dict[str, str]]:
    """Unflatten multi-file AlphaEvolve candidate text into [{path, content}]."""
    pattern = re.compile(
        r"""
        ^File:\s+(?P<path>[^\n]+)\n
        (?:Language:\s+(?P<language>[^\n]+)\n)?
        (?:Description:\s+(?P<description>[^\n]+)\n)?
        ---\n
        (?P<content>.*?)
        \n---""",
        re.MULTILINE | re.VERBOSE | re.DOTALL,
    )
    files: List[Dict[str, str]] = []
    for match in pattern.finditer(flat_content):
        safe_path = sanitize_relative_path(match.group("path"))
        files.append({"path": safe_path, "content": match.group("content")})
    return files


def fix_multi_files_program(program: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize multi-file program responses when flattened into files[0].content."""
    files = program.get("content", {}).get("files", [])
    if files and isinstance(files[0].get("content"), str) and files[0]["content"].startswith("File:"):
        unflattened = unflatten_program_content(files[0]["content"])
        if unflattened:
            program["content"]["files"] = unflattened
    return program


class AlphaEvolveAPIError(RuntimeError):
    """Raised on non-2xx HTTP responses from the AlphaEvolve API."""

    def __init__(self, status_code: int, url: str, body: str):
        self.status_code = status_code
        self.url = url
        self.body = body
        super().__init__(f"AlphaEvolve API HTTP {status_code} on {url}: {body[:1200]}")


class AlphaEvolveRestClient:
    """Minimal, self-contained REST client for AlphaEvolve on Discovery Engine v1alpha."""

    def __init__(
        self,
        project_id: str,
        engine_id: str = "global-gemini-enterprise",
        location: str = "global",
        collection: str = "default_collection",
        assistant_id: str = "default_assistant",
        timeout_sec: float = 60.0,
    ):
        self.project_id = project_id
        self.engine_id = engine_id
        self.location = location
        self.collection = collection
        self.assistant_id = assistant_id
        self.timeout_sec = timeout_sec

        host = "discoveryengine.googleapis.com"
        if location.upper() == "EU":
            host = f"eu-{host}"
        elif location.upper() == "US":
            host = f"us-{host}"
        self.base_url = f"https://{host}"
        self.base_path = (
            f"projects/{self.project_id}/locations/{self.location}"
            f"/collections/{self.collection}/engines/{self.engine_id}"
        )

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {get_gcp_access_token()}",
            "Content-Type": "application/json",
            "X-Goog-User-Project": self.project_id,
        }

    def _request(
        self,
        method: str,
        url: str,
        payload: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Any:
        if params:
            url = f"{url}?{urllib.parse.urlencode(params)}"
        data_bytes = json.dumps(payload).encode("utf-8") if payload is not None else None
        req = urllib.request.Request(url, data=data_bytes, headers=self._headers(), method=method)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
                raw = resp.read().decode("utf-8")
                return json.loads(raw) if raw.strip() else {}
        except urllib.error.HTTPError as exc:
            err_body = exc.read().decode("utf-8", errors="replace")
            raise AlphaEvolveAPIError(exc.code, url, err_body) from exc

    def create_session(self) -> str:
        """Create a Gemini Enterprise session via default_assistant:streamAssist."""
        url = f"{self.base_url}/v1alpha/{self.base_path}/assistants/{self.assistant_id}:streamAssist"
        resp = self._request(
            "POST",
            url,
            payload={
                "query": {"text": "starting alpha evolve query"},
                "assistSkippingMode": "REQUEST_ASSIST",
            },
        )
        answers = resp if isinstance(resp, list) else [resp]
        for ans in answers:
            sess = ans.get("sessionInfo", {}).get("session")
            if sess:
                return str(sess)
        raise RuntimeError(f"Could not extract sessionInfo.session from streamAssist response: {resp}")

    def create_experiment(self, session_name: str, exp_config: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.base_url}/v1alpha/{session_name}/alphaEvolveExperiments"
        return self._request("POST", url, payload={"config": exp_config})

    def get_experiment(self, experiment_name: str) -> Dict[str, Any]:
        url = f"{self.base_url}/v1alpha/{experiment_name}"
        return self._request("GET", url)

    def create_initial_program(self, experiment_name: str, initial_program: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.base_url}/v1alpha/{experiment_name}/alphaEvolvePrograms"
        return self._request("POST", url, payload=initial_program)

    def start_experiment(self, experiment_name: str) -> Dict[str, Any]:
        url = f"{self.base_url}/v1alpha/{experiment_name}:start"
        return self._request("POST", url, payload={"name": experiment_name})

    def resume_experiment(self, experiment_name: str) -> Dict[str, Any]:
        url = f"{self.base_url}/v1alpha/{experiment_name}:resume"
        resp = self._request("POST", url, payload={"name": experiment_name})
        for attempt in range(6):
            state = self.get_experiment(experiment_name).get("state")
            if state == "RUNNING":
                break
            time.sleep(attempt + 1)
        return resp

    def acquire_programs(self, experiment_name: str, desired_programs_count: int = 1) -> List[Dict[str, Any]]:
        url = f"{self.base_url}/v1alpha/{experiment_name}:acquirePrograms"
        resp = self._request(
            "POST",
            url,
            payload={
                "parent": experiment_name,
                "desired_programs_count": desired_programs_count,
            },
        )
        programs = resp.get("programs", []) if isinstance(resp, dict) else []
        return [fix_multi_files_program(p) for p in programs]

    def submit_program_evaluations(
        self,
        experiment_name: str,
        submissions: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        url = f"{self.base_url}/v1alpha/{experiment_name}:submitProgramsEvaluations"
        return self._request(
            "POST",
            url,
            payload={
                "parent": experiment_name,
                "evaluation_submissions": submissions,
            },
        )

    def list_programs(self, experiment_name: str, page_size: int = 100) -> List[Dict[str, Any]]:
        url = f"{self.base_url}/v1alpha/{experiment_name}/alphaEvolvePrograms"
        all_progs: List[Dict[str, Any]] = []
        page_token: Optional[str] = None
        while True:
            params: Dict[str, Any] = {"pageSize": page_size}
            if page_token:
                params["pageToken"] = page_token
            resp = self._request("GET", url, params=params)
            for p in resp.get("alphaEvolvePrograms", []):
                all_progs.append(fix_multi_files_program(p))
            page_token = resp.get("nextPageToken")
            if not page_token:
                break
        return all_progs
