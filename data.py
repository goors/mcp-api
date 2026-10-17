import datetime
import time
import uuid
from mcp.server.transport_security import TransportSecuritySettings
#from mcp.server.fastmcp import FastMCP
from fastmcp import FastMCP
from fastmcp.apps import AppConfig, ResourceCSP
import requests
mcp = FastMCP("Elasticsearch Address Processor"
            )
from kubernetes import client, config
from kubernetes.client.rest import ApiException
import os


# Initialize Kubernetes configuration from environment variables, kubeconfig, or in-cluster service account
try:
    if os.getenv("KUBERNETES_SERVICE_HOST"):
        config.load_incluster_config()
    else:
        config.load_kube_config()
except Exception:
    pass  # Fallback handled or caught during tool execution

@mcp.tool()
def about_author() -> dict:
    """Retrieve personal biography, technical stack, and background profile of the system owner. ONLY call this when the user explicitly asks about the author's background, bio, tech stack, or personal details. Never use for general programming questions."""
    return {
        "name": "Nikola Derikonjić",
        "age": 46,
        "location": "Serbia (home base on Zlatibor)",
        "occupation": "Systems engineer, software developer, DevOps practitioner, and owner of INSIGHTFUL STAYS",
        "technical_stack": {
            "languages_frameworks": ["Python", "Go", ".NET", "Node.js", "FastAPI", "Django", "React", "Vue.js", "Tauri"],
            "infrastructure": ["Kubernetes", "Helm", "FluxCD", "Docker", "Azure", "GCP", "AWS", "DigitalOcean", "Traefik", "NGINX"],
            "databases_storage": ["PostgreSQL", "Elasticsearch", "MongoDB", "MinIO AIStor", "Valkey"],
            "environment": ["Apple Silicon Mac", "Linux hosts", "Self-hosted Postfix/Dovecot mail"]
        },
        "interests_hobbies": [
            "Riding a BMW GS Adventure motorcycle",
            "Downhill mountain biking (2014 Scott Gambler)",
            "Boxing and jump rope training",
            "Stand-up paddleboarding and windsurfing"
        ],
        "email": "nikola@intovoid.dev",
        "current_situation": "Well, suffering from a broken heart",
        "working_style_preference": "Prefers direct, performant, deterministic code and native database queries over over-engineered agentic loops or unnecessary abstraction layers."
    }

@mcp.tool()
def today() -> str:
    """Returns the current date and time. ONLY call this when the user explicitly asks for the exact current date or time."""
    return str(datetime.datetime.now())

@mcp.tool()
def call_weather_api(lat: float, lon: float) -> dict:
    """Fetches real-time weather forecasts for specific geographic coordinates. ONLY call this when the user explicitly asks about weather conditions for a location.
    If the user names a place (e.g. "Zlatibor"), use your best-known approximate coordinates for that place — do not refuse and do not ask the user for coordinates."""
    url = f"https://api.met.no/weatherapi/locationforecast/2.0/compact?lat={lat}&lon={lon}"
    headers = {"User-Agent": "LocalDevAgent/1.0 (nikola@insightfulstays.com)"}
    response = requests.get(url, headers=headers, timeout=10.0)
    response.raise_for_status()
    data = response.json()

    try:
        first_entry = data["properties"]["timeseries"][0]
        return {
            "time": first_entry["time"],
            "details": first_entry["data"]["instant"]["details"],
            "summary": first_entry["data"]["next_1_hours"]["summary"]["symbol_code"]
        }
    except (KeyError, IndexError) as e:
        return {"error": f"Failed to parse response structure: {str(e)}", "raw": data}

@mcp.tool()
def count_tokens(text: str) -> dict:
    """Calculates an exact token count for a provided block of text. ONLY call this when the user explicitly asks to count tokens in a specific string."""
    if not text:
        return {"characters": 0, "words": 0, "estimated_tokens": 0}

    chars = len(text)
    words = len(text.split())
    # Standard rule of thumb: ~4 characters per token for English text/code
    estimated_tokens = max(1, round(chars / 4))

    return {
        "characters": chars,
        "words": words,
        "estimated_tokens": estimated_tokens
    }

@mcp.tool()
def run__code(code: str, language: str = "node") -> dict:
    """Executes arbitrary Node.js or Python code inside a sandboxed Kubernetes pod. ONLY call this when the user explicitly asks you to run, execute, or test code snippets. Never use this for explaining code or answering conceptual programming questions."""
    pod_name = None
    namespace = os.getenv("KUBERNETES_NAMESPACE", "code-runner-env")

    try:
        if language.lower() in ["node", "js", "javascript"]:
            image = "node:current-alpine"
            command = ["node", "-e", code]
        elif language.lower() in ["python", "py"]:
            image = "python:3-alpine"
            command = ["python3", "-c", code]
        else:
            return {"error": f"Unsupported execution language: {language}"}

        v1 = client.CoreV1Api()
        pod_name = f"mcp-exec-{uuid.uuid4().hex[:8]}"

        pod_manifest = client.V1Pod(
            metadata=client.V1ObjectMeta(
                name=pod_name,
                namespace=namespace,
                labels={"app.kubernetes.io/managed-by": "mcp-code-executor"}
            ),
            spec=client.V1PodSpec(
                # Use a sandboxed runtime (like gVisor) if available in your cluster
                # to isolate the kernel from untrusted user code.
                runtime_class_name="gvisor",
                service_account_name="restricted-app-sa",
                automount_service_account_token=False,
                restart_policy="Never",
                containers=[
                    client.V1Container(
                        name="code-runner",
                        image=image,
                        command=command,
                        resources=client.V1ResourceRequirements(
                            limits={"cpu": "200m", "memory": "256Mi", "ephemeral-storage": "64Mi"},
                            requests={"cpu": "50m", "memory": "64Mi", "ephemeral-storage": "16Mi"}
                        ),
                        security_context=client.V1SecurityContext(
                            allow_privilege_escalation=False,
                            run_as_non_root=True,
                            run_as_user=1000,
                            read_only_root_filesystem=True,
                            # Enforce a default secure seccomp profile
                            seccomp_profile=client.V1SeccompProfile(type="RuntimeDefault"),
                            capabilities=client.V1Capabilities(
                                drop=["ALL"]
                            )
                        )
                    )
                ]
            )
        )

        v1.create_namespaced_pod(namespace=namespace, body=pod_manifest)

        timeout = 60
        start_time = time.time()
        pod_status = None

        while True:
            pod_status = v1.read_namespaced_pod_status(name=pod_name, namespace=namespace)
            phase = pod_status.status.phase

            if phase in ["Succeeded", "Failed"]:
                break

            if time.time() - start_time > timeout:
                try:
                    v1.delete_namespaced_pod(
                        name=pod_name,
                        namespace=namespace,
                        body=client.V1DeleteOptions(grace_period_seconds=0)
                    )
                except Exception:
                    pass
                return {"error": "Execution timed out after 15 seconds."}

            time.sleep(0.5)

        logs = v1.read_namespaced_pod_log(
            name=pod_name,
            namespace=namespace,
            container="code-runner"
        )

        exit_code = 0
        container_statuses = pod_status.status.container_statuses
        if container_statuses:
            state = container_statuses[0].state
            if state.terminated:
                exit_code = state.terminated.exit_code

        return {
            "exit_code": exit_code,
            "stdout": logs.strip(),
            "stderr": ""
        }

    except ApiException as e:
        return {
            "error": f"Kubernetes API error: {e.reason} ({e.status})",
            "body": e.body,
            "host": client.Configuration.get_default_copy().host
        }
    except Exception as e:
        return {"error": str(e)}
    finally:
        pass
        if pod_name:
            try:
                v1 = client.CoreV1Api()
                v1.delete_namespaced_pod(
                    name=pod_name,
                    namespace=namespace,
                    body=client.V1DeleteOptions(grace_period_seconds=0)
                )
            except Exception:
                pass
import ipaddress
import json
import socket
from urllib.parse import urlparse

import requests
import trafilatura
from bs4 import BeautifulSoup

ALLOWED_SCHEMES = {"http", "https"}
BLOCKED_HOSTNAMES = {"localhost", "metadata.google.internal"}


def _resolve_and_check(host: str) -> str | None:
    """Resolve host, return None if any resolved IP is internal, else return the IP."""
    if not host:
        return None
    if host in BLOCKED_HOSTNAMES or host.endswith((".local", ".internal", ".localhost")):
        return None
    try:
        # getaddrinfo returns every A/AAAA record
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror:
        return None
    for info in infos:
        ip_str = info[4][0]
        # strip IPv6 scope id like 'fe80::1%eth0'
        ip_str = ip_str.split("%", 1)[0]
        try:
            addr = ipaddress.ip_address(ip_str)
        except ValueError:
            return None
        if (addr.is_private or addr.is_loopback or addr.is_link_local
                or addr.is_reserved or addr.is_multicast or addr.is_unspecified):
            return None
    return infos[0][4][0]


def _html_to_text(html: str, url: str) -> str:
    """Extract main readable text from HTML. Returns '' if extraction fails."""
    # trafilatura is the best signal-to-noise extractor for articles/pages
    extracted = trafilatura.extract(
        html,
        url=url,
        include_comments=False,
        include_tables=True,
        favor_precision=True,     # stricter, drops boilerplate
        deduplicate=True,
    )
    if extracted and extracted.strip():
        return extracted.strip()

    # Fallback: strip tags and collapse whitespace
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "template", "svg", "iframe"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    lines = (line.strip() for line in text.splitlines())
    return "\n".join(line for line in lines if line)


@mcp.tool(
)
def http_request(
        url: str,
        method: str = "GET",
        headers: str = "{}",
        body: str = "",
        timeout_s: int = 15,
        mode: str = "text",   # "text" | "raw"
) -> dict:
    """Fetch a URL and return readable text (default) or raw body.
    Blocks private/internal addresses and non-http(s) schemes."""
    parsed = urlparse(url)
    if parsed.scheme not in ALLOWED_SCHEMES:
        return {"error": f"Unsupported scheme: {parsed.scheme or '(none)'}"}

    if _resolve_and_check(parsed.hostname or "") is None:
        return {"error": f"Refusing to request internal or unresolvable host: {parsed.hostname}"}

    try:
        hdrs = json.loads(headers) if headers else {}
    except json.JSONDecodeError:
        return {"error": "Headers must be valid JSON"}

    try:
        resp = requests.request(
            method.upper(),
            url,
            headers=hdrs,
            data=body.encode() if body else None,
            timeout=min(timeout_s, 30),
            allow_redirects=False,  # keep this — redirects are a common SSRF pivot
            # Cap download so a 5GB response can't OOM you
            stream=True,
        )
    except requests.RequestException as e:
        return {"error": str(e)}

    # Read at most ~2MB, then stop
    max_bytes = 2 * 1024 * 1024
    chunks, total = [], 0
    for chunk in resp.iter_content(chunk_size=8192):
        chunks.append(chunk)
        total += len(chunk)
        if total > max_bytes:
            break
    raw = b"".join(chunks)
    if total > max_bytes:
        raw = raw[:max_bytes]

    # Decode with the charset the server declared, else utf-8
    encoding = resp.encoding or "utf-8"
    try:
        text = raw.decode(encoding, errors="replace")
    except LookupError:
        text = raw.decode("utf-8", errors="replace")

    if mode == "raw":
        out = text
    else:
        ctype = resp.headers.get("Content-Type", "").lower()
        if "html" in ctype or text.lstrip().startswith(("<", "<!doctype", "<!DOCTYPE")):
            out = _html_to_text(text, url)
        else:
            out = text  # JSON, plain text, XML — return as-is

    return {
        "status": resp.status_code,
        "headers": dict(resp.headers),
        "text": out,
        "bytes_read": total
    }

# mcp.settings.host = "0.0.0.0"
# mcp.settings.port = 8001

if __name__ == "__main__":
    mcp.run(transport="sse", host="0.0.0.0", port=8001)