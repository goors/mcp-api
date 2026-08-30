import datetime
import subprocess
from mcp.server.fastmcp import FastMCP
import requests
mcp = FastMCP("Elasticsearch Address Processor")

@mcp.tool()
def about_author() -> dict:
    """Returns background details, role, and preferences of the user operating this system."""
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
    return str(datetime.datetime.now())

@mcp.tool()
def call_weather_api(lat: float, lon: float) -> dict:
    """Fetches weather forecast from MET Norway locationforecast API and returns only the current/first record."""
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
    """Calculates an exact token count for the provided text using standard estimation or word heuristics."""
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
def run_docker_code(code: str, language: str = "node") -> dict:
    """Executes code inside an ephemeral Docker container and returns stdout/stderr."""
    try:
        # Ignore whatever bloated image string the model passes and enforce a local lightweight tag
        image = "node:current-alpine" if language.lower() in ["node", "js", "javascript"] else "python:3-alpine"

        if language.lower() in ["node", "js", "javascript"]:
            command = ["docker", "run", "--rm", "-i", image, "node", "-e", code]
        elif language.lower() in ["python", "py"]:
            command = ["docker", "run", "--rm", "-i", image, "python3", "-c", code]
        else:
            return {"error": f"Unsupported execution language: {language}"}

        print(command)
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=15
        )

        return {
            "exit_code": result.returncode,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip()
        }
    except subprocess.TimeoutExpired:
        return {"error": "Execution timed out after 15 seconds."}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    mcp.run()