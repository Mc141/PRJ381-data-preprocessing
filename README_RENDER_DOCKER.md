# Local Docker build & test for this FastAPI project

This repository contains a FastAPI app in `app/` and a Dockerfile tailored to build and run it.

Below are the exact PowerShell commands to build, run, and test the container locally.

1) Build the image (run from repo root):

```powershell
# Replace <tag> with a name you prefer, e.g. prj381-api:local
docker build -t prj381-api:local .
```

2) Run the container (map port 8000):

```powershell
docker run --rm -p 8000:8000 --name prj381-run prj381-api:local
```

3) Quick smoke test (from another terminal):

```powershell
# check root/status endpoint
curl http://localhost:8000/  
# or the status router if present
curl http://localhost:8000/status
```

If the app exposes other endpoints (e.g. `/predictions`), test them similarly.

Notes:
- The Dockerfile upgrades pip/wheel/setuptools and installs system libs required by packages like rasterio and lxml.
- The image uses Python 3.12 as requested; building may take time the first run due to compiling some packages.
- If build or runtime fails because a system dependency is missing, paste the failing logs and I'll adjust the Dockerfile.
