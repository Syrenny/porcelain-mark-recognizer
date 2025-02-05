import subprocess

from environment import settings


def run_backend():
    return subprocess.Popen(["uvicorn", "backend:app", "--host", settings.api_host, "--port", settings.api_port])


def run_frontend():
    return subprocess.Popen(["streamlit", "run", "frontend.py", "--server.address", settings.front_host, "--server.port", settings.front_port])


if __name__ == "__main__":
    backend_process = run_backend()
    frontend_process = run_frontend()

    try:
        backend_process.wait()
        frontend_process.wait()
    except KeyboardInterrupt:
        backend_process.terminate()
        frontend_process.terminate()
