"""Simple launcher for the voice + route-planning app.

This script asks whether you want to run the Mac or Windows variant
of the application, then starts the corresponding Gradio server.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    project_root = Path(__file__).resolve().parent

    print("=== Travel Resolver launcher ===")
    print("1) Mac  (apps/app-mac.py)")
    print("2) Windows (apps/app.py)")
    choice = input("Choix (1/2, mac/windows) : ").strip().lower()

    if choice in {"1", "mac", "m"}:
        script_name = "apps/app-mac.py"
    elif choice in {"2", "windows", "win", "w"}:
        script_name = "apps/app.py"
    else:
        print("Choix non reconnu, je pars sur Mac (app-mac.py).")
        script_name = "apps/app-mac.py"

    script_path = project_root / script_name
    if not script_path.exists():
        print(f"Impossible de trouver {script_name} à la racine du projet.")
        sys.exit(1)

    venv_python = project_root / ".venv" / "bin" / "python"
    if not venv_python.exists():
        venv_python = project_root / ".venv" / "Scripts" / "python.exe"

    python_exe = str(venv_python) if venv_python.exists() else sys.executable
    cmd = [python_exe, str(script_path)]
    print(f"Lancement de {script_name} avec : {' '.join(cmd)}")
    subprocess.run(cmd, check=False)


if __name__ == "__main__":
    main()
