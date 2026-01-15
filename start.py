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
    print("1) Mac  (app-mac.py)")
    print("2) Windows (app.py)")
    choice = input("Choix (1/2, mac/windows) : ").strip().lower()

    if choice in {"1", "mac", "m"}:
        script_name = "app-mac.py"
    elif choice in {"2", "windows", "win", "w"}:
        script_name = "app.py"
    else:
        print("Choix non reconnu, je pars sur Mac (app-mac.py).")
        script_name = "app-mac.py"

    script_path = project_root / script_name
    if not script_path.exists():
        print(f"Impossible de trouver {script_name} à la racine du projet.")
        sys.exit(1)

    cmd = [sys.executable, str(script_path)]
    print(f"Lancement de {script_name} avec : {' '.join(cmd)}")
    subprocess.run(cmd, check=False)


if __name__ == "__main__":
    main()

