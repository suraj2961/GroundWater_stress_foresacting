import subprocess
import sys
import os

SCRIPT_DIR = os.path.dirname(__file__)
PY = sys.executable

def run(name):
    path = os.path.join(SCRIPT_DIR, name)
    print("Running", name)
    subprocess.run([PY, path], check=True, cwd=SCRIPT_DIR)

if __name__ == "__main__":
    try:
        run("merge_panel.py")
        run("forecast.py")
        run("generate_dashboard.py")
        # open dashboard (Windows)
        dashboard = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "outputs", "dashboard.html"))
        print("Opening", dashboard)
        os.startfile(dashboard)
    except subprocess.CalledProcessError as e:
        print("Step failed:", e)
        sys.exit(e.returncode)