import subprocess
import sys
import os

def install_requirements(requirements_file="utils/setup/requirements.txt"):
    if not os.path.exists(requirements_file):
        print(f"{requirements_file} not found!")
        return
    
    installed = []
    failed = []

    with open(requirements_file, "r") as f:
        for line in f:
            pkg = line.strip()
            if not pkg or pkg.startswith("#"):  # skip empty lines & comments
                continue

            print(f"\nInstalling: {pkg}")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
                installed.append(pkg)
            except subprocess.CalledProcessError:
                print(f"❌ Failed to install {pkg}")
                failed.append(pkg)

    print("\n===== Installation Summary =====")
    if installed:
        print("✅ Installed:", ", ".join(installed))
    if failed:
        print("❌ Failed:", ", ".join(failed))

if __name__ == "__main__":
    install_requirements()
