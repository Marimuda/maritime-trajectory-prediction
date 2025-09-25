# \!/usr/bin/env python3
"""
Script to set up the environment for maritime trajectory prediction
- Creates necessary directories
- Downloads example data if needed
- Configures environment variables
"""

import argparse
import subprocess
import sys
from pathlib import Path


def create_directories():
    """Create necessary directories for the project"""
    # Get project root
    project_root = Path(__file__).resolve().parent.parent

    # Create directories
    dirs = [
        "data",
        "data/raw",
        "data/processed",
        "outputs",
        "outputs/models",
        "outputs/logs",
        "outputs/figures",
    ]

    for dir_name in dirs:
        dir_path = project_root / dir_name
        dir_path.mkdir(exist_ok=True)
        print(f"Created directory: {dir_path}")


def check_dependencies():
    """Check if all dependencies are installed"""
    # Get project root
    project_root = Path(__file__).resolve().parent.parent

    # Check dependencies using pip
    requirements_file = project_root / "requirements.txt"
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "check"], check=True, capture_output=True
        )
        print("All dependencies are satisfied.")
    except subprocess.CalledProcessError:
        print("Some dependencies are missing or have conflicts.")
        install = input("Do you want to install/upgrade dependencies? (y/n): ")
        if install.lower() == "y":
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-r", str(requirements_file)],
                check=True,
            )
            print("Dependencies installed successfully.")


def configure_environment():
    """Configure environment variables"""
    # Get project root
    project_root = Path(__file__).resolve().parent.parent

    # Set environment variables
    env_vars = {
        "MARITIME_DATA_DIR": str(project_root / "data"),
        "MARITIME_OUTPUT_DIR": str(project_root / "outputs"),
        "PYTHONPATH": str(project_root),
    }

    # Write to .env file
    env_file = project_root / ".env"
    with open(env_file, "w") as f:
        for key, value in env_vars.items():
            f.write(f"{key}={value}\n")

    print(f"Environment variables written to {env_file}")
    print("Please source this file or set these variables in your environment:")
    for key, value in env_vars.items():
        print(f"export {key}={value}")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Set up the environment for maritime trajectory prediction"
    )
    parser.add_argument(
        "--no-dirs", action="store_true", help="Skip directory creation"
    )
    parser.add_argument(
        "--no-deps", action="store_true", help="Skip dependency checking"
    )
    parser.add_argument(
        "--no-env", action="store_true", help="Skip environment configuration"
    )

    return parser.parse_args()


def main():
    """Main function"""
    args = parse_args()

    print("Setting up environment for maritime trajectory prediction...")

    if not args.no_dirs:
        create_directories()

    if not args.no_deps:
        check_dependencies()

    if not args.no_env:
        configure_environment()

    print(r"Environment setup complete\!")


if __name__ == "__main__":
    main()
