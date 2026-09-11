"""Create a project-local virtual environment with pip."""

from pathlib import Path
import venv

# Resolve relative to this file so the working directory does not matter.
venv.EnvBuilder(with_pip=True).create(Path(__file__).resolve().parent / ".venv")
print("Created .venv. Follow README.md to install dependencies.")