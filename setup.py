from __future__ import annotations

from pathlib import Path

from setuptools import find_packages, setup

BASE_DIR = Path(__file__).resolve().parent
README_PATH = BASE_DIR / "README.md"
REQUIREMENTS_PATH = BASE_DIR / "requirements.txt"


def read_requirements() -> list[str]:
    if not REQUIREMENTS_PATH.exists():
        return []
    return [
        line.strip()
        for line in REQUIREMENTS_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]


setup(
    name="pspace",
    version="1.0",
    description="Probabilistic space package for uncertainty quantification and optimization under uncertainty",
    long_description=README_PATH.read_text(encoding="utf-8") if README_PATH.exists() else "",
    long_description_content_type="text/markdown",
    author="Komahan Boopathy",
    author_email="komibuddy@gmail.com",
    packages=find_packages(exclude=("tests", "examples")),
    include_package_data=True,
    install_requires=read_requirements(),
    python_requires=">=3.9",
)
