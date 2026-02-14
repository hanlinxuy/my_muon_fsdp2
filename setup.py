"""Setup configuration for Muon FSDP2 package.

This file defines the package metadata, dependencies, and installation options
for the Muon FSDP2 package.
"""

from setuptools import find_packages, setup


def get_version() -> str:
    """Get package version from __init__.py."""
    # Read version from __init__.py (src layout)
    init_path = "src/muon_fsdp/__init__.py"
    with open(init_path, "r", encoding="utf-8") as f:
        content = f.read()

    for line in content.splitlines():
        if line.startswith("__version__"):
            # Parse version string
            version = line.split("=")[1].strip().strip('"').strip("'")
            return version

    return "0.1.0"  # Default version


def get_description() -> str:
    """Get package description."""
    readme_path = "README.md"
    try:
        with open(readme_path, "r", encoding="utf-8") as f:
            content = f.read()
            # Get first paragraph as description
            for line in content.split("\n"):
                if line.strip():
                    return line.strip()
    except FileNotFoundError:
        pass

    return "Muon optimizer with FSDP support"


setup(
    name="muon-fsdp",
    version=get_version(),
    author="Muon FSDP Contributors",
    description=get_description(),
    long_description=open("README.md", encoding="utf-8").read()
    if "README.md" in __import__("os").listdir(".")
    else None,
    long_description_content_type="text/markdown",
    url="https://github.com/muon-fsdp/muon-fsdp",
    license="MIT",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS :: macOS",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    packages=find_packages(where=["src"], exclude=["tests", "examples"]),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.10.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "pytest-mock>=3.10.0",
        ],
        "lint": [
            "ruff>=0.1.0",
            "mypy>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "muon-fsdp=muon_fsdp.cli:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
