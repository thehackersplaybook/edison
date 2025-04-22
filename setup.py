from setuptools import setup, find_packages

long_description = ""

with open("README.md", "r") as fh:
    long_description = fh.read()


setup(
    name="edison",
    version="0.0.1",
    packages=find_packages(exclude=["tests*", "e2e*"]),
    install_requires=[
        "python-dotenv>=1.0.1",
        "pydantic>=2.10.6",
        "rich>=13.9.4",
        "openai>=1.61.1",
        "openai-agents>=0.0.11",
        "argparse>=1.4.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
        ],
    },
    author="Aditya Patange (AdiPat)",
    author_email="contact.adityapatange@gmail.com",
    description="A simple, effective and powerful package to integrate Deep Research capabilities in Python.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/thehackersplaybook/edison",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    license_files=("LICENSE",),
)
