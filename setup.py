from setuptools import setup, find_packages

setup(
    name="arxiv-ideate",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "numpy",
        "sentence-transformers",
        "scikit-learn",
        "hdbscan",
        "umap-learn",
        "kagglehub",
    ],
    entry_points={
        "console_scripts": [
            "arxiv-ideate=main:main",
        ],
    },
    author="Arxiv Ideate Team",
    description="An AI-powered research discovery and ideation engine.",
)
