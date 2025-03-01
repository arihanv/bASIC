from setuptools import setup, find_packages

setup(
    name="kernelbench",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "anthropic>=0.18.0",
        "python-dotenv>=1.0.0",
        "openai>=1.12.0",
        "torch>=2.2.0",
        "numpy>=1.24.0",
    ],
    python_requires=">=3.8",
    author="Ryan Rong",
    description="Tools for testing and benchmarking CUDA kernel implementations using LLM-generated code",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
