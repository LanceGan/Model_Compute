from setuptools import setup, find_packages

setup(
    name="model-compute",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "streamlit>=1.30.0",
        "plotly>=5.18.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
    ],
    python_requires=">=3.10",
)
