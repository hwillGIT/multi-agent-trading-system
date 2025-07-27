"""
Setup configuration for the Multi-Agent Trading System.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="multi-agent-trading-system",
    version="1.0.0",
    author="Trading System Team",
    author_email="team@tradingsystem.com",
    description="A comprehensive multi-agent trading system for market analysis and strategy generation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/multi-agent-trading-system",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Topic :: Office/Business :: Financial :: Investment",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.11",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.4.0",
            "pre-commit>=3.3.0",
        ],
        "jupyter": [
            "jupyter>=1.0.0",
            "jupyterlab>=4.0.0",
            "notebook>=7.0.0",
            "ipywidgets>=8.0.0",
        ],
        "monitoring": [
            "prometheus-client>=0.17.0",
            "grafana-api>=1.0.3",
        ],
        "web": [
            "streamlit>=1.25.0",
            "dash>=2.11.0",
            "plotly>=5.14.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "trading-system=main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "trading_system": [
            "config/*.yaml",
            "config/*.json",
        ],
    },
    zip_safe=False,
)