from setuptools import setup, find_packages

setup(
    name="ai-agent",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "duckduckgo-search",
        "playwright",
        "beautifulsoup4",
        "sentence-transformers",
        "faiss-cpu",
        "requests",
    ],
) 