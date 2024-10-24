# setup.py

from setuptools import setup, find_packages

with open("readme.md", "r") as f:
    long_description = f.read()

setup(
    name="langgraph_lib",
    version="0.1.0",
    description="A toolkit to serve LangGraph agents using FastAPI.",
    author="Vishvdeep Dasadiya",
    author_email="vishvdeep.dasadiya.vd@gmail.com",
    url="https://github.com/aiwithvd/langgraph_lib",
    packages=find_packages(where="src"),
    long_description=long_description,
    long_description_content_type="text/markdown",
    package_dir={"": "src"},
    license="MIT",  # Changed 'libraries' to 'license'
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "fastapi>=0.95.2",
        "uvicorn[standard]>=0.22.0",
        "httpx>=0.24.1",
        "pydantic>=1.10.7",
        "langchain_core>=0.0.5",
        "langgraph>=0.1.0",
        "langsmith>=0.0.4",
    ],
    python_requires=">=3.7",
    keywords="langgraph fastapi agent toolkit ai",
    project_urls={
        "Bug Reports": "https://github.com/aiwithvd/langgraph_lib/issues",
        "Source": "https://github.com/aiwithvd/langgraph_lib",
    },
)