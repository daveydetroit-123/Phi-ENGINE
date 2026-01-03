from setuptools import setup, find_packages

setup(
    name="phi-engine",
    version="1.0.0",
    description="Geometric Stabilization for Neural Networks",
    author="The Monkey and The Toaster",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch>=2.0.0",
        "numpy"
    ],
)
