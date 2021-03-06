import pathlib
from setuptools import setup, find_packages

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()


setup(
    name='doorpost_detector',
    version='0.1.0',
    description="Detect doorposts from pointclouds",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/h0uter/doorpost_detector",
    author="Wouter Meijer",
    author_email="",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
    ],
    packages=find_packages(include=['doorpost_detector', 'doorpost_detector.*']),
    include_package_data=False,
    install_requires=["numpy", "open3d", "pyransac3d", "matplotlib"],
)
