from setuptools import setup, find_packages

setup(
    name='doorpost_detector',
    version='0.1.0',
    packages=find_packages(include=['doorpost_detector', 'doorpost_detector.*'])
    # packages=["doorpost_detector"]
)
