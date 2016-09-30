from setuptools import setup

with open('requirements.txt') as f:
    required = list(f.read().splitlines())

with open('sequencer/__init__.py') as f:
    for line in f:
        if line.find("__version__") >= 0:
            version = line.split("=")[1].strip()
            version = version.strip('"')
            version = version.strip("'")
            continue

setup(
    # Application name:
    name="Sequencer",

    version=version,
    # Application author details:
    author="Brandon Ogle",

    # Packages
    packages=["sequencer"],

    # Details
    #url="http://pypi.python.org/pypi/MyApplication_v010/",

    license="LICENSE.txt",
    description="Python Package to sequence an input network",

    long_description=open("README.md").read(),

    # Dependent packages (distributions)
    install_requires= required,
    scripts = ["run_sequencer.py"]
)
