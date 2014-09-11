from distutils.core import setup
with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    # Application name:
    name="Sequencer",

    # Version number (initial):
    version="0.0.1",

    # Application author details:
    author="Brandon Ogle",
    author_email="oglebrandon@gmail.com",

    # Packages
    packages=["sequencer"],

    # Include additional files into the package
    include_package_data=True,

    # Details
    #url="http://pypi.python.org/pypi/MyApplication_v010/",

    license="LICENSE.txt",
    description="Python Package to sequence an input network",

    long_description=open("README.md").read(),

    # Dependent packages (distributions)
    install_requires= required
)
