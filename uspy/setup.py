import setuptools

# python packaging guides:
# http://python-packaging.readthedocs.io/en/latest/minimal.html
# https://packaging.python.org/tutorials/
# https://setuptools.readthedocs.io/en/latest/setuptools.html

# versioning scheme
# https://packaging.python.org/guides/distributing-packages-using-setuptools/#choosing-a-versioning-scheme

# possible classifiers:
# https://pypi.org/pypi?%3Aaction=list_classifiers

# including scripts in the distribution (use entry_points for cross-platform compatibility)
# https://setuptools.readthedocs.io/en/latest/setuptools.html#automatic-script-creation


__version__ = "0.1dev"

with open("README.md", "r") as fh:
    long_description = fh.read()


required_packages = ["gdal>=2.2.4",
                    "numpy>=1.11.0",
                    "matplotlib>=2.0.0",
                    "scikit-image>=1.3.0",
                    "opencv-python>=0.4",
                    "opencv-contrib-python>=0.4",
                    "scipy>=1.1.0",
                    "scikit-learn>=0.19.1"]


setuptools.setup(
    name="nmapy",
    version=__version__,
    author="Jacob Arndt",
    author_email="arndt204@umn.edu",
    license="MIT",
    description="a library for creating spatial-contextual features for satellite images",
    keywords="python gis geography image settlement texture spatial",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jwarndt/nmapy",
    packages=setuptools.find_packages(),
    install_requires=required_packages,
    classifiers=(
        "Programming Language :: Python",
        'Development Status :: 3 - Alpha',
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering"
    ),
    entry_points={
        "console_scripts": []
    }
)