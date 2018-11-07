import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="src",
    version="0.0.1",
    author="Stig Johan Berggren",
    author_email="stigjb@gmail.com",
    description="Master thesis in NLP",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.uio.no/stigjb/master-thesis",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
