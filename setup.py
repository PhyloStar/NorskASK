import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="masterthesis",
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
    install_requires=[
        'numpy==1.15.1',
        'pandas==0.23.4',
        'keras==2.2.2',
        'gensim==3.6.0',
        'scipy==1.1.0',
        'scikit-learn==0.19.2',
        'matplotlib==2.2.3',
        'tensorflow==1.10.0',
        'ipykernel'
    ]
)
