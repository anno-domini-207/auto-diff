import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()
setuptools.setup(
    name='AnnoDomini',
    version='0.28',
    scripts=['AnnoDomini/AutoDiff.py'],
    author="Simon Warchol",
    author_email="simonwarchol@g.harvard.edu",
    description="Harvard CS207 Automatic Differentiation Project",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/anno-domini-207/cs207-FinalProject/",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
