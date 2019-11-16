import setuptools
with open("README.md", "r") as fh:
    long_description = fh.read()
setuptools.setup(
     name='AnnoDomini',  
     version='0.1',
     scripts=['AnnoDomini/AutoDiff.py'] ,
     author="Simon Warchol",
     author_email="simonwarchol@g.harvard.edu",
     description="Harvar CS207 Automatic Differentiation Projec",
     long_description=long_description,
   long_description_content_type="text/markdown",
     url="https://github.com/javatechy/dokr",
     packages=setuptools.find_packages(),
     classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: MIT License",
         "Operating System :: OS Independent",
     ],
 )