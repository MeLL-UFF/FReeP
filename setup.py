import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="freep", # Replace with your own username
    version="0.0.1",
    author="Daniel Junior",
    author_email="danieljunior@id.uff.br",
    description="Feature Recommender from Preferences",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/MeLL-UFF/FReeP",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.5.4',
    dependency_links=['https://github.com/joeddav/get_smarties.git#egg=get_smarties'],
    install_requires=[
        "scipy==1.1.0",
        "numpy==1.18.2",
        "pandas==0.23.4",
        "scikit-learn==0.19.2",
        "tqdm==4.43.0",
    ]
)