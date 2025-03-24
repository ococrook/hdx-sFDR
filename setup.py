from setuptools import setup, find_packages

setup(
    name="hdx_sFDR",
    version="0.1.0",
    packages=find_packages(),
    install_requires=required_packages,
    author="Oliver Crook",
    author_email="oliver.crook@chem.ox.ac.uk",
    description="Correcting p-values for peptide overlap in HDX-MS experiments",
    keywords="hdx, protein, structure",
    url="https://github.com/ococrook/hdx-sFDR",
)