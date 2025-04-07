from setuptools import setup, find_packages

# Try to read requirements.txt
try:
    with open('requirements.txt') as f:
        required_packages = f.read().splitlines()
except FileNotFoundError:
    print("Warning: requirements.txt not found")
except Exception as e:
    print(f"Warning: Could not read requirements.txt: {e}")

setup(
    name="hdx_sFDR",
    version="0.1.10",
    packages=find_packages(),
    install_requires=required_packages,
    author="Oliver Crook",
    author_email="oliver.crook@chem.ox.ac.uk",
    description="Correcting p-values for peptide overlap in HDX-MS experiments",
    keywords="hdx, protein, structure",
    url="https://github.com/ococrook/hdx-sFDR",
)