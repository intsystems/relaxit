import io
import os
import sys
from setuptools import setup

version_info = {}
with open(os.path.join("src", "relaxit", "_version.py")) as f:
    exec(f.read(), version_info)

def read(file_path):
    with io.open(file_path, "r", encoding="utf-8") as f:
        return f.read()

try:
    long_description = open("README.md", encoding="utf-8").read()
except Exception as e:
    sys.stderr.write("Failed to read README.md: {}\n".format(e))
    sys.stderr.flush()
    long_description = ""

setup(
    name="relaxit",
    version=version_info["__version__"],
    license="MIT",
    author="Daniil Dorin, Igor Ignashin, Nikita Kiselev, Andrey Veprikov",
    author_email="research.n.math@gmail.com",
    description="A Python library for discrete variables relaxation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/intsystems/discrete-variables-relaxation",
    packages=["relaxit"],
    package_dir={"relaxit": "src/relaxit"}, 
    install_requires=["pyro-ppl==1.9.1"],
)