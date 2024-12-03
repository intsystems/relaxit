import sys
from setuptools import setup

try:
    long_description = open("README.md", encoding="utf-8").read()
except Exception as e:
    sys.stderr.write("Failed to read README.md: {}\n".format(e))
    sys.stderr.flush()
    long_description = ""

setup(
    name="relaxit",
    version="1.0.0",
    license="MIT",
    author="Daniil Dorin, Igor Ignashin, Nikita Kiselev, Andrey Veprikov",
    author_email="research.n.math@gmail.com",
    description="A Python library for discrete variables relaxation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/intsystems/discrete-variables-relaxation",
    packages=["relaxit"],
    package_dir={"relaxit": "src"}, 
    install_requires=["pyro-ppl==1.9.1"],
)