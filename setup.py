# from setuptools import setup, find_packages

# setup(
#     name='relaxit',
#     version='0.1',
#     packages=find_packages(),
# )


import io
import re
from setuptools import setup, find_packages

from src.relaxit import __version__


def read(file_path):
    with io.open(file_path, "r", encoding="utf-8") as f:
        return f.read()


readme = read("README.rst")
# # вычищаем локальные версии из файла requirements (согласно PEP440)
requirements = '\n'.join(
    re.findall(r'^([^\s^+]+).*$',
               read('requirements.txt'),
               flags=re.MULTILINE))


setup(
    # metadata
    name="relaxit",
    version=__version__,
    license="MIT",
    author="",
    author_email="",
    description="relaxit, python package",
    long_description=readme,
    url="https://github.com/intsystems/discrete-variables-relaxation",
    # options
    package_dir= {'' : 'src'} , 
    packages=find_packages(where= 'srd'),
    install_requires=requirements,
)
