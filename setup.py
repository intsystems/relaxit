# from setuptools import setup, find_packages

# setup(
#     name='relaxit',
#     version='0.1',
#     packages=find_packages(),
# )


import io
import re
import os
from setuptools import setup, find_packages

version_info = {}
with open(os.path.join("src", "relaxit", "_version.py")) as f:
    exec(f.read(), version_info)


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
    version=version_info['__version__'],
    license="MIT",
    author="",
    author_email="",
    description="relaxit, python package",
    long_description=readme,
    url="https://github.com/intsystems/discrete-variables-relaxation",
    # options
    package_dir= {'' : 'src'} , 
    packages=find_packages(where= 'src'),
    install_requires=requirements,
)
