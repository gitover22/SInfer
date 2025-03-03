import importlib.util
import logging
import io
import os
import sys
import subprocess
from typing import List
from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext

def load_module_from_path(module_name, path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

ROOT_DIR = os.path.dirname(__file__) 
logger = logging.getLogger(__name__)

envs = load_module_from_path('envs', os.path.join(ROOT_DIR, 'sinfer', 'envs.py'))

SINFER_TARGET_DEVICE = envs.SINFER_TARGET_DEVICE

assert sys.platform.startswith("linux"), "SInfer only supports Linux platform (including WSL)."

def get_path(*filepath) -> str:
    return os.path.join(ROOT_DIR, *filepath)

def find_version(filepath: str) -> str:
    with open(filepath, 'r') as file:
        lines = file.readlines()
    for line in lines:
        if 'SINFER_VERSION=' in line:
            version = line.split('=')[1].strip()
            return version

    raise RuntimeError("Unable to find version string.")

def get_commit_id() -> str:
    git_short_hash = subprocess.run(
        ['git', 'rev-parse', '--short', 'HEAD'],
        stdout=subprocess.PIPE,
        text=True
    ).stdout.strip()
    return git_short_hash

def get_sinfer_version() -> str:
    version = (
        find_version(get_path(os.path.dirname(os.path.abspath(__file__))
        + "/sinfer", "version"))+ "+mlu")
    if get_commit_id():
        version += "." + get_commit_id()
    return version

def read_readme() -> str:
    p = get_path("README.md")
    if os.path.isfile(p):
        return io.open(get_path("README.md"), "r", encoding="utf-8").read()
    else:
        return ""

def get_requirements() -> List[str]:
    def _read_requirements(filename: str) -> List[str]:
        with open(get_path(filename)) as f:
            requirements = f.read().strip().split("\n")
        resolved_requirements = []
        for line in requirements:
            if line.startswith("-r "):
                resolved_requirements += _read_requirements(line.split()[1])
            else:
                resolved_requirements.append(line)
        return resolved_requirements

    requirements = _read_requirements("requirements-mlu.txt")
    return requirements

package_data = {
    "sinfer": ["py.typed", "model_executor/layers/fused_moe/configs/*.json", "version_config"]
}
if envs.SINFER_USE_PRECOMPILED:
    ext_modules = []
    package_data["sinfer"].append("*.so")
    
class cmake_build_ext(build_ext):
    def run(self):
        if os.path.exists('device_info'):
            build_temp = os.path.join('device_info', 'build')
            if not os.path.exists(build_temp):
                os.makedirs(build_temp)
            cmake_cmd = [
                'cmake',
                '-S', 'device_info',
                '-B', build_temp,
                '-DCMAKE_INCLUDE_PATH=/usr/local/neuware/include',
                '-DCMAKE_LIBRARY_PATH=/usr/local/neuware/lib64'
            ]
            self.spawn(cmake_cmd)

            build_cmd = ['cmake', '--build', build_temp]
            self.spawn(build_cmd)

def _find_device_info_file() -> bool:
    if os.path.exists("./device_info"):
        return True
    return False

if _find_device_info_file():
    ext_modules=[
           Extension(
                'get_device_info',
                sources=['device_info/get_device_info.cpp'],)
            ]
    cmdclass={"build_ext": cmake_build_ext}
else:
    ext_modules = []
    cmdclass = {}

setup(
    name="sinfer",
    version=get_sinfer_version(),
    author="huafeng",
    license="Apache 2.0",
    description=("A smart inference framework for LLMs on MLU370"),
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="",
    project_urls={
        "Homepage": "https://github.com/gitover22/SInfer",
    },
    classifiers=[
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: Apache Software License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    packages=find_packages(exclude=("benchmarks", "examples", "tests*")),
    python_requires=">=3.9",
    install_requires=get_requirements(),
    ext_modules = ext_modules,
    extras_require={
        "tensorizer": ["tensorizer==2.9.0"],     
    },
    cmdclass=cmdclass,
    package_data=package_data,
)
