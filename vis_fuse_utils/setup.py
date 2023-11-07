# From: https://github.com/pybind/cmake_example/blob/master/setup.py

import os
import subprocess
import sys
from pathlib import Path

from setuptools import Extension, setup, find_packages
from setuptools.command.build_ext import build_ext


class CMakeExtension(Extension):

    def __init__(self, name: str, sourcedir: str = "") -> None:
        super().__init__(name, sources=[])
        self.sourcedir = os.fspath(Path(sourcedir).resolve())


class CMakeBuild(build_ext):

    def build_extension(self, ext: CMakeExtension) -> None:
        # Must be in this form due to bug in .resolve() only fixed in Python 3.10+
        # type: ignore[no-untyped-call]
        ext_fullpath = Path.cwd() / self.get_ext_fullpath(ext.name)
        extdir = ext_fullpath.parent.resolve()

        cfg = "Release"

        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}{os.sep}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-DCMAKE_BUILD_TYPE={cfg}",    # not used on MSVC, but no harm
        ]
        build_args = [f"-j4"]

        build_temp = Path(self.build_temp) / ext.name
        if not build_temp.exists():
            build_temp.mkdir(parents=True)

        subprocess.run(["cmake", ext.sourcedir] + cmake_args,
                       cwd=build_temp,
                       check=True)
        subprocess.run(["cmake", "--build", "."] + build_args,
                       cwd=build_temp,
                       check=True)


setup(
    name="vis_fuse_utils",
    version="0.0.1",
    author="Ruichen Zheng",
    author_email="ankbzpx@hotmail.com",
    description=
    "Some utility functions. Closest neighbor cuda implementation modified from https://github.com/vincentfpgarcia/kNN-CUDA",
    package_dir={'': 'src-py'},
    packages=find_packages(where="src-py"),
    ext_modules=[CMakeExtension("vis_fuse_utils_bind")],
    cmdclass={"build_ext": CMakeBuild},
    zip_safe=False,
    extras_require={"test": ["pytest>=6.0"]},
    python_requires=">=3.7",
)
