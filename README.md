# Documentation

Design of insulative material with 1D heat transfer model

## Installation

Install required softwares

```
conda create -y --name dymos-env python=3.8
conda activate dymos-env
conda config --add channels conda-forge
conda install compilers matplotlib mpi4py petsc4py swig
python -m pip install dymos
```

Install pyoptsparse

```
cd build_pyoptsparse_master
chmod +x build_pyoptsparse.sh
./build_pyoptsparse.sh
```
