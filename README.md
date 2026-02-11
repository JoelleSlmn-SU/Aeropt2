# Aeropt2  
**Aerodynamic Shape Optimisation Framework**

Aeropt2 is a Python-based research framework for aerodynamic shape optimisation (ASO).  
It supports mesh-based geometry morphing, modal/RBF parameterisation, and gradient-free optimisation (Bayesian and evolutionary methods), with workflows designed for both local use and high-performance computing (HPC) environments.

The framework has primarily been developed and tested for supersonic intake optimisation, but is general enough to be applied to other external aerodynamic configurations.

---

## Repository philosophy

This repository contains the **optimisation and geometry-morphing framework only**.

It does **not** include:
- Geometry or mesh files  
- CFD solvers or meshers  
- Cluster-specific executables  

Aeropt2 assumes familiarity with:
- CFD workflows
- Mesh-based geometry manipulation
- HPC batch systems (e.g. Slurm)
- Python research codebases

---

## SET UP - LOCAL

1) Clone Aeropt2
 - Open a terminal
 - cd <where you keep projects>
 - git clone https://github.com/JoelleSlmn-SU/Aeropt2.git
 - cd Aeropt2

2) Create the Python environment
 - python -m pip install -r requirements.txt

3) Sanity check imports
 - python -c "import numpy, scipy, sklearn, matplotlib; print('core ok')"
 - python -c "import pyvista; print('pyvista ok')"
 - python -c "import PyQt5; print('pyqt ok')"

4) Set up a local example directory
 - "Test Case"

5) Run Aeropt.py


## SET UP - REMOTE

1) Create the expected directory structure on HPC
 - From the HPC home
 - mkdir -p ~/aeropt/Scripts
 - mkdir -p /scratch/$USER/aeropt
 - mkdir -p /scratch/$USER/aeropt/aeropt_out

2) Put Aeropt2 code on the HPC 
 - cd ~/aeropt/Scripts
 - git clone https://github.com/JoelleSlmn-SU/Aeropt2.git .

3) Install / activate Python on HPC
 - module load anaconda
 - conda env create -f ~/aeropt/Scripts/environment.yml
 - conda activate aeropt

4) Load necessary compiler/MPI modules
 - module load compiler/gnu/12/1.0
 - module load mpi/intel/2020/0

5) Confirm externam executables exist on HPC
 - ls -l /home/s.o.hassan/XieZ/work/Meshers/volume/src/a.Mesh3D
 - ls -l /home/s.o.hassan/bin/Gen3d_jj
 - ls -l /home/s.o.hassan/bin/UnsMgnsg3d


## Expected directory layout

Output directory will be generated on the HPC side upon generating it locally.

$HOME/
 ├── aeropt/
 │   └── Scripts/              # Aeropt2 repository (this repo)
 │       ├── MeshGeneration/
 │       ├── Optimisation/
 │       ├── FileRW/
 │       ├── Remote/
 │       ├── GUI/
 │       └── ...
 │
 └── scratch/
 │ ├── aeropt/
     └── aeropt_out/           
         ├── orig/             


