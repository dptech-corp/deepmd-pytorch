#move this file to $lammps_dir/build and run
cmake ../cmake/ -DPKG_DEEPMD=ON -DPKG_MOLECULE=ON -DDEEPMD_INCLUDE_PATH=~/Softwares/deepmd-pytorch/source/api_cc/include/ -DDEEPMD_LIB_PATH=~/Softwares/deepmd-pytorch/source/api_cc/
make -j10