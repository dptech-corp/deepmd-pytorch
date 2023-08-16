#include <torch/script.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#ifndef COMMON_H
#define COMMON_H
#include <iostream>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <Eigen/Dense>

template <typename VALUETYPE>
void make_env_mat(std::vector<std::vector<int>>& nlist,
                std::vector<std::vector<int>>& nlist_loc,
                std::vector<std::vector<int>>& nlist_type,
                std::vector<VALUETYPE>& merged_coord_shift,
                std::vector<int>& merged_mapping,
                std::vector<VALUETYPE>& coord_wrapped,
                const std::vector<VALUETYPE>& coord,
                const std::vector<int>& atype,
                const std::vector<VALUETYPE>& box,
                VALUETYPE rcut,
                const std::vector<int>& sec);

template <typename VALUETYPE>
class Region3D {
 public:
  Region3D();
  ~Region3D();
  Region3D(const std::vector<VALUETYPE>& box);
  VALUETYPE volume;
  std::vector<VALUETYPE> get_face_distance();
  void phys2inter(std::vector<VALUETYPE>& inner_coord, const std::vector<VALUETYPE>& coord);
  void inter2phys(std::vector<VALUETYPE>& coord, const std::vector<VALUETYPE>& inner_coord);
 private:
  Eigen::Matrix3d boxt;
  Eigen::Matrix3d rec_boxt;
  Eigen::Vector3d c_xy, c_yz, c_zx;
  VALUETYPE _h2xy, _h2yz, _h2zx;
};

template <typename VALUETYPE>
void normalize_coord(std::vector<VALUETYPE>& coord,
                     Region3D<VALUETYPE>& region);

int compute_serial_id(const std::vector<int>& cid, const std::vector<int>& ncell);

template <typename VALUETYPE>
void append_neighbors(std::vector<VALUETYPE>& merged_coord_shift,
                      std::vector<int>& merged_atype,
                      std::vector<int>& merged_mapping,
                      std::vector<int>& a2c,
                      std::map<std::vector<int>, int>& region_id_map_serial,
                      std::vector<std::vector<int>>& cell_list,
                      const std::vector<int>& ncell,
                      const std::vector<VALUETYPE>& coord,
                      const std::vector<int>& atype,
                      Region3D<VALUETYPE>& region,
                      VALUETYPE rcut);

template <typename VALUETYPE>
void build_inside_clist(std::vector<int>& a2c,
                        std::vector<std::vector<int>>& c2a,
                        const std::vector<VALUETYPE>& coord,
                        Region3D<VALUETYPE>& region,
                        const std::vector<int>& ncell);

template <typename VALUETYPE>
void build_neighbor_list(std::vector<std::vector<int>>& nlist,
                         std::vector<std::vector<int>>& nlist_loc,
                         std::vector<std::vector<int>>& nlist_atype,
                         int nloc,
                         const std::vector<VALUETYPE>& coord,
                         const std::vector<int>& atype,
                         VALUETYPE rcut,
                         const std::vector<int>& ncell,
                         const std::vector<int>& sec,
                         const std::vector<int>& merged_mapping,
                         const std::vector<int>& a2c,
                         const std::map<std::vector<int>, int>& region_id_map_serial, 
                         const std::vector<std::vector<int>>& cell_list,
                         bool type_split = true,
                         bool min_check = false)
{
  int nall = merged_mapping.size();
  nlist.resize(nloc);
  nlist_loc.resize(nloc);
  nlist_atype.resize(nloc);
  int ntype = sec.size();
  std::vector<int> sec_(ntype);
  std::vector<int> sec_offset(ntype);
  sec_[0] = sec[0];
  sec_offset[0] = 0;
  for (int i=1; i<ntype; i++) {
    sec_[i] = sec[i] - sec[i-1];
    sec_offset[i] = sec[i-1];
  }
  for (int i=0; i<nloc; i++) {
    nlist[i].assign(sec[ntype-1], -1);
    nlist_loc[i].assign(sec[ntype-1], -1);
    nlist_atype[i].assign(sec[ntype-1], -1);
    std::vector<int> region_id(3);
    int region_serial_id = a2c[i];
    region_id[0] = int(region_serial_id/(ncell[1]*ncell[2]));
    region_id[1] = int((region_serial_id - region_id[0]*ncell[1]*ncell[2])/ncell[2]);
    region_id[2] = region_serial_id - region_id[0]*ncell[1]*ncell[2] - region_id[1]*ncell[2];
    // printf("%d %d %d\n", region_id[0], region_id[1], region_id[2]);
    std::vector<int> neighbor_regions(27);
    int m = 0;
    for (int ii=-1; ii<2; ii++) {
      for (int jj=-1; jj<2; jj++) {
        for (int kk=-1; kk<2; kk++) {
          // if (! (ii==0 && jj==0 && kk==0)) {
            std::vector<int> neighbor_region_id = {region_id[0]+ii, region_id[1]+jj, region_id[2]+kk};
            int neighbor_region_serial_id = region_id_map_serial.find(neighbor_region_id)->second;
            neighbor_regions[m] = neighbor_region_serial_id;
            m++;
            // std::vector<int> neighbor_atoms = cell_list[neighbor_region_serial_id];
          // }
        }
      }
    }
    
    std::vector<int> neighbor_atoms;
    // std::vector<int> neighbor_atoms_atype;
    for (int ii=0; ii<27; ii++) {
      neighbor_atoms.insert(neighbor_atoms.end(), cell_list[neighbor_regions[ii]].begin(), cell_list[neighbor_regions[ii]].end());
    }
    auto iter = std::find(neighbor_atoms.begin(), neighbor_atoms.end(), i);
    neighbor_atoms.erase(iter);
    int num_neighbor_atoms = neighbor_atoms.size();
    std::vector<int> neighbor_atoms_in_rcut;
    std::vector<int> neighbor_atype_in_rcut;
    std::vector<VALUETYPE> neighbor_distances_in_rcut;

    int n=0;
    for (int j=0; j<num_neighbor_atoms; j++) {
      std::vector<VALUETYPE> coord_i(3);
      std::vector<VALUETYPE> coord_j(3);
      int idx_neighbor = neighbor_atoms[j];
      std::copy(coord.begin()+(3*i), coord.begin()+(3*i+3), coord_i.begin());
      std::copy(coord.begin()+(3*idx_neighbor), coord.begin()+(3*idx_neighbor+3), coord_j.begin());
      VALUETYPE tmp_distance = std::sqrt(std::pow(coord_j[0] - coord_i[0], 2) + std::pow(coord_j[1] - coord_i[1], 2) + std::pow(coord_j[2] - coord_i[2], 2));
      if (tmp_distance < rcut) {
        neighbor_atoms_in_rcut.push_back(idx_neighbor);
        neighbor_distances_in_rcut.push_back(tmp_distance);
        neighbor_atype_in_rcut.push_back(atype[merged_mapping[idx_neighbor]]);
        n++;
      }
      // neighbor_atoms_in_rcut.resize(n);
    }
    // std::cout << neighbor_atoms_in_rcut << std::endl;
    // std::cout << neighbor_distances_in_rcut << std::endl;
    // std::cout << neighbor_atype_in_rcut << std::endl;
    // std::cout << std::endl;

    // neighbor_atoms_atype.resize(num_neighbor_atoms);
    // for (int ii=0; ii<num_neighbor_atoms; ii++) {
    //   neighbor_atoms_atype[ii] = atype[merged_mapping[neighbor_atoms[ii]]];
    // }

    // std::vector<int> neighbor_atoms_each_type(ntype);
    std::vector<int> neighbor_atoms_onetype;
    std::vector<VALUETYPE> neighbor_distances_onetype;
    for (int ii=0; ii<ntype; ii++) {
      int number_atoms_onetype = std::count(neighbor_atype_in_rcut.begin(), neighbor_atype_in_rcut.end(), ii);
      // neighbor_atoms_each_type[ii] = std::count(neighbor_atype_in_rcut.begin(), neighbor_atype_in_rcut.end(), ii);
      neighbor_atoms_onetype.resize(number_atoms_onetype);
      neighbor_distances_onetype.resize(number_atoms_onetype);
      int k=0;
      for (int jj=0; jj<n; jj++) {
        if(neighbor_atype_in_rcut[jj] == ii) {
          neighbor_atoms_onetype[k] = neighbor_atoms_in_rcut[jj];
          neighbor_distances_onetype[k] = neighbor_distances_in_rcut[jj];
          k++;
        }
      }
      std::vector<size_t> indices(neighbor_atoms_onetype.size());
      std::iota(indices.begin(), indices.end(), 0);
      std::sort(indices.begin(), indices.end(), [&neighbor_distances_onetype](size_t i1, size_t i2) { return neighbor_distances_onetype[i1] < neighbor_distances_onetype[i2]; });

      // std::cout << indices << std::endl;

      // std::vector<int> sorted_atoms(neighbor_atoms_onetype.size());
      // std::vector<VALUETYPE> sorted_distances(neighbor_atoms_onetype.size());

      // printf("i = %d ii = %d indices.size: %ld\n", i, ii, indices.size());
      for (int jj=0; jj<indices.size(); jj++) {

          // sorted_atoms[jj] = neighbor_atoms_onetype[indices[jj]];
          // sorted_distances[jj] = neighbor_distances_onetype[indices[jj]];

          if (jj<sec_[ii]) {
            // if (i==14) { printf("ii = %d jj = %d sec_offset = %d i = %d %d %d %d\n", ii, jj, sec_offset[ii], i, sec_offset[ii]+jj, indices[jj], neighbor_atoms_onetype[indices[jj]]); }
            nlist[i][sec_offset[ii]+jj] = neighbor_atoms_onetype[indices[jj]];
            // printf("%d ", nlist[i][sec_offset[ii]+jj]);
            nlist_atype[i][sec_offset[ii]+jj] = ii;
            nlist_loc[i][sec_offset[ii]+jj] = merged_mapping[neighbor_atoms_onetype[indices[jj]]];
          }
      }
      // printf("\n");

      // std::cout << sorted_atoms << std::endl;
      // std::cout << sorted_distances << std::endl;
      // std::cout << std::endl;

      // std::cout << neighbor_atoms_onetype << std::endl;
      // std::cout << neighbor_distances_onetype << std::endl;
      // std::cout << std::endl;
    }
    // if(i==0) { std::cout << neighbor_atoms << std::endl; }
    // std::cout << neighbor_atoms_each_type << std::endl;
    
    // std::cout << std::count(neighbor_atoms_atype.begin(), neighbor_atoms_atype.end(), 0) << " " << std::count(neighbor_atoms_atype.begin(), neighbor_atoms_atype.end(), 1) << std::endl;
    // std::cout << neighbor_atoms.size() << std::endl;
  }
  
  // printf("iterated over all local atoms\n");
}

#endif