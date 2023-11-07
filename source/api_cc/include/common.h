#include <torch/script.h>
#ifndef COMMON_H
#define COMMON_H
#include <iostream>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <Eigen/Dense>

template <typename VALUETYPEIN, typename VALUETYPEOUT>
void TensortoVec(c10::IValue in, std::vector<VALUETYPEOUT>& out) {
  std::vector<VALUETYPEIN> values;
  if (in.isTensorList()) {
      auto tensor_list = in.toTensorList().vec();  // Extract the underlying vector
      for (const at::Tensor& tensor : tensor_list) {
          values.push_back(tensor.item<VALUETYPEIN>());
      }
  } else {
      std::cerr << "in is not a tensor list\n";
  }
  out.resize(values.size());
  for (int ii=0; ii<values.size(); ii++) {
      out[ii] = static_cast<VALUETYPEOUT>(values[ii]);
  }
}

template <typename VALUETYPE>
class Region3D {
 public:
  Region3D();
  ~Region3D() {}

  Region3D(const std::vector<VALUETYPE>& box){
    Eigen::Vector3d a1((std::vector<VALUETYPE>(box.begin(), box.begin()+3)).data());
    Eigen::Vector3d a2((std::vector<VALUETYPE>(box.begin()+3, box.begin()+6)).data());
    Eigen::Vector3d a3((std::vector<VALUETYPE>(box.begin()+6, box.begin()+9)).data());
    boxt << a1, a2, a3;
    rec_boxt = boxt.inverse();
    volume = boxt.determinant();
    c_xy = a1.cross(a2);
    _h2xy = volume / c_xy.norm();
    c_yz = a2.cross(a3);
    _h2yz = volume / c_yz.norm();
    c_zx = a3.cross(a1);
    _h2zx = volume / c_zx.norm();
}

  VALUETYPE volume;

  std::vector<VALUETYPE> get_face_distance(){
    std::vector<VALUETYPE> temp(3);
    temp[0] = _h2yz;
    temp[1] = _h2zx;
    temp[2] = _h2xy;
    return temp;
}

  void phys2inter(std::vector<VALUETYPE>& inner_coord, const std::vector<VALUETYPE>& coord){
    int natom = coord.size()/3;
    inner_coord.resize(natom*3);
    for (int ii=0; ii<natom; ii++) {
        Eigen::RowVector3d tmp_coord((std::vector<VALUETYPE>(coord.begin()+(3*ii), coord.begin()+(3*(ii+1)))).data());
        Eigen::RowVector3d tmp_inner_coord = tmp_coord * rec_boxt;
        for (int jj=0; jj<3; jj++) {
            inner_coord[3*ii+jj] = tmp_inner_coord[jj];
        }
    }
}

  void inter2phys(std::vector<VALUETYPE>& coord, const std::vector<VALUETYPE>& inner_coord){
    int natom = inner_coord.size()/3;
    coord.resize(natom*3);
    for (int ii=0; ii<natom; ii++) {
        Eigen::RowVector3d tmp_coord((std::vector<VALUETYPE>(inner_coord.begin()+(3*ii), inner_coord.begin()+(3*(ii+1)))).data());
        Eigen::RowVector3d tmp_inner_coord = tmp_coord * boxt;
        for (int jj=0; jj<3; jj++) {
            coord[3*ii+jj] = tmp_inner_coord[jj];
        }
    }
}

 private:
  Eigen::Matrix3d boxt;
  Eigen::Matrix3d rec_boxt;
  Eigen::Vector3d c_xy, c_yz, c_zx;
  VALUETYPE _h2xy, _h2yz, _h2zx;
};

template <typename VALUETYPE>
void normalize_coord(std::vector<VALUETYPE>& coord,
                     Region3D<VALUETYPE>& region){
    std::vector<VALUETYPE> inner_coord;
    region.phys2inter(inner_coord, coord);
    for (int ii=0; ii<inner_coord.size(); ii++) { 
        inner_coord[ii] = std::fmod(inner_coord[ii], 1.0);
        if (inner_coord[ii] < 0) inner_coord[ii] += 1;
    }
    region.inter2phys(coord, inner_coord);
}

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
                      VALUETYPE rcut){
    int natom = coord.size()/3;
    a2c.resize(natom);
    int nlocal_region = ncell[0]*ncell[1]*ncell[2];
    int nghost_region = (ncell[0]+2)*(ncell[1]+2)*(ncell[2]+2)-ncell[0]*ncell[1]*ncell[2];
    cell_list.resize(nlocal_region+nghost_region);

    std::vector<std::vector<int>> c2a(nlocal_region);
    build_inside_clist(a2c, c2a, coord, region, ncell);

    for (int ii=0; ii<nlocal_region; ii++) {
        cell_list[ii].assign(c2a[ii].begin(), c2a[ii].end());
    }

    std::vector<std::vector<int>> ghost_region_id(nghost_region);
    std::vector<std::vector<int>> mirror_region_id(nghost_region);
    // std::vector<int> ghost_region_serial_id(nghost_region);
    std::vector<int> mirror_region_serial_id(nghost_region);
    std::vector<std::vector<int>> ghost_region_shift_period_to_inside(nghost_region);
    int m = 0;
    for (int ii=-1; ii<ncell[0]+1; ii++) {
        for (int jj=-1; jj<ncell[1]+1; jj++) {
            for (int kk=-1; kk<ncell[2]+1; kk++) {
                if (ii<0 || ii>=ncell[0] || jj<0 || jj>=ncell[1] || kk<0 || kk>=ncell[2]) {
                    ghost_region_id[m].resize(3);
                    ghost_region_id[m][0] = ii;
                    ghost_region_id[m][1] = jj;
                    ghost_region_id[m][2] = kk;
                    region_id_map_serial[ghost_region_id[m]] = nlocal_region + m;
                    // ghost_region_serial_id[m] = nlocal_region + m;

                    mirror_region_id[m].resize(3);
                    mirror_region_id[m][0] = ii % ncell[0];
                    mirror_region_id[m][1] = jj % ncell[1];
                    mirror_region_id[m][2] = kk % ncell[2];

                    for (int ll = 0; ll < 3; ll++) {
                        if (mirror_region_id[m][ll] < 0) mirror_region_id[m][ll] += ncell[ll];
                    }
                    mirror_region_serial_id[m] = compute_serial_id(mirror_region_id[m], ncell);

                    ghost_region_shift_period_to_inside[m].resize(3);
                    ghost_region_shift_period_to_inside[m][0] = -int(std::floor(VALUETYPE(ii)/VALUETYPE(ncell[0])));
                    ghost_region_shift_period_to_inside[m][1] = -int(std::floor(VALUETYPE(jj)/VALUETYPE(ncell[1])));
                    ghost_region_shift_period_to_inside[m][2] = -int(std::floor(VALUETYPE(kk)/VALUETYPE(ncell[2])));
                    m++;
                }
                else {
                    std::vector<int> tmp_region_id = {ii, jj, kk};
                    region_id_map_serial[tmp_region_id] = compute_serial_id(tmp_region_id, ncell);
                }
            }
        }
    }
    // printf("ghost region serial id:\n");
    // for (int ii=0; ii<nghost_region; ii++) {
    //     printf("%d\n", ghost_region_serial_id[ii]);
    // }
    int nghost = 0;
    std::vector<int> nghost_each_region(nghost_region);
    for (int ii=0; ii<nghost_region; ii++) {
        nghost_each_region[ii] = c2a[mirror_region_serial_id[ii]].size();
        nghost += nghost_each_region[ii];
    }
    int nall = natom + nghost;
    std::vector<int> ghost_atom_mirror_region_id(nghost);
    std::vector<int> ghost_atom_ghost_region_idx(nghost); // Note: the elements in this vector range from [0, # ghost regions)
    m = 0;
    for (int ii=0; ii<nghost_region; ii++) {
        std::fill(ghost_atom_ghost_region_idx.begin()+m, ghost_atom_ghost_region_idx.begin()+(m+nghost_each_region[ii]), ii);
        m += nghost_each_region[ii];
    }
    
    std::vector<std::vector<VALUETYPE>> coord_shift(nghost);
    merged_coord_shift.resize(3*nall);
    merged_atype.resize(nall);
    merged_mapping.resize(nall);
    for (int ii=0; ii<nghost; ii++) {
        std::vector<int> inner_int = ghost_region_shift_period_to_inside[ghost_atom_ghost_region_idx[ii]];
        std::vector<VALUETYPE> inner_coord_shift(inner_int.begin(), inner_int.end());
        std::vector<VALUETYPE> shift(3); 
        region.inter2phys(shift, inner_coord_shift);
        std::copy(shift.begin(), shift.end(), merged_coord_shift.begin()+3*(natom+ii));
    }

    for (int ii=0; ii<natom; ii++) {
        merged_mapping[ii] = ii;
    }
    int last_size = 0;
    for (int ii=0; ii<nghost_region; ii++) {
        std::vector<int> tmp_mapping = c2a[mirror_region_serial_id[ii]];
        std::vector<int> tmp_id(tmp_mapping.size());
        for (int jj=0; jj<tmp_id.size(); jj++) {
            tmp_id[jj] = natom+last_size+jj;
        }
        cell_list[nlocal_region+ii].assign(tmp_id.begin(), tmp_id.end());
        std::copy(tmp_mapping.begin(), tmp_mapping.end(), merged_mapping.begin()+(natom+last_size));
        last_size += tmp_mapping.size();
    }

    for (int ii=0; ii<nall; ii++) {
        merged_atype[ii] = atype[merged_mapping[ii]];
    }
}

template <typename VALUETYPE>
void build_inside_clist(std::vector<int>& a2c,
                        std::vector<std::vector<int>>& c2a,
                        const std::vector<VALUETYPE>& coord,
                        Region3D<VALUETYPE>& region,
                        const std::vector<int>& ncell){
    std::vector<VALUETYPE> inter_cell_size(3);
    for (int ii=0; ii<3; ii++) inter_cell_size[ii] = 1./ncell[ii];
    int natom = coord.size()/3;
    std::vector<VALUETYPE> inner_coord;
    std::vector<VALUETYPE> cell_offset(natom*3);
    region.phys2inter(inner_coord, coord);
    for (int ii=0; ii<natom; ii++) {
        for (int jj=0; jj<3; jj++) {
            cell_offset[3*ii+jj] = std::floor(inner_coord[3*ii+jj] / inter_cell_size[jj]);
            if (cell_offset[3*ii+jj] < 0) cell_offset[3*ii+jj] = 0;
        }
    }
    for (int ii=0; ii<natom; ii++) {
        a2c[ii] = compute_serial_id(std::vector<int>(cell_offset.begin()+(3*ii), cell_offset.begin()+(3*ii+3)), ncell);
        c2a[a2c[ii]].push_back(ii);
    }
}

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
                const std::vector<int>& sec)
{
  Region3D<VALUETYPE> region(box);
  coord_wrapped = coord;
  normalize_coord(coord_wrapped, region);
  std::vector<int> merged_atype;
  std::vector<int> a2c;
  std::map<std::vector<int>, int> region_id_map_serial;
  std::vector<std::vector<int>> cell_list;

  std::vector<VALUETYPE> face_distance(3);
  face_distance = region.get_face_distance();
  std::vector<int> ncell(3);
  for (int ii=0; ii<3; ii++) { 
      ncell[ii] = std::floor(face_distance[ii]/rcut);
      if(ncell[ii] == 0) ncell[ii] = 1;
  }
  append_neighbors(merged_coord_shift, merged_atype, merged_mapping, a2c, region_id_map_serial, cell_list, ncell, coord_wrapped, atype, region, rcut);

//   for (std::map<std::vector<int>, int>::iterator iter=region_id_map_serial.begin(); iter!=region_id_map_serial.end(); iter++){
//     std::cout << iter->first << " " << iter->second << std::endl;
//   }
//   for (int ii=0; ii<64; ii++) {
//     std::cout << cell_list[ii] << std::endl;
//   }
  int nall = merged_mapping.size();
  std::vector<VALUETYPE> merged_coord(3*nall);
  for (int ii=0; ii<nall; ii++) {
    // std::vector<VALUETYPE> shifted_coord(3);
    int idx = merged_mapping[ii];
    for (int jj=0; jj<3; jj++) {
        // shifted_coord[jj] = coord[3*idx+jj] - merged_coord_shift[3*ii+jj];
        merged_coord[3*ii+jj] = coord_wrapped[3*idx+jj] - merged_coord_shift[3*ii+jj];
    }
  }
//   for (int ii=0; ii<nall; ii++) {
//     printf("%.4f %.4f %.4f\n", merged_coord[3*ii+0], merged_coord[3*ii+1], merged_coord[3*ii+2]);
//   }
  int nloc = atype.size();
  build_neighbor_list(nlist, nlist_loc, nlist_type, nloc, merged_coord, merged_atype, rcut, ncell, sec, merged_mapping, a2c, region_id_map_serial, cell_list);
//   for (int ii=0; ii<nlist.size(); ii++) {
//     std::cout << nlist[ii] << std::endl;
//   }
//   std::cout << std::endl;
//   for (int ii=0; ii<nlist.size(); ii++) {
//     std::cout << nlist_loc[ii] << std::endl;
//   }
//   std::cout << std::endl;
//   for (int ii=0; ii<nlist.size(); ii++) {
//     std::cout << nlist_type[ii] << std::endl;
//   }
}
struct InputNlist
{
  int inum;
  int * ilist;
  int * numneigh;
  int ** firstneigh;
  InputNlist () 
      : inum(0), ilist(NULL), numneigh(NULL), firstneigh(NULL)
      {};
  InputNlist (
      int inum_, 
      int * ilist_,
      int * numneigh_, 
      int ** firstneigh_
      ) 
      : inum(inum_), ilist(ilist_), numneigh(numneigh_), firstneigh(firstneigh_)
      {};
  ~InputNlist(){};
};

struct NeighborListData {
  /// Array stores the core region atom's index
  std::vector<int> ilist;
  /// Array stores the core region atom's neighbor index
  std::vector<std::vector<int>> jlist;
  /// Array stores the number of neighbors of core region atoms
  std::vector<int> numneigh;
  /// Array stores the the location of the first neighbor of core region atoms
  std::vector<int*> firstneigh;

 public:
  void copy_from_nlist(const InputNlist& inlist);
  void make_inlist(InputNlist& inlist, int& max_num_neighbors);
};
#endif