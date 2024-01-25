// SPDX-License-Identifier: LGPL-3.0-or-later
#include "common.h"

template void make_env_mat<double>(std::vector<std::vector<int>> &nlist,
                                   std::vector<std::vector<int>> &nlist_loc,
                                   std::vector<std::vector<int>> &nlist_type,
                                   std::vector<double> &merged_coord_shift,
                                   std::vector<int> &merged_mapping,
                                   std::vector<double> &coord_wrapped,
                                   const std::vector<double> &coord,
                                   const std::vector<int> &atype,
                                   const std::vector<double> &box, double rcut,
                                   const std::vector<int> &sec);

int compute_serial_id(const std::vector<int> &cid,
                      const std::vector<int> &ncell) {
  return cid[0] * ncell[1] * ncell[2] + cid[1] * ncell[2] + cid[2];
}

template void normalize_coord<double>(std::vector<double> &coord,
                                      Region3D<double> &region);

template void append_neighbors<double>(
    std::vector<double> &merged_coord_shift, std::vector<int> &merged_atype,
    std::vector<int> &merged_mapping, std::vector<int> &a2c,
    std::map<std::vector<int>, int> &region_id_map_serial,
    std::vector<std::vector<int>> &cell_list, const std::vector<int> &ncell,
    const std::vector<double> &coord, const std::vector<int> &atype,
    Region3D<double> &region, double rcut);

template void build_inside_clist<double>(std::vector<int> &a2c,
                                         std::vector<std::vector<int>> &c2a,
                                         const std::vector<double> &coord,
                                         Region3D<double> &region,
                                         const std::vector<int> &ncell);

template void build_neighbor_list<double>(
    std::vector<std::vector<int>> &nlist,
    std::vector<std::vector<int>> &nlist_loc,
    std::vector<std::vector<int>> &nlist_atype, int nloc,
    const std::vector<double> &coord, const std::vector<int> &atype,
    double rcut, const std::vector<int> &ncell, const std::vector<int> &sec,
    const std::vector<int> &merged_mapping, const std::vector<int> &a2c,
    const std::map<std::vector<int>, int> &region_id_map_serial,
    const std::vector<std::vector<int>> &cell_list, bool type_split = true,
    bool min_check = false);
