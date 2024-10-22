import logging
import numpy as np
import torch
from deepmd_pt.utils import env
from typing import Union


class Region3D(object):

    def __init__(self, boxt):
        """Construct a simulation box."""
        boxt = boxt.reshape([3, 3])
        self.boxt = boxt  # 用于世界坐标转内部坐标
        self.rec_boxt = torch.linalg.inv(self.boxt)  # 用于内部坐标转世界坐标

        # 计算空间属性
        self.volume = torch.linalg.det(self.boxt)  # 平行六面体空间的体积

        boxt = boxt.permute(1, 0)
        c_yz = torch.cross(boxt[1], boxt[2])
        self._h2yz = self.volume / torch.linalg.norm(c_yz)
        c_zx = torch.cross(boxt[2], boxt[0])
        self._h2zx = self.volume / torch.linalg.norm(c_zx)
        c_xy = torch.cross(boxt[0], boxt[1])
        self._h2xy = self.volume / torch.linalg.norm(c_xy)

    def phys2inter(self, coord):
        """Convert physical coordinates to internal ones."""
        return coord @ self.rec_boxt

    def inter2phys(self, coord):
        """Convert internal coordinates to physical ones."""
        return coord @ self.boxt

    def get_face_distance(self):
        """Return face distinces to each surface of YZ, ZX, XY."""
        return torch.stack([self._h2yz, self._h2zx, self._h2xy])


def normalize_coord(coord, region: Region3D, nloc: int):
    """Move outer atoms into region by mirror.

    Args:
    - coord: shape is [nloc*3]
    """
    tmp_coord = coord.clone()
    inter_cood = torch.remainder(region.phys2inter(tmp_coord), 1.0)
    tmp_coord = region.inter2phys(inter_cood)
    return tmp_coord


def compute_serial_cid(cell_offset, ncell):
    """Tell the sequential cell ID in its 3D space.

    Args:
    - cell_offset: shape is [3]
    - ncell: shape is [3]
    """

    cell_offset[:, 0] *= ncell[1] * ncell[2]
    cell_offset[:, 1] *= ncell[2]
    return cell_offset.sum(-1)


def compute_pbc_shift(cell_offset, ncell):
    """Tell shift count to move the atom into region."""
    shift = torch.zeros_like(cell_offset)
    shift = shift + (cell_offset < 0) * -(torch.div(cell_offset, ncell, rounding_mode='floor'))
    shift = shift + (cell_offset >= ncell) * -(torch.div((cell_offset - ncell), ncell, rounding_mode='floor') + 1)
    assert torch.all(cell_offset + shift * ncell >= 0)
    assert torch.all(cell_offset + shift * ncell < ncell)
    return shift


def build_inside_clist(coord, region: Region3D, ncell):
    """Build cell list on atoms inside region.

    Args:
    - coord: shape is [nloc*3]
    - ncell: shape is [3]
    """
    loc_ncell = int(torch.prod(ncell))  # 模拟区域内的 Cell 数量
    nloc = coord.numel() // 3  # 原子数量
    inter_cell_size = 1. / ncell

    inter_cood = region.phys2inter(coord.view(-1, 3))
    cell_offset = torch.floor(inter_cood / inter_cell_size).to(torch.long)
    # numerical error brought by conversion from phys to inter back and force
    # may lead to negative value
    cell_offset[cell_offset < 0] = 0
    delta = cell_offset - ncell
    a2c = compute_serial_cid(cell_offset, ncell)  # cell id of atoms
    arange = torch.arange(0, loc_ncell, 1, device=env.PREPROCESS_DEVICE)
    cellid = (a2c == arange.unsqueeze(-1))  # one hot cellid
    c2a = cellid.nonzero()
    lst = []
    cnt = 0
    bincount = torch.bincount(a2c, minlength=loc_ncell)
    for i in range(loc_ncell):
        n = bincount[i]
        lst.append(c2a[cnt: cnt + n, 1])
        cnt += n
    return a2c, lst


def append_neighbors(coord, region: Region3D, atype, rcut: float):
    """Make ghost atoms who are valid neighbors.

    Args:
    - coord: shape is [nloc*3]
    - atype: shape is [nloc]
    """
    to_face = region.get_face_distance()

    # 计算 3 个方向的 Cell 大小和 Cell 数量
    ncell = torch.floor(to_face / rcut).to(torch.long)
    ncell[ncell == 0] = 1  # 模拟区域内的 Cell 数量
    cell_size = to_face / ncell
    ngcell = torch.floor(rcut / cell_size).to(torch.long) + 1  # 模拟区域外的 Cell 数量，存储的是 Ghost 原子

    # 借助 Cell 列表添加边界外的 Ghost 原子
    a2c, c2a = build_inside_clist(coord, region, ncell)
    xi = torch.arange(-ngcell[0], ncell[0] + ngcell[0], 1, device=env.PREPROCESS_DEVICE)
    yi = torch.arange(-ngcell[1], ncell[1] + ngcell[1], 1, device=env.PREPROCESS_DEVICE)
    zi = torch.arange(-ngcell[2], ncell[2] + ngcell[2], 1, device=env.PREPROCESS_DEVICE)
    xyz = xi.view(-1, 1, 1, 1) * torch.tensor([1, 0, 0], dtype=torch.long, device=env.PREPROCESS_DEVICE)
    xyz = xyz + yi.view(1, -1, 1, 1) * torch.tensor([0, 1, 0], dtype=torch.long, device=env.PREPROCESS_DEVICE)
    xyz = xyz + zi.view(1, 1, -1, 1) * torch.tensor([0, 0, 1], dtype=torch.long, device=env.PREPROCESS_DEVICE)
    xyz = xyz.view(-1, 3)
    mask_a = (xyz >= 0).all(dim=-1)
    mask_b = (xyz < ncell).all(dim=-1)
    mask = ~torch.logical_and(mask_a, mask_b)
    xyz = xyz[mask]  # cell coord
    shift = compute_pbc_shift(xyz, ncell)
    coord_shift = region.inter2phys(shift.to(env.GLOBAL_PT_FLOAT_PRECISION))
    mirrored = shift * ncell + xyz
    cid = compute_serial_cid(mirrored, ncell)

    n_atoms = coord.shape[0]
    aid = [c2a[ci] + i * n_atoms for i, ci in enumerate(cid)]
    aid = torch.cat(aid)
    tmp = torch.div(aid, n_atoms, rounding_mode='trunc')
    aid = aid % n_atoms
    tmp_coord = coord[aid] - coord_shift[tmp]
    tmp_atype = atype[aid]

    # 合并内部原子和 Ghost 原子信息
    merged_coord = torch.cat([coord, tmp_coord])
    merged_coord_shift = torch.cat([torch.zeros_like(coord), coord_shift[tmp]])
    merged_atype = torch.cat([atype, tmp_atype])
    merged_mapping = torch.cat([torch.arange(atype.numel(), device=env.PREPROCESS_DEVICE), aid])
    return merged_coord_shift, merged_atype, merged_mapping


def build_neighbor_list(nloc: int, coord, atype, rcut: float, sec, mapping, type_split=True, min_check=False):
    """For each atom inside region, build its neighbor list.

    Args:
    - coord: shape is [nall*3]
    - atype: shape is [nall]
    """
    nall = coord.numel() // 3
    coord = coord.float()
    nlist = [[] for _ in range(nloc)]
    coord_l = coord.view(-1, 1, 3)[:nloc]
    coord_r = coord.view(1, -1, 3)
    distance = coord_l - coord_r
    distance = torch.linalg.norm(distance, dim=-1)
    DISTANCE_INF = distance.max().detach() + rcut
    distance[:nloc, :nloc] += torch.eye(nloc, dtype=torch.bool, device=env.PREPROCESS_DEVICE) * DISTANCE_INF
    if min_check:
        if distance.min().abs() < 1e-6:
            RuntimeError("Atom dist too close!")
    if not type_split:
        sec = sec[-1:]
    lst = []
    selected = torch.zeros((nloc, sec[-1].item()), device=env.PREPROCESS_DEVICE).long() - 1
    selected_loc = torch.zeros((nloc, sec[-1].item()), device=env.PREPROCESS_DEVICE).long() - 1
    selected_type = torch.zeros((nloc, sec[-1].item()), device=env.PREPROCESS_DEVICE).long() - 1
    for i, nnei in enumerate(sec):
        if i > 0:
            nnei = nnei - sec[i - 1]
        if not type_split:
            tmp = distance
        else:
            mask = atype.unsqueeze(0) == i
            tmp = distance + (~mask) * DISTANCE_INF
        if tmp.shape[1] >= nnei:
            _sorted, indices = torch.topk(tmp, nnei, dim=1, largest=False)
        else:
            # when nnei > nall
            indices = torch.zeros((nloc, nnei), device=env.PREPROCESS_DEVICE).long() - 1
            _sorted = torch.ones((nloc, nnei), device=env.PREPROCESS_DEVICE).long() * DISTANCE_INF
            _sorted_nnei, indices_nnei = torch.topk(tmp, tmp.shape[1], dim=1, largest=False)
            _sorted[:, :tmp.shape[1]] = _sorted_nnei
            indices[:, :tmp.shape[1]] = indices_nnei
        mask = (_sorted < rcut).to(torch.long)
        indices_loc = mapping[indices]
        indices = indices * mask + -1 * (1 - mask)  # -1 for padding
        indices_loc = indices_loc * mask + -1 * (1 - mask)  # -1 for padding
        if i == 0:
            start = 0
        else:
            start = sec[i - 1]
        end = min(sec[i], start + indices.shape[1])
        selected[:, start:end] = indices[:, :nnei]
        selected_loc[:, start:end] = indices_loc[:, :nnei]
        selected_type[:, start:end] = atype[indices[:, :nnei]] * mask + -1 * (1 - mask)
    return selected, selected_loc, selected_type


def compute_smooth_weight(distance, rmin: float, rmax: float):
    """Compute smooth weight for descriptor elements."""
    min_mask = distance <= rmin
    max_mask = distance >= rmax
    mid_mask = torch.logical_not(torch.logical_or(min_mask, max_mask))
    uu = (distance - rmin) / (rmax - rmin)
    vv = uu * uu * uu * (-6 * uu * uu + 15 * uu - 10) + 1
    return vv * mid_mask + min_mask


def make_env_mat(coord,
                 atype,
                 region,
                 rcut: Union[float, list],
                 sec,
                 pbc=True,
                 type_split=True,
                 min_check=False):
    """Based on atom coordinates, return environment matrix.

    Returns
        selected: nlist, [nloc, nnei]
        merged_coord_shift: shift on nall atoms, [nall, 3]
        merged_mapping: mapping from nall index to nloc index, [nall]
    """
    # 将盒子外的原子，通过镜像挪入盒子内
    hybrid = isinstance(rcut, list)
    _rcut = rcut
    if hybrid:
        _rcut = max(rcut)
    if pbc:
        merged_coord_shift, merged_atype, merged_mapping = append_neighbors(coord, region, atype, _rcut)
        merged_coord = coord[merged_mapping] - merged_coord_shift
        if merged_coord.shape[0] <= coord.shape[0]:
            logging.warning('No ghost atom is added for system ')
    else:
        merged_coord_shift = torch.zeros_like(coord)
        merged_atype = atype.clone()
        merged_mapping = torch.arange(atype.numel(), device=env.PREPROCESS_DEVICE)
        merged_coord = coord.clone()

    # 构建邻居列表，并按 sel_a 筛选
    if not hybrid:
        selected, selected_loc, selected_type = build_neighbor_list(coord.shape[0], merged_coord, merged_atype, rcut, sec,
                                                                    merged_mapping, type_split=type_split, min_check=min_check)
    else:
        selected, selected_loc, selected_type = [], [], []
        for ii, single_rcut in enumerate(rcut):
            selected_tmp, selected_loc_tmp, selected_type_tmp = build_neighbor_list(coord.shape[0], merged_coord, merged_atype,
                                                                        single_rcut, sec[ii],
                                                                        merged_mapping, type_split=type_split,
                                                                        min_check=min_check)
            selected.append(selected_tmp)
            selected_loc.append(selected_loc_tmp)
            selected_type.append(selected_type_tmp)
    return selected, selected_loc, selected_type, merged_coord_shift, merged_mapping
