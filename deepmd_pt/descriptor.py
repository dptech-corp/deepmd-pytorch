import logging
import numpy as np
import torch

from collections import namedtuple

from deepmd_pt import env


NeighborInfo = namedtuple('NeighborInfo', [
    'index',    # 编号
    'type',     # 元素
    'distance'  # 距离
])


class Region3D(object):

    def __init__(self, boxt):
        '''Construct a simulation box.'''
        logging.debug('Box: %s', boxt)
        self.boxt = boxt.reshape([3, 3])  # 用于世界坐标转内部坐标
        self.rec_boxt = np.linalg.inv(self.boxt)  # 用于内部坐标转世界坐标

        # 计算空间属性
        self.volume = np.linalg.det(self.boxt)  # 平行六面体空间的体积
        assert self.volume > 0, 'Negative volume of box is detected!'
        c_yz = np.cross(boxt[3:6], boxt[6:])
        self._h2yz = self.volume / np.linalg.norm(c_yz)
        c_zx = np.cross(boxt[6:], boxt[:3])
        self._h2zx = self.volume / np.linalg.norm(c_zx)
        c_xy = np.cross(boxt[:3], boxt[3:6])
        self._h2xy = self.volume / np.linalg.norm(c_xy)

    def phys2inter(self, coord):
        '''Convert physical coordinates to internal ones.'''
        assert coord.shape == (3,), 'Invalid atom coordinates!'
        return self.rec_boxt.dot(coord)

    def inter2phys(self, coord):
        '''Convert internal coordinates to physical ones.'''
        assert coord.shape == (3,), 'Invalid atom coordinates!'
        return self.boxt.dot(coord)

    def get_face_distance(self):
        '''Return face distinces to each surface of YZ, ZX, XY.'''
        return np.array([self._h2yz, self._h2zx, self._h2xy], dtype=env.GLOBAL_NP_FLOAT_PRECISION)


def normalize_coord(coord, region, nloc):
    '''Move outer atoms into region by mirror.

    Args:
    - coord: shape is [nloc*3]
    '''
    tmp_coord = coord.copy()
    for aid in range(nloc):  # 枚举原子
        offset = aid * 3
        a_coord = tmp_coord[offset:offset+3]
        logging.debug('Raw coords: %s', a_coord)
        inter_cood = region.phys2inter(a_coord) % 1.0
        logging.debug(' -> inter coords: %s', inter_cood)
        tmp_coord[offset:offset+3] = region.inter2phys(inter_cood)
        logging.debug(' -> phys coords: %s', tmp_coord[offset:offset+3])
    return tmp_coord


def compute_serial_cid(cell_offset, ncell):
    '''Tell the sequential cell ID in its 3D space.

    Args:
    - cell_offset: shape is [3]
    - ncell: shape is [3]
    '''
    return (cell_offset[0]*ncell[1] + cell_offset[1])*ncell[2] + cell_offset[2]


def compute_pbc_shift(cell_offset, ncell):
    '''Tell shift count to move the atom into region.'''
    shift = 0
    if cell_offset < 0:
        shift = 1
        assert cell_offset + ncell > 0
    elif cell_offset >= ncell:
        shift = -1
        assert cell_offset - ncell < ncell
    return shift


def build_inside_clist(coord, region, ncell):
    '''Build cell list on atoms inside region.

    Args:
    - coord: shape is [nloc*3]
    - ncell: shape is [3]
    '''
    loc_ncell = np.prod(ncell)  # 模拟区域内的 Cell 数量
    nloc = coord.size // 3  # 原子数量
    inter_cell_size = 1. / ncell
    logging.debug('Cell size of internal coords:', inter_cell_size)
    clist = [[] for _ in range(loc_ncell)]  # 模拟区域内的 Cell 列表
    for aid in range(nloc):  # 枚举原子
        a_coord = coord[aid*3:aid*3+3]
        inter_cood = region.phys2inter(a_coord)
        cell_offset = np.floor(inter_cood / inter_cell_size).astype(np.int32)
        assert not np.any(cell_offset[cell_offset < 0]), 'No outside cell should be used!'
        delta = cell_offset - ncell
        assert not np.any(delta[delta >= 0]), 'No outside cell should be used!'
        cid = compute_serial_cid(cell_offset, ncell)
        clist[cid].append(aid)
    return clist


def append_neighbors(coord, region, atype, rcut):
    '''Make ghost atoms who are valid neighbors.

    Args:
    - coord: shape is [nloc*3]
    - atype: shape is [nloc]
    '''
    to_face = region.get_face_distance()

    # 计算 3 个方向的 Cell 大小和 Cell 数量
    ncell = np.floor(to_face/rcut).astype(np.int32)
    ncell[ncell == 0] = 1  # 模拟区域内的 Cell 数量
    logging.debug('Cell count:', ncell)
    cell_size = to_face / ncell
    ngcell = np.floor(rcut / cell_size).astype(np.int32) + 1  # 模拟区域外的 Cell 数量，存储的是 Ghost 原子
    logging.debug('Outer-box cell count:', ngcell)
    expanded = cell_size * ngcell
    assert not np.any(expanded[expanded < rcut]), 'Cell sizes calculated by `rcut` and `box` is invalid!'

    # 借助 Cell 列表添加边界外的 Ghost 原子
    clist = build_inside_clist(coord, region, ncell)
    tmp_coord = []
    tmp_atype = []
    tmp_mapping = []
    for xi in range(-ngcell[0], ncell[0]+ngcell[0]):
        x_shift = compute_pbc_shift(xi, ncell[0])
        for yi in range(-ngcell[1], ncell[1]+ngcell[1]):
            y_shift = compute_pbc_shift(yi, ncell[1])
            for zi in range(-ngcell[2], ncell[2]+ngcell[2]):
                z_shift = compute_pbc_shift(zi, ncell[2])
                if (xi >= 0 and xi < ncell[0]) and (yi >= 0 and yi < ncell[1]) and (zi >= 0 and zi < ncell[2]):
                    continue  # 无需对内部原子重复处理
                pbc_shift = np.array([x_shift, y_shift, z_shift], dtype=np.int32)
                coord_shift = region.inter2phys(pbc_shift)
                mirrored = pbc_shift*ncell + [xi, yi, zi]
                cid = compute_serial_cid(mirrored, ncell)
                for aid in clist[cid]:
                    a_coord = coord[aid*3:aid*3+3]
                    tmp_coord.append(a_coord - coord_shift)
                    tmp_atype.append(atype[aid])
                    tmp_mapping.append(aid)
    logging.debug('%d atoms are appended as ghost', len(tmp_atype))

    # 合并内部原子和 Ghost 原子信息
    merged_coord = np.concatenate([coord.reshape([-1, 3]), tmp_coord]).reshape([-1])
    merged_atype = np.concatenate([atype, tmp_atype])
    merged_mapping = np.concatenate([np.arange(atype.size), tmp_mapping])
    return merged_coord, merged_atype, merged_mapping


def build_neighbor_list(nloc, coord, atype, rcut):
    '''For each atom inside region, build its neighbor list.

    Args:
    - coord: shape is [nloc*3]
    - atype: shape is [nloc]
    '''
    nall = coord.size // 3
    nlist = [[] for _ in range(nloc)]
    for aid in range(nloc):
        a_coord = coord[aid*3:aid*3+3]
        for nid in range(nall):
            if aid == nid:
                continue
            n_coord = coord[nid*3:nid*3+3]
            distance = np.linalg.norm(n_coord - a_coord)
            if distance < rcut:
                ni = NeighborInfo(nid, atype[nid], distance)
                nlist[aid].append(ni)
    return nlist


def select_neighbors(nlist, sec):
    '''Select nearest neighbors of each element.'''
    nloc = len(nlist)
    nnei = sec[-1]  # 最大的邻居数量
    selected = np.full([nloc, nnei], -1, dtype=np.int32)
    for aid in range(nloc):
        nlist[aid].sort(key=lambda item: (item.type, item.distance, item.index))
        type_accum = np.zeros(sec.shape, dtype=np.int32)
        type_accum[1:] = sec[:-1]
        for ni in nlist[aid]:
            if type_accum[ni.type] < sec[ni.type]:  # 卡邻居中某种元素的数量
                selected[aid, type_accum[ni.type]] = ni.index
                type_accum[ni.type] += 1
            else:
                # logging.warning('Count of neighbor element %d is overflowed!' % ni.type)
                pass
    return selected


def compute_smooth_weight(distance, rmin, rmax):
    '''Compute smooth weight for descriptor elements.'''
    mask = torch.logical_and(distance > rmin, distance < rmax)
    uu = (distance - rmin) / (rmax - rmin)
    vv = uu*uu*uu * (-6 * uu*uu + 15*uu - 10) + 1
    return vv * mask


def make_env_mat(coord, atype,  # 原子坐标和相应类型
                 box,           # 模拟盒子
                 rcut,          # 截断半径
                 sec):          # 邻居中某元素的最大数量
    '''Based on atom coordinates, return environment matrix.

    Args:
    - coord: shape is [nloc*3]
    - atype: shape is [nloc]
    - box: shape is [9]
    - sec: shape is [max(atype)+1]
    '''
    region = Region3D(box)
    nloc = atype.shape[0]

    # 将盒子外的原子，通过镜像挪入盒子内
    tmp_coord = normalize_coord(coord, region, nloc)
    merged_coord, merged_atype, merged_mapping = append_neighbors(tmp_coord, region, atype, rcut)
    assert merged_coord.shape[0] > tmp_coord.shape[0], 'No ghost atom is added!'

    # 构建邻居列表，并按 sel_a 筛选
    nlist = build_neighbor_list(nloc, merged_coord, merged_atype, rcut)
    selected = select_neighbors(nlist, sec)
    return selected, merged_coord, merged_mapping


def make_se_a_mat(selected, coord, rcut, ruct_smth):
    '''Based on environment matrix, build descriptor of type `se_a`.'''
    coord = coord.view(-1, 3)
    selected = torch.tensor(selected, device=coord.device, dtype=torch.long)
    mask = selected>=0
    selected = selected * mask
    coord_l = coord[range(selected.shape[0])].view(-1, 1, 3)
    coord_r = torch.index_select(coord, 0, selected.view(-1))
    coord_r = coord_r.view(selected.shape+(3,))
    diff = coord_l - coord_r
    length = torch.linalg.norm(diff, dim=-1, keepdim=True)
    t0 = 1/length
    t1 = diff/length**2
    weight = compute_smooth_weight(length, ruct_smth, rcut)
    descriptor = torch.cat([t0, t1], dim=-1) *weight * mask.unsqueeze(-1)
    return descriptor

class SmoothDescriptor(torch.autograd.Function):
    '''Function wrapper for `se_a` descriptor.'''

    @staticmethod
    def forward(ctx,
        coord, atype, natoms, box,  # 动态的 torch.Tensor 或 numpy.ndarray
        mean, stddev, deriv_stddev, # 静态的 numpy.ndarray
        rcut, rcut_smth, sec  # 静态的 Python 对象
    ):
        '''Generate descriptor matrix from atom coordinates and other context.

        Args:
        - coord: Batched atom coordinates with shape [nframes, natoms[1]*3].
        - atype: Batched atom types with shape [nframes, natoms[1]].
        - natoms: Batched atom statisics with shape [len(sec)+2].
        - box: Batched simulation box with shape [nframes, 9].
        - mean: Average value of descriptor per element type with shape [len(sec), nnei, 4].
        - stddev: Standard deviation of descriptor per element type with shape [len(sec), nnei, 4].
        - deriv_stddev:  StdDev of descriptor derivative per element type with shape [len(sec), nnei, 4, 3].
        - rcut: Cut-off radius.
        - rcut_smth: Smooth hyper-parameter for pair force & energy.
        - sec: Cumulative count of neighbors by element.

        Returns:
        - descriptor: Shape is [nframes, natoms[1]*nnei*4].
        '''
        nnei = sec[-1]  # 总的邻居数量
        nframes = coord.shape[0]  # 样本数量
        nloc, nall = natoms[0], natoms[1]  # 原子数量和包含 Ghost 原子的数量
        assert nloc == nall, 'In PBC, `nloc` === `nall`!'
        assert nframes == atype.shape[0], 'Batch size differs!'
        assert nframes == box.shape[0], 'Batch size differs!'
        assert nall*3 == coord.shape[1], 'Atom count differs!'
        assert nall == atype.shape[1], 'Atom count differs!'
        assert 9 == box.shape[1], 'Box size is invalid!'
        assert len(sec) == natoms.shape[0] - 2, 'Element type mismatches!'

        coord_np = coord.detach().cpu().numpy()
        descriptor_list = []
        deriv_list = []
        nlist_list = []
        for sid in range(nframes):  # 枚举样本
            logging.debug('[FW] In-batch ID: %d', sid)
            selected, merged_coord, merged_mapping = make_env_mat(coord_np[sid], atype[sid], box[sid], rcut, sec)
            merged_coord = torch.tensor(merged_coord, device=coord.device)
            se_a = make_se_a_mat(selected, merged_coord, rcut, rcut_smth)
            for aid in range(nloc):
                a_type = atype[sid, aid]
                t_avg = mean[a_type]
                t_std = stddev[a_type]
                se_a[aid] = (se_a[aid] - t_avg) / t_std
            descriptor_list.append(se_a.reshape([-1]))

            nlist = np.full([nloc, nnei], -1, dtype=np.int32)
            for aid in range(nloc):
                neighbors = selected[aid]
                for idx, nid in enumerate(neighbors):
                    if nid >= 0:
                        nlist[aid,idx] = merged_mapping[nid]
            nlist_list.append(nlist.reshape([-1]))
        descriptor_list = np.stack(descriptor_list)
        nlist_list = np.stack(nlist_list)
        return torch.from_numpy(descriptor_list)

__all__ = ['SmoothDescriptor']
