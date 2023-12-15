import numpy as np
import os
import torch
import unittest
import json
from typing import Optional
from pathlib import Path

from deepmd_pt.model.descriptor import DescrptHybrid
from deepmd_pt.utils import env
from deepmd_pt.utils.region import normalize_coord
from deepmd_pt.utils.nlist import extend_coord_with_ghosts, build_neighbor_list
from deepmd_pt.model.network import TypeEmbedNet

dtype = torch.float64
torch.set_default_dtype(dtype)

CUR_DIR = os.path.dirname(__file__)

# should be a stand-alone function!!!!
def process_nlist(
    nlist, 
    extended_atype, 
    mapping: Optional[torch.Tensor] = None,
):
    # process the nlist_type and nlist_loc
    nframes, nloc = nlist.shape[:2]
    nmask = nlist == -1
    nlist[nmask] = 0
    if mapping is not None:
        nlist_loc = torch.gather(
          mapping, 
          dim=1, 
          index=nlist.reshape(nframes, -1),
        ).reshape(nframes, nloc, -1)
        nlist_loc[nmask] = -1
    else:
        nlist_loc = None
    nlist_type = torch.gather(
      extended_atype, 
      dim=1, 
      index=nlist.reshape(nframes, -1),
    ).reshape(nframes, nloc,-1)
    nlist_type[nmask] = -1
    nlist[nmask] = -1
    return nlist_loc, nlist_type, nframes, nloc

def process_nlist_gathered(
    nlist, 
    extended_atype, 
    split_sel,
    mapping: Optional[torch.Tensor] = None,
):
    nlist_list = list(torch.split(nlist, split_sel, -1))
    nframes, nloc = nlist_list[0].shape[:2]
    nlist_type_list = []
    nlist_loc_list = []
    for nlist_item in nlist_list:
        nmask = nlist_item == -1
        nlist_item[nmask] = 0
        if mapping is not None:
            nlist_loc_item = torch.gather(mapping, dim=1, index=nlist_item.reshape(nframes, -1)).reshape(
                nframes, nloc,
                -1)
            nlist_loc_item[nmask] = -1
            nlist_loc_list.append(nlist_loc_item)
        nlist_type_item = torch.gather(extended_atype, dim=1, index=nlist_item.reshape(nframes, -1)).reshape(
            nframes,
            nloc,
            -1)
        nlist_type_item[nmask] = -1
        nlist_type_list.append(nlist_type_item)
        nlist_item[nmask] = -1

    if mapping is not None:
        nlist_loc = torch.cat(nlist_loc_list, -1)
    else:
        nlist_loc = None
    nlist_type = torch.cat(nlist_type_list, -1)
    return nlist_loc, nlist_type, nframes, nloc



class TestDPA2(unittest.TestCase):
  def setUp(self):
    cell = [
      5.122106549439247480e+00,4.016537340154059388e-01,6.951654033828678081e-01,
      4.016537340154059388e-01,6.112136112297989143e+00,8.178091365465004481e-01,
      6.951654033828678081e-01,8.178091365465004481e-01,6.159552512682983760e+00,
    ]
    self.cell = torch.Tensor(cell).view(1,3,3).to(env.DEVICE)
    coord = [
      2.978060152121375648e+00,3.588469695887098077e+00,2.792459820604495491e+00,3.895592322591093115e+00,2.712091020667753760e+00,1.366836847133650501e+00,9.955616170888935690e-01,4.121324820711413039e+00,1.817239061889086571e+00,3.553661462345699906e+00,5.313046969500791583e+00,6.635182659098815883e+00,6.088601018589653080e+00,6.575011420004332585e+00,6.825240650611076099e+00
    ]
    self.coord = torch.Tensor(coord).view(1,-1,3).to(env.DEVICE)
    self.atype = torch.IntTensor([0, 0, 0, 1, 1]).view(1,-1).to(env.DEVICE)
    self.ref_d = torch.Tensor([
      8.435412613327306630e-01,-4.717109614540972440e-01,-1.812643456954206256e+00,-2.315248767961955167e-01,-7.112973006771171613e-01,-4.162041919507591392e-01,-1.505159810095323181e+00,-1.191652416985768403e-01,8.439214937875325617e-01,-4.712976890460106594e-01,-1.812605149396642856e+00,-2.307222236291133766e-01,-7.115427800870099961e-01,-4.164729253167227530e-01,-1.505483119125936797e+00,-1.191288524278367872e-01,8.286420823261241297e-01,-4.535033763979030574e-01,-1.787877160970498425e+00,-1.961763875645104460e-01,-7.475459187804838201e-01,-5.231446874663764346e-01,-1.488399984491664219e+00,-3.974117581747104583e-02,8.283793431613817315e-01,-4.551551577556525729e-01,-1.789253136645859943e+00,-1.977673627726055372e-01,-7.448826048241211639e-01,-5.161350182531234676e-01,-1.487589463573479209e+00,-4.377376017839779143e-02,8.295404560710329944e-01,-4.492219258475603216e-01,-1.784484611185287450e+00,-1.901182059718481143e-01,-7.537407667483000395e-01,-5.384371277650709109e-01,-1.490368056268364549e+00,-3.073744832541754762e-02
    ]).to(env.DEVICE)
    with open(Path(CUR_DIR)/"models"/"dpa2.json") as fp:
      self.model_json = json.load(fp)
    self.file_model_param = Path(CUR_DIR)/"models"/"dpa2.pth"
    self.file_type_embed = Path(CUR_DIR)/"models"/"dpa2_tebd.pth"

  def test_consistency(self):
    # torch.manual_seed(0)
    model_hybrid_dpa2 = self.model_json
    dparams = model_hybrid_dpa2["descriptor"]
    ntypes = len(model_hybrid_dpa2["type_map"])
    dlist = dparams.pop("list")
    des = DescrptHybrid(
      dlist,
      ntypes,
      hybrid_mode=dparams["hybrid_mode"],
    )
    des.load_state_dict(torch.load(self.file_model_param))
    all_rcut = [ii["rcut"] for ii in dlist]
    all_nsel = [ii["sel"]  for ii in dlist]
    rcut_max = max(all_rcut)
    coord = self.coord
    atype = self.atype
    box = self.cell
    nf, nloc = coord.shape[:2]
    coord_normalized = normalize_coord(
      coord, box.reshape(-1, 3, 3))
    extended_coord, extended_atype, mapping = extend_coord_with_ghosts(
      coord_normalized, atype, box, rcut_max)
    ## single nlist
    # nlist = build_neighbor_list(
    #   extended_coord, extended_atype, nloc,
    #   rcut_max, nsel, distinguish_types=False)
    nlist_list = []
    for rcut, sel in zip(all_rcut, all_nsel):
      nlist_list.append(
        build_neighbor_list(
          extended_coord, 
          extended_atype, 
          nloc,
          rcut, 
          sel, 
          distinguish_types=False,
        ))
      nlist = torch.cat(nlist_list, -1)
    nlist_loc, nlist_type, nframes, nloc = \
      process_nlist_gathered(
        nlist, 
        extended_atype, 
        des.split_sel,
        mapping=mapping,
      )
    # handel type_embedding
    type_embedding = TypeEmbedNet(ntypes, 8)
    type_embedding.load_state_dict(torch.load(self.file_type_embed))
    atype_tebd = type_embedding(atype)
    nlist_type[nlist_type == -1] = ntypes
    nlist_tebd = type_embedding(nlist_type)

    ## to save model parameters
    # torch.save(des.state_dict(), 'model_weights.pth')
    # torch.save(type_embedding.state_dict(), 'model_weights.pth')
    descriptor, env_mat, diff, rot_mat, sw = \
      des(
        extended_coord, 
        nlist, 
        atype, 
        nlist_type=nlist_type,
        nlist_loc=nlist_loc, 
        atype_tebd=atype_tebd,
        nlist_tebd=nlist_tebd,
      )
    torch.testing.assert_close(descriptor.view(-1), self.ref_d, atol=1e-10, rtol=1e-10)
