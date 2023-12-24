import numpy as np
import os
import torch
import unittest
import json
from typing import Optional
from pathlib import Path

from deepmd_pt.model.descriptor import DescrptHybrid, DescrptDPA2
from deepmd_pt.utils import env
from deepmd_pt.utils.region import normalize_coord
from deepmd_pt.utils.nlist import extend_coord_with_ghosts, build_neighbor_list
from deepmd_pt.model.network import TypeEmbedNet
from deepmd_pt.model.model.ener import process_nlist, process_nlist_gathered

dtype = torch.float64
torch.set_default_dtype(dtype)

CUR_DIR = os.path.dirname(__file__)


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
    with open(Path(CUR_DIR)/"models"/"dpa2_hyb.json") as fp:
      self.model_json = json.load(fp)
    self.file_model_param = Path(CUR_DIR)/"models"/"dpa2.pth"
    self.file_type_embed = Path(CUR_DIR)/"models"/"dpa2_tebd.pth"

  def test_descriptor_hyb(self):
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
    # handel type_embedding
    type_embedding = TypeEmbedNet(ntypes, 8)
    type_embedding.load_state_dict(torch.load(self.file_type_embed))

    ## to save model parameters
    # torch.save(des.state_dict(), 'model_weights.pth')
    # torch.save(type_embedding.state_dict(), 'model_weights.pth')
    descriptor, env_mat, diff, rot_mat, sw = \
      des(
        nlist, 
        extended_coord,
        extended_atype,
        type_embedding(extended_atype),
        mapping=mapping,
      )
    torch.testing.assert_close(descriptor.view(-1), self.ref_d, atol=1e-10, rtol=1e-10)




  def test_descriptor(self):
    with open(Path(CUR_DIR)/"models"/"dpa2.json") as fp:
      self.model_json = json.load(fp)
    model_dpa2 = self.model_json
    ntypes = len(model_dpa2["type_map"])
    dparams = model_dpa2["descriptor"]
    dparams["ntypes"] = ntypes
    assert dparams.pop("type") == "dpa2"
    des = DescrptDPA2(
      **dparams,
    )
    target_dict = des.state_dict()
    source_dict = torch.load(self.file_model_param)
    type_embd_dict = torch.load(self.file_type_embed)
    target_dict = translate_hybrid_and_type_embd_dicts_to_dpa2(
      target_dict,
      source_dict,
      type_embd_dict,
    )
    des.load_state_dict(target_dict)
    torch.save(des.state_dict(), 'model_weights.pth')    
    
    coord = self.coord
    atype = self.atype
    box = self.cell
    nf, nloc = coord.shape[:2]
    coord_normalized = normalize_coord(
      coord, box.reshape(-1, 3, 3))
    extended_coord, extended_atype, mapping = extend_coord_with_ghosts(
      coord_normalized, atype, box, des.repinit.rcut)

    nlist = build_neighbor_list(
      extended_coord, extended_atype, nloc,
      des.repinit.rcut, des.repinit.sel, distinguish_types=False)
    descriptor, env_mat, diff, rot_mat, sw = \
      des(
        nlist, 
        extended_coord,
        extended_atype,
        None,
        mapping=mapping,
      )
    torch.testing.assert_close(descriptor.view(-1), self.ref_d, atol=1e-10, rtol=1e-10)


def translate_hybrid_and_type_embd_dicts_to_dpa2(
    target_dict,
    source_dict,
    type_embd_dict,
):
  all_keys = list(target_dict.keys())
  record = [False for ii in all_keys]
  for kk, vv in source_dict.items():
    tk = kk.replace("descriptor_list.1", "repformer")
    tk = tk.replace("descriptor_list.0", "repinit")
    tk = tk.replace("sequential_transform.0", "g1_shape_tranform")
    record[all_keys.index(tk)] = True
    target_dict[tk] = vv  
  assert len(type_embd_dict.keys()) == 1
  kk = [ii for ii in type_embd_dict.keys()][0]
  tk = "type_embedding." + kk
  record[all_keys.index(tk)] = True
  target_dict[tk] = type_embd_dict[kk]
  assert(all(record))
  return target_dict
  
