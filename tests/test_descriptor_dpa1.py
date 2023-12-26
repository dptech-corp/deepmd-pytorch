import numpy as np
import os
import torch
import unittest
import json
from typing import Optional
from pathlib import Path

from deepmd_pt.model.descriptor import DescrptBlockSeAtten, DescrptDPA1
from deepmd_pt.utils import env
from deepmd_pt.utils.region import normalize_coord
from deepmd_pt.utils.nlist import extend_coord_with_ghosts, build_neighbor_list
from deepmd_pt.model.network import TypeEmbedNet
from deepmd_pt.model.model.ener import process_nlist, process_nlist_gathered

dtype = torch.float64
torch.set_default_dtype(dtype)

CUR_DIR = os.path.dirname(__file__)


class TestDPA1(unittest.TestCase):
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
      8.390509308881095168e-03,-3.412049981839327187e-03,6.134524164534142093e-03,-4.869659114157234153e-03,-3.412049981839327187e-03,1.389926640573683385e-03,-2.492242439900287619e-03,1.977904251902020023e-03,6.134524164534142093e-03,-2.492242439900287619e-03,4.487696916396456334e-03,-3.562873966485257966e-03,-4.869659114157234153e-03,1.977904251902020023e-03,-3.562873966485257966e-03,2.828731246989599473e-03,6.720053086229856075e-04,-2.730244163316378051e-04,4.913787834335673656e-04,-3.900924719518377771e-04,-1.310553280597828718e-02,5.327229517958668309e-03,-9.584284190123706915e-03,7.608578158948461553e-03,8.897479944574801861e-03,-3.619486700779906210e-03,6.504148788253220036e-03,-5.162846352688373747e-03,-2.707276824638176534e-03,1.101743128416786833e-03,-1.978348056108133329e-03,1.570258247442622105e-03,8.653754618809864074e-03,-3.586892629863162006e-03,6.282138224856510626e-03,-4.984864924611785882e-03,-3.586892629863162006e-03,1.492815991155331621e-03,-2.596939208238854043e-03,2.059458608896123748e-03,6.282138224856510626e-03,-2.596939208238854043e-03,4.569101868654634226e-03,-3.627052390124005984e-03,-4.984864924611785882e-03,2.059458608896123748e-03,-3.627052390124005984e-03,2.879490359579183877e-03,6.669657604991129519e-04,-2.764293742654066631e-04,4.834474537429071469e-04,-3.835184164130038607e-04,-1.348115518759216737e-02,5.581182821261431760e-03,-9.795140389836738334e-03,7.773879168933476615e-03,9.247346160937685092e-03,-3.835348164542660022e-03,6.711201103709644143e-03,-5.324971487920848214e-03,-2.797292023036055536e-03,1.162151955597194412e-03,-2.026756662764096426e-03,1.607579081660859320e-03,9.270290047763521044e-03,-3.503129725797998317e-03,7.088926802955019917e-03,-5.682103108732513425e-03,-3.503129725797998317e-03,1.331004878576913259e-03,-2.672531257296904827e-03,2.140853367246304401e-03,7.088926802955019917e-03,-2.672531257296904827e-03,5.427402081970027561e-03,-4.351646081457302521e-03,-5.682103108732513425e-03,2.140853367246304401e-03,-4.351646081457302521e-03,3.489385307028703906e-03,7.164097982283839901e-04,-2.688211163539352554e-04,5.483109122590840157e-04,-4.396245057359132215e-04,-1.479719542263101609e-02,5.586430250565351321e-03,-1.132143700641702680e-02,9.075887780821859790e-03,9.745644926466342367e-03,-3.687914440894251265e-03,7.449297163144590408e-03,-5.970273784111346065e-03,-2.838843318251735911e-03,1.074108043954517377e-03,-2.168322615157486886e-03,1.737521024940204967e-03,9.866877035453004707e-03,-3.813216116288485266e-03,7.120161200173152211e-03,-5.701876425667436882e-03,-3.813216116288485266e-03,1.476183293695302289e-03,-2.748090331629427386e-03,2.200259566694174208e-03,7.120161200173152211e-03,-2.748090331629427386e-03,5.143403459174657015e-03,-4.119495023007761422e-03,-5.701876425667436882e-03,2.200259566694174208e-03,-4.119495023007761422e-03,3.299492395932577312e-03,1.370824131166433488e-03,-5.318944754084920738e-04,9.859567041104898202e-04,-7.891993570020113717e-04,-1.507416180648146579e-02,5.821233573540246856e-03,-1.088444346743589961e-02,8.717099722250312363e-03,9.765958363289788807e-03,-3.773102455371011264e-03,7.049157325005818604e-03,-5.645204781612849117e-03,-3.534566381644995824e-03,1.368626944767693289e-03,-2.546649866391029020e-03,2.038923902412996536e-03,7.472975674461762238e-03,-2.984494758306116546e-03,5.328658756865414098e-03,-4.251620491158816101e-03,-2.984494758306116546e-03,1.194756210179206881e-03,-2.125344760730021876e-03,1.695317291286985856e-03,5.328658756865414098e-03,-2.125344760730021876e-03,3.802430822218539220e-03,-3.034304736833612608e-03,-4.251620491158816101e-03,1.695317291286985856e-03,-3.034304736833612608e-03,2.421419645162038552e-03,9.738325276536374678e-04,-3.890631849521523737e-04,6.941002815031252927e-04,-5.538053281445152369e-04,-1.138251745729029242e-02,4.543189135192816572e-03,-8.119150905070714880e-03,6.478495168470982737e-03,7.512894138032612246e-03,-3.001712903217535616e-03,5.356042280108381029e-03,-4.273245644801249021e-03,-2.669728390258197404e-03,1.067352970546816453e-03,-1.902431738061107240e-03,1.517744096018183112e-03
    ]).to(env.DEVICE)
    with open(Path(CUR_DIR)/"models"/"dpa1.json") as fp:
      self.model_json = json.load(fp)
    self.file_model_param = Path(CUR_DIR)/"models"/"dpa1.pth"
    self.file_type_embed = Path(CUR_DIR)/"models"/"dpa2_tebd.pth"

  def test_descriptor_block(self):
    # torch.manual_seed(0)
    model_dpa1 = self.model_json
    dparams = model_dpa1["descriptor"]
    ntypes = len(model_dpa1["type_map"])
    assert "se_atten" == dparams.pop("type")
    dparams["ntypes"] = ntypes
    des = DescrptBlockSeAtten(
      **dparams,
    )
    des.load_state_dict(torch.load(self.file_model_param))
    rcut = dparams["rcut"]
    nsel = dparams["sel"]
    coord = self.coord
    atype = self.atype
    box = self.cell
    nf, nloc = coord.shape[:2]
    coord_normalized = normalize_coord(
      coord, box.reshape(-1, 3, 3))
    extended_coord, extended_atype, mapping = extend_coord_with_ghosts(
      coord_normalized, atype, box, rcut)
    # single nlist
    nlist = build_neighbor_list(
      extended_coord, extended_atype, nloc,
      rcut, nsel, distinguish_types=False)
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
        mapping=None,
      )
    # np.savetxt('tmp.out', descriptor.detach().numpy().reshape(1,-1), delimiter=",")
    self.assertEqual(descriptor.shape[-1], des.get_dim_out())
    self.assertAlmostEqual(6., des.get_rcut())
    self.assertEqual(30, des.get_nsel())
    self.assertEqual(2, des.get_ntype())
    torch.testing.assert_close(descriptor.view(-1), self.ref_d, atol=1e-10, rtol=1e-10)


  def test_descriptor(self):
    with open(Path(CUR_DIR)/"models"/"dpa1.json") as fp:
      self.model_json = json.load(fp)
    model_dpa2 = self.model_json
    ntypes = len(model_dpa2["type_map"])
    dparams = model_dpa2["descriptor"]
    dparams["ntypes"] = ntypes
    assert dparams.pop("type") == "se_atten"
    dparams["concat_output_tebd"] = False
    des = DescrptDPA1(
      **dparams,
    )
    target_dict = des.state_dict()
    source_dict = torch.load(self.file_model_param)
    type_embd_dict = torch.load(self.file_type_embed)
    target_dict = translate_se_atten_and_type_embd_dicts_to_dpa1(
      target_dict,
      source_dict,
      type_embd_dict,
    )
    des.load_state_dict(target_dict)

    coord = self.coord
    atype = self.atype
    box = self.cell
    nf, nloc = coord.shape[:2]
    coord_normalized = normalize_coord(
      coord, box.reshape(-1, 3, 3))
    extended_coord, extended_atype, mapping = extend_coord_with_ghosts(
      coord_normalized, atype, box, des.get_rcut())
    nlist = build_neighbor_list(
      extended_coord, extended_atype, nloc,
      des.get_rcut(), des.get_nsel(), distinguish_types=False,
    )
    descriptor, env_mat, diff, rot_mat, sw = \
      des(
        nlist,
        extended_coord,
        extended_atype,
        mapping=mapping,
      )
    self.assertEqual(descriptor.shape[-1], des.get_dim_out())
    self.assertAlmostEqual(6., des.get_rcut())
    self.assertEqual(30, des.get_nsel())
    self.assertEqual(2, des.get_ntype())
    torch.testing.assert_close(descriptor.view(-1), self.ref_d, atol=1e-10, rtol=1e-10)

    dparams["concat_output_tebd"] = True
    des = DescrptDPA1(
      **dparams,
    )
    descriptor, env_mat, diff, rot_mat, sw = \
      des(
        nlist,
        extended_coord,
        extended_atype,
        mapping=mapping,
      )
    self.assertEqual(descriptor.shape[-1], des.get_dim_out())


def translate_se_atten_and_type_embd_dicts_to_dpa1(
    target_dict,
    source_dict,
    type_embd_dict,
):
  all_keys = list(target_dict.keys())
  record = [False for ii in all_keys]
  for kk, vv in source_dict.items():
    tk = "se_atten." + kk
    record[all_keys.index(tk)] = True
    target_dict[tk] = vv
  assert len(type_embd_dict.keys()) == 1
  kk = [ii for ii in type_embd_dict.keys()][0]
  tk = "type_embedding." + kk
  record[all_keys.index(tk)] = True
  target_dict[tk] = type_embd_dict[kk]
  assert(all(record))
  return target_dict
