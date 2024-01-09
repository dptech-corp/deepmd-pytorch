import torch, copy
import unittest
import itertools
import numpy as np

try :
  from deepmd_utils.model_format import (
    DescrptSeA as DPDescrptSeA,
    PRECISION_DICT as DP_PRECISION_DICT,
  )
  support_se_e2_a = True
except ModuleNotFoundError:
  support_se_e2_a = False
except ImportError:
  support_se_e2_a = False

from deepmd_pt.model.descriptor.se_a import (
  DescrptSeA
)
from deepmd_pt.utils.env import (
  PRECISION_DICT,
  DEFAULT_PRECISION,
)
from deepmd_pt.utils import env
from .test_mlp import get_tols

dtype = env.GLOBAL_PT_FLOAT_PRECISION

class TestCaseSingleFrameWithNlist():
    def setUp(self):
        # nloc == 3, nall == 4
        self.nloc = 3
        self.nall = 4
        self.nf, self.nt = 1, 2
        self.coord_ext = np.array(
            [
                [0, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
                [0, -2, 0],
            ],
            dtype=np.float64,
        ).reshape([1, self.nall * 3])
        self.atype_ext = np.array([0, 0, 1, 0], dtype=int).reshape([1, self.nall])
        # sel = [5, 2]
        self.sel = [5, 2]
        self.nlist = np.array(
            [
                [1, 3, -1, -1, -1, 2, -1],
                [0, -1, -1, -1, -1, 2, -1],
                [0, 1, -1, -1, -1, 0, -1],
            ],
            dtype=int,
        ).reshape([1, self.nloc, sum(self.sel)])
        self.rcut = 0.4
        self.rcut_smth = 2.2  


# to be merged with the tf test case
@unittest.skipIf(not support_se_e2_a, "EnvMat not supported")
class TestDescrptSeA(unittest.TestCase, TestCaseSingleFrameWithNlist):
  def setUp(self):
    TestCaseSingleFrameWithNlist.setUp(self)

  def test_consistency(
      self,
  ):
    rng = np.random.default_rng()
    nf, nloc, nnei = self.nlist.shape
    davg = rng.normal(size=(self.nt, nnei, 4))
    dstd = rng.normal(size=(self.nt, nnei, 4))
    dstd = 0.1 + np.abs(dstd)

    for idt, prec in itertools.product(
        [False, True],
        ["float64", "float32"],
    ):
      dtype = PRECISION_DICT[prec]
      rtol, atol = get_tols(prec)
      err_msg = f"idt={idt} prec={prec}"
      # sea new impl
      dd0 = DescrptSeA(
        self.rcut, self.rcut_smth, self.sel, 
        precision=prec,
        resnet_dt=idt,
        old_impl=False,
      )
      dd0.mean = davg
      dd0.dstd = dstd
      rd0, _,_,_,_ = dd0(
        torch.tensor(self.nlist, dtype=int), 
        torch.tensor(self.coord_ext, dtype=dtype), 
        torch.tensor(self.atype_ext, dtype=int),
      )
      # old impl
      dd1 = DescrptSeA.deserialize(dd0.serialize())
      dd1.old_impl = False
      rd1, _,_,_,_ = dd1(
        torch.tensor(self.nlist, dtype=int), 
        torch.tensor(self.coord_ext, dtype=dtype), 
        torch.tensor(self.atype_ext, dtype=int),
      )
      np.testing.assert_allclose(
        rd0.detach().numpy(), rd1.detach().numpy(),
        rtol=rtol, atol=atol, err_msg=err_msg,
      )
      # dp impl
      dd2 = DPDescrptSeA.deserialize(dd0.serialize())
      rd2 = dd2.call(
        self.coord_ext, self.atype_ext, self.nlist,
      )
      np.testing.assert_allclose(
        rd0.detach().numpy(), rd2, 
        rtol=rtol, atol=atol, err_msg=err_msg,
      )
      

  def test_jit(
      self,
  ):
    rng = np.random.default_rng()
    nf, nloc, nnei = self.nlist.shape
    davg = rng.normal(size=(self.nt, nnei, 4))
    dstd = rng.normal(size=(self.nt, nnei, 4))
    dstd = 0.1 + np.abs(dstd)

    for idt, prec in itertools.product(
        [False, True],
        ["float64", "float32"],
    ):
      dtype = PRECISION_DICT[prec]
      rtol, atol = get_tols(prec)
      err_msg = f"idt={idt} prec={prec}"
      # sea new impl
      dd0 = DescrptSeA(
        self.rcut, self.rcut_smth, self.sel, 
        precision=prec,
        resnet_dt=idt,
        old_impl=False,
      )
      dd0.mean = davg
      dd0.dstd = dstd
      dd1 = DescrptSeA.deserialize(dd0.serialize())
      model = torch.jit.script(dd0)
      model = torch.jit.script(dd1)
