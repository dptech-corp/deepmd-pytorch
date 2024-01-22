import torch, copy
import unittest
import itertools
import numpy as np

try:
    from deepmd_utils.model_format import (
        # DescrptSeA as DPDescrptSeA,
        PRECISION_DICT as DP_PRECISION_DICT,
    )

    support_se_atten = True
except ModuleNotFoundError:
    support_se_atten = False
except ImportError:
    support_se_atten = False

from deepmd_pt.model.descriptor.dpa1 import (
    DescrptDPA1
)
from deepmd_pt.utils.env import (
    PRECISION_DICT,
    DEFAULT_PRECISION,
)
from deepmd_pt.utils import env
from .test_mlp import get_tols
from IPython import embed

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


@unittest.skipIf(not support_se_atten, "EnvMat not supported")
class TestDescrptSeAtten(unittest.TestCase, TestCaseSingleFrameWithNlist):
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
            dd0 = DescrptDPA1(
                self.rcut, self.rcut_smth, self.sel, self.nt,
                # precision=prec,
                # resnet_dt=idt,
                old_impl=False,
            ).to(env.DEVICE)
            dd0.se_atten.mean = torch.tensor(davg, dtype=dtype, device=env.DEVICE)
            dd0.se_atten.dstd = torch.tensor(dstd, dtype=dtype, device=env.DEVICE)
            rd0, _, _, _, _ = dd0(
                torch.tensor(self.coord_ext, dtype=dtype, device=env.DEVICE),
                torch.tensor(self.atype_ext, dtype=int, device=env.DEVICE),
                torch.tensor(self.nlist, dtype=int, device=env.DEVICE),
            )
            # # serialization
            # dd1 = DescrptDPA1.deserialize(dd0.serialize())
            # rd1, _, _, _, _ = dd1(
            #     torch.tensor(self.coord_ext, dtype=dtype, device=env.DEVICE),
            #     torch.tensor(self.atype_ext, dtype=int, device=env.DEVICE),
            #     torch.tensor(self.nlist, dtype=int, device=env.DEVICE),
            # )
            # np.testing.assert_allclose(
            #     rd0.detach().cpu().numpy(), rd1.detach().cpu().numpy(),
            #     rtol=rtol, atol=atol, err_msg=err_msg,
            # )
            # dp impl
            # dd2 = DPDescrptSeA.deserialize(dd0.serialize())
            # rd2 = dd2.call(
            #     self.coord_ext, self.atype_ext, self.nlist,
            # )
            # np.testing.assert_allclose(
            #     rd0.detach().cpu().numpy(), rd2,
            #     rtol=rtol, atol=atol, err_msg=err_msg,
            # )
            # old impl
            if idt == False and prec == "float64":
                dd3 = DescrptDPA1(
                    self.rcut, self.rcut_smth, self.sel, self.nt,
                    # precision=prec,
                    # resnet_dt=idt,
                    old_impl=True,
                ).to(env.DEVICE)
                dd0_state_dict = dd0.se_atten.state_dict()
                dd3_state_dict = dd3.se_atten.state_dict()

                dd0_state_dict_attn = dd0.se_atten.dpa1_attention.state_dict()
                dd3_state_dict_attn = dd3.se_atten.dpa1_attention.state_dict()
                embed()
                for i in dd3_state_dict:
                    dd3_state_dict[i] = dd0_state_dict[i.replace('.deep_layers.', '.layers.')
                        .replace('filter_layers_old.', 'filter_layers.networks.')].detach().clone()
                    if '.bias' in i and 'attn_layer_norm' not in i:
                        dd3_state_dict[i] = dd3_state_dict[i].unsqueeze(0)
                dd3.se_atten.load_state_dict(dd3_state_dict)

                dd0_state_dict_tebd = dd0.type_embedding.state_dict()
                dd3_state_dict_tebd = dd3.type_embedding_old.state_dict()
                for i in dd3_state_dict_tebd:
                    dd3_state_dict_tebd[i] = dd0_state_dict_tebd[i.replace('embedding.weight', 'matrix')].detach().clone()
                dd3.type_embedding_old.load_state_dict(dd3_state_dict_tebd)

                rd3, _, _, _, _ = dd3(
                    torch.tensor(self.coord_ext, dtype=dtype, device=env.DEVICE),
                    torch.tensor(self.atype_ext, dtype=int, device=env.DEVICE),
                    torch.tensor(self.nlist, dtype=int, device=env.DEVICE),
                )
                np.testing.assert_allclose(
                    rd0.detach().cpu().numpy(), rd3.detach().cpu().numpy(),
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
            dd0 = DescrptDPA1(
                self.rcut, self.rcut_smth, self.sel, self.nt,
                # precision=prec,
                # resnet_dt=idt,
                old_impl=False,
            )
            dd0.se_atten.mean = torch.tensor(davg, dtype=dtype, device=env.DEVICE)
            dd0.se_atten.dstd = torch.tensor(dstd, dtype=dtype, device=env.DEVICE)
            # dd1 = DescrptDPA1.deserialize(dd0.serialize())
            model = torch.jit.script(dd0)
            # model = torch.jit.script(dd1)
