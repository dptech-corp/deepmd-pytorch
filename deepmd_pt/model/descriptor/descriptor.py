import numpy as np
import torch

from deepmd_pt.utils import env
from deepmd_pt.utils.plugin import Plugin, PluginVariant
from typing import Callable

try:
    from typing import Final
except:
    from torch.jit import Final


class Descriptor(torch.nn.Module):

    __plugins = Plugin()
    local_cluster = False

    @staticmethod
    def register(key: str) -> Callable:
        """Register a descriptor plugin.

        Parameters
        ----------
        key : str
            the key of a descriptor

        Returns
        -------
        Descriptor
            the registered descriptor

        Examples
        --------
        >>> @Descriptor.register("some_descrpt")
            class SomeDescript(Descriptor):
                pass
        """
        return Descriptor.__plugins.register(key)

    def __new__(cls, *args, **kwargs):
        if cls is Descriptor:
            try:
                descrpt_type = kwargs["type"]
            except KeyError:
                raise KeyError("the type of descriptor should be set by `type`")
            if descrpt_type in Descriptor.__plugins.plugins:
                cls = Descriptor.__plugins.plugins[descrpt_type]
            else:
                raise RuntimeError("Unknown descriptor type: " + descrpt_type)
        return super().__new__(cls)

    @property
    def dim_out(self):
        """
        Returns the output dimension of this descriptor
        """
        return self.filter_neuron[-1] * self.axis_neuron

    @property
    def dim_in(self):
        """
        Returns the atomic input dimension of this descriptor
        """
        return self.tebd_dim

    def compute_input_stats(self, merged):
        """Update mean and stddev for descriptor elements.
        """
        raise NotImplementedError

    def share_params(self, base_class, shared_level, resume=False):
        assert self.__class__ == base_class.__class__, "Only descriptors of the same type can share params!"
        if shared_level == 0:
            # link buffers
            if hasattr(self, 'mean') and not resume:
                # in case of change params during resume
                sumr_base, suma_base, sumn_base, sumr2_base, suma2_base = \
                    base_class.sumr, base_class.suma, base_class.sumn, base_class.sumr2, base_class.suma2
                sumr, suma, sumn, sumr2, suma2 = self.sumr, self.suma, self.sumn, self.sumr2, self.suma2
                base_class.init_desc_stat(sumr_base + sumr, suma_base + suma, sumn_base + sumn, sumr2_base + sumr2, suma2_base + suma2)
                self.mean = base_class.mean
                self.stddev = base_class.stddev
            # self.load_state_dict(base_class.state_dict()) # this does not work, because it only inits the model
            # the following will successfully link all the params except buffers
            for item in self._modules:
                self._modules[item] = base_class._modules[item]
        else:
            raise NotImplementedError

    def forward(self, **kwargs):
        """Calculate descriptor.
        """
        raise NotImplementedError


def compute_std(sumv2, sumv, sumn, rcut_r):
    """Compute standard deviation."""
    if sumn == 0:
        return 1.0 / rcut_r
    val = np.sqrt(sumv2 / sumn - np.multiply(sumv / sumn, sumv / sumn))
    if np.abs(val) < 1e-2:
        val = 1e-2
    return val
