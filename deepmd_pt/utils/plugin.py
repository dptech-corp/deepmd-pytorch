# SPDX-License-Identifier: LGPL-3.0-or-later
"""Base of plugin systems."""
from deepmd_utils.utils.plugin import (
    Plugin,
    PluginVariant,
    VariantABCMeta,
    VariantMeta,
)

__all__ = [
    "Plugin",
    "VariantMeta",
    "VariantABCMeta",
    "PluginVariant",
]
