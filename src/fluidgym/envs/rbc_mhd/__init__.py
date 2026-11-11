"""Rayleigh-Bénard Convection (RBC) environments under a horizontal magnetic field."""

from .rbc_mhd_env_3d import RBC_MHD_3D_DEFAULT_CONFIG, RBCMHDEnv3D

__all__ = [
    "RBCMHDEnv3D",
    "RBC_MHD_3D_DEFAULT_CONFIG",
]
