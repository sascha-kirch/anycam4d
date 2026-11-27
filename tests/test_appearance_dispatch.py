"""The appearance kind crosses the pybind boundary as an int, so both sides must agree on it."""

from __future__ import annotations

import pytest

from anycam4d.cuda.ext import HAS_CUDA_EXT
from anycam4d.cuda.ext import MAX_TEMPORAL_DEGREE_CUDA
from anycam4d.cuda.ext import AppearanceKind as CudaAppearanceKind
from anycam4d.primitives.appearance import APPEARANCE_CUDA_ID
from anycam4d.primitives.appearance import MAX_TEMPORAL_DEGREE
from anycam4d.enums import AppearanceType
from anycam4d.primitives.appearance import APPEARANCE_PARAM


@pytest.mark.skipif(not HAS_CUDA_EXT, reason="the C++ enum ships with the CUDA extension")
def test_appearance_ids_mirror_the_cuda_enum() -> None:
    """A drifted id silently selects the wrong colour leaf: renders, wrong basis, no error."""
    exported = {name: int(value) for name, value in CudaAppearanceKind.__members__.items()}

    assert exported == {kind.value.removeprefix("sh4d_").capitalize(): id_ for kind, id_ in APPEARANCE_CUDA_ID.items()}


@pytest.mark.skipif(not HAS_CUDA_EXT, reason="the C++ constant ships with the CUDA extension")
def test_the_temporal_degree_mirrors_the_cuda_constant() -> None:
    """Python sizes sh_spacetime by its copy and the kernel splits the stride by the C++ one."""
    assert MAX_TEMPORAL_DEGREE_CUDA == MAX_TEMPORAL_DEGREE


def test_every_appearance_model_has_an_id_and_a_parameter() -> None:
    """The model consults both maps, so a missing entry would fail only at render."""
    assert set(APPEARANCE_CUDA_ID) == set(APPEARANCE_PARAM) == set(AppearanceType)
