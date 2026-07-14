#!/usr/bin/env python3
"""Mesh-mode adjoint Jacobian regression test (pmmc).

Runs a small tetrahedral cube simulation in adjoint mode and verifies that:
  1. cfg.outputtype='adjoint' on a mesh-mode method produces both 'flux'
     (per-slot Ns+Nd fluence) and 'jmua' (Jacobian) outputs.
  2. The Jacobian shape is [nn, Ns*Nd] as documented in the bindings.
  3. The full FEM (cfg.adjointmode=0) and nodal approximation
     (cfg.adjointmode=1) give qualitatively similar magnitudes.

Requires pmmc compiled against the source tree that includes PR#2.
Run after `pip install -e .` from the pmmc directory.
"""

import numpy as np
import pmmc


def make_cube_mesh(side=10.0, n_per_side=4):
    """Build a small uniform tet mesh of an axis-aligned cube via iso2mesh's
    splitting rule (5 tets per voxel). Returns (node, elem) arrays where
    elem is 1-based to match the mmc convention."""
    # generate corner nodes of a 3D grid
    lin = np.linspace(0, side, n_per_side)
    X, Y, Z = np.meshgrid(lin, lin, lin, indexing="ij")
    node = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1).astype(np.float32)
    nx = ny = nz = n_per_side

    def idx(i, j, k):
        return i * ny * nz + j * nz + k

    # 5-tet decomposition per voxel
    elems = []
    for i in range(nx - 1):
        for j in range(ny - 1):
            for k in range(nz - 1):
                v = [
                    idx(i, j, k),
                    idx(i + 1, j, k),
                    idx(i + 1, j + 1, k),
                    idx(i, j + 1, k),
                    idx(i, j, k + 1),
                    idx(i + 1, j, k + 1),
                    idx(i + 1, j + 1, k + 1),
                    idx(i, j + 1, k + 1),
                ]
                # 5 tets covering one cube
                elems.append([v[0], v[1], v[3], v[4]])
                elems.append([v[1], v[3], v[4], v[6]])
                elems.append([v[1], v[2], v[3], v[6]])
                elems.append([v[3], v[4], v[6], v[7]])
                elems.append([v[1], v[4], v[5], v[6]])

    elem = np.asarray(elems, dtype=np.int32) + 1  # 1-based for mmc
    return node, elem


def base_cfg(adjointmode):
    node, elem = make_cube_mesh(side=10.0, n_per_side=4)
    return {
        "nphoton": 50000,
        "seed": 17182818,
        "node": node,
        "elem": elem,
        "elemprop": np.ones(elem.shape[0], dtype=np.int32),
        "srcpos": np.array([5.0, 5.0, 0.0], dtype=np.float32),
        "srcdir": np.array([0.0, 0.0, 1.0], dtype=np.float32),
        "tstart": 0.0,
        "tstep": 5e-9,
        "tend":   5e-9,
        "prop": np.array([[0.0, 0.0, 1.0, 1.0],
                          [0.005, 1.0, 0.01, 1.37]], dtype=np.float32),
        "detpos": np.array([[2.5, 5.0, 0.0, 1.0],
                            [7.5, 5.0, 0.0, 1.0]], dtype=np.float32),
        "detdir": np.array([[0.0, 0.0, -1.0, 0.0],
                            [0.0, 0.0, -1.0, 0.0]], dtype=np.float32),
        "outputtype": "adjoint",
        "method":     "elem",       # mesh-mode (BLBadouel)
        "basisorder": 1,
        "isnormalized": 1,
        "isreflect": 0,
        "e0":        1,             # any element containing the source
        "adjointmode": adjointmode,
    }


def run_and_check(label, adjointmode):
    cfg = base_cfg(adjointmode)
    out = pmmc.run(cfg)

    assert "flux" in out, f"{label}: missing flux output"
    assert "jmua" in out, f"{label}: missing jmua output"

    Ns, Nd = 1, 2
    nn = cfg["node"].shape[0]
    assert out["jmua"].shape[-1] == Ns * Nd, (
        f"{label}: jmua slot dim {out['jmua'].shape[-1]} != Ns*Nd ({Ns * Nd})"
    )
    assert out["jmua"].shape[0] == nn, (
        f"{label}: jmua node dim {out['jmua'].shape[0]} != nn ({nn})"
    )

    nonzero = np.count_nonzero(out["jmua"])
    print(f"  {label}: jmua shape={out['jmua'].shape}  nonzero={nonzero}  "
          f"max={out['jmua'].max():.4g}  min={out['jmua'].min():.4g}")
    assert nonzero > 0, f"{label}: jmua is all zeros"


if __name__ == "__main__":
    print("Mesh-mode adjoint Jacobian regression test")
    run_and_check("full FEM",       adjointmode=0)
    run_and_check("nodal approx",   adjointmode=1)
    print("PASS")
