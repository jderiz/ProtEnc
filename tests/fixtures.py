from collections import OrderedDict

import numpy as np
import pytest


@pytest.fixture
def protein_dict():
    return OrderedDict(
        [
            (
                "prot1",
                "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
            ),
            (
                "prot2",
                "KALTARQQEVFDLIRDHISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE",
            ),
            (
                "prot3",
                "GPCTSFPQALCVQWKNAYALCWLDCILSALVHSEELKNTVTGLCSKEESIFWRLLTKYNQANTLLYTSQLSGVKDGDCKKLTSEIFAEIETCLNEVRDEIFISLQPQL"
                "RCTLGDMESPVFAFPLLLKLETHIEKLFLYSFSWDFECSQCGHQYQNRHMKSLVTFTNVIPEWHPLNAAHFGPCNNCNSKSQIRKMVLEKVSPIFMLHFVEGLPQNDL"
                "QHYAFHFEGCLYQITSVIQYRANNHFITWILDADGSWLECDDLKGPCSERHKKFEVPASEIHIVIWERKIS",
            ),
        ]
    )


@pytest.fixture
def proteins(protein_dict):
    return list(protein_dict.values())


def write_backbone_pdb(path, length, seed=0):
    """Write a chain of ``length`` residues 3.8 A apart with N/CA/C/O atoms.

    The geometry is arbitrary; it only has to be finite and give every residue
    neighbors. Returns the path as a string.
    """
    rng = np.random.default_rng(seed)
    ca = np.cumsum(rng.normal(size=(length, 3)), axis=0)
    ca = ca / np.linalg.norm(np.diff(ca, axis=0), axis=1).mean() * 3.8
    offsets = {
        "N": (-1.4, 0.3, 0.0),
        "CA": (0.0, 0.0, 0.0),
        "C": (1.4, 0.4, 0.0),
        "O": (2.1, 1.3, 0.2),
    }
    lines = []
    serial = 1
    for i, c in enumerate(ca, start=1):
        for atom, off in offsets.items():
            x, y, z = c + np.array(off)
            lines.append(
                f"ATOM  {serial:5d} {atom:<4s} ALA A{i:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           {atom[0]:>2s}"
            )
            serial += 1
    path.write_text("\n".join(lines) + "\nEND\n")
    return str(path)
