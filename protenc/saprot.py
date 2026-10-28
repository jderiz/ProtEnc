"""Structure-aware tokens for SaProt: each residue becomes ``<AA><3Di letter>``.

SaProt reads a residue and its Foldseek 3Di state as one token (``"Ev"`` is glutamate
in 3Di state ``v``). ``#`` in the 3Di slot means "structure unknown", which is also how
the model is run on sequences alone. 3Di letters come from ``mini3di``, a NumPy port of
the Foldseek encoder, so no Foldseek binary is needed.
"""

from pathlib import Path

import numpy as np

MASK_3DI = "#"
# The vocabulary only holds these residues. Anything else ("X", "B", "U", ...) is
# masked as "##": an out-of-vocabulary pair such as "X#" is not one token, and a run of
# them is merged into a single unknown token, which drops rows from the embedding.
STANDARD_RESIDUES = frozenset("ACDEFGHIKLMNPQRSTVWY")
# SaProt masks the 3Di state of AlphaFold residues below this pLDDT.
PLDDT_THRESHOLD = 70.0

_encoder = None


def _residue(aa: str) -> str:
    return aa if aa in STANDARD_RESIDUES else MASK_3DI


def sequence_only(sequence: str) -> str:
    """``"MKV"`` -> ``"M#K#V#"``; residues outside the vocabulary become ``"##"``."""
    return "".join(f"{_residue(aa)}{MASK_3DI}" for aa in sequence.upper())


def combine(sequence: str, states: str) -> str:
    """Interleave residues with their 3Di states (one state per residue)."""
    if len(sequence) != len(states):
        raise ValueError(
            f"{len(sequence)} residues but {len(states)} 3Di states to combine"
        )
    return "".join(
        f"{_residue(aa)}{s if aa in STANDARD_RESIDUES else MASK_3DI}"
        for aa, s in zip(sequence.upper(), states)
    )


def _get_encoder():
    global _encoder
    if _encoder is None:
        try:
            from mini3di import Encoder
        except ImportError as e:
            raise ImportError(
                "SaProt with structures needs the 'mini3di' package (pip install mini3di)."
            ) from e
        _encoder = Encoder()
    return _encoder


def _load_structure(path: str | Path):
    from Bio.PDB import MMCIFParser, PDBParser

    path = Path(path)
    parser = (
        MMCIFParser(QUIET=True)
        if path.suffix.lower() in {".cif", ".mmcif"}
        else PDBParser(QUIET=True)
    )
    return parser.get_structure(path.stem, str(path))[0]


def _residue_coords(residues) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """CA, CB, N and C coordinates per residue; NaN where an atom is missing."""
    ca, cb, n, c = (np.full((len(residues), 3), np.nan, np.float32) for _ in range(4))
    targets = {"CA": ca, "CB": cb, "N": n, "C": c}
    for i, residue in enumerate(residues):
        for atom in residue.get_atoms():
            if atom.get_name() in targets:
                targets[atom.get_name()][i] = atom.coord
    return ca, cb, n, c


def chain_states(
    path: str | Path,
    chain_list: list[str] | None = None,
    plddt_mask: bool = False,
) -> list[str]:
    """3Di string per chain, one letter per residue that has a CA atom.

    Args:
        path: PDB or mmCIF file.
        chain_list: Chains to encode, in output order. ``None`` takes every chain
            in file order.
        plddt_mask: Replace the state of residues whose CA B-factor (pLDDT in
            AlphaFold files) is below 70 with ``#``.

    Hetero residues (ligands, ions, water) are skipped, so a calcium ion's ``CA``
    atom is not read as a residue. Residues that have a CA but too few neighbouring
    atoms to describe get ``#``.
    """
    model = _load_structure(path)
    chains = {chain.id: chain for chain in model}
    if chain_list is None:
        chain_list = list(chains)
    missing = [c for c in chain_list if c not in chains]
    if missing:
        raise ValueError(f"Chains {missing} not in {path} (has {list(chains)})")

    encoder = _get_encoder()
    out = []
    for chain_id in chain_list:
        residues = [r for r in chains[chain_id] if r.id[0] == " " and "CA" in r]
        ca, cb, n, c = _residue_coords(residues)
        states = encoder.encode_atoms(ca, cb, n, c)
        letters = np.array(list(encoder.build_sequence(states).lower()))
        letters[np.ma.getmaskarray(states)] = MASK_3DI
        if plddt_mask:
            plddt = np.array([r["CA"].get_bfactor() for r in residues])
            letters[plddt < PLDDT_THRESHOLD] = MASK_3DI
        out.append("".join(letters))
    return out
