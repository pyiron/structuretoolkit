from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
from ase.atoms import Atoms
from ase.build import bulk, surface

from structuretoolkit.analyse import get_symmetry
from structuretoolkit.analyse.symmetry import Symmetry
from structuretoolkit.common.pymatgen import ase_to_pymatgen, pymatgen_to_ase


def get_high_index_surface_info(
    element: str,
    crystal_structure: str,
    lattice_constant: float,
    terrace_orientation: list | None = None,
    step_orientation: list | None = None,
    kink_orientation: list | None = None,
    step_down_vector: list | None = None,
    length_step: int = 3,
    length_terrace: int = 3,
    length_kink: int = 1,
):
    """
    Gives the miller indices of high index surface required to create a stepped and kink surface, based
    on the general orientation and length of terrace, step and kinks respectively. The microfacet notation used is
    based on the work of Van Hove et al.,[1].

    [1] Van Hove, M. A., and G. A. Somorjai. "A new microfacet notation for high-Miller-index surfaces of cubic
    materials with terrace, step and kink structures." Surface Science 92.2-3 (1980): 489-518.

    Args:
        element (str): The parent element eq. "N", "O", "Mg" etc.
        crystal_structure (str): The crystal structure of the lattice
        lattice_constant (float): The lattice constant
        terrace_orientation (list): The miller index of the terrace eg., [1,1,1]
        step_orientation (list): The miller index of the step eg., [1,1,0]
        kink_orientation (list): The miller index of the kink eg., [1,1,1]
        step_down_vector (list): The direction for stepping down from the step to next terrace eg., [1,1,0]
        length_terrace (int): The length of the terrace along the kink direction in atoms eg., 3
        length_step (int): The length of the step along the step direction in atoms eg., 3
        length_kink (int): The length of the kink along the kink direction in atoms eg., 1


    Returns:
        high_index_surface: The high miller index surface which can be used to create slabs
        fin_kink_orientation: The kink orientation lying in the terrace
        fin_step_orientation: The step orientation lying in the terrace
    """
    terrace_orientation = (
        terrace_orientation if terrace_orientation is not None else [1, 1, 1]
    )
    step_orientation = step_orientation if step_orientation is not None else [1, 1, 0]
    kink_orientation = kink_orientation if kink_orientation is not None else [1, 1, 1]
    step_down_vector = step_down_vector if step_down_vector is not None else [1, 1, 0]
    basis = bulk(
        name=element, crystalstructure=crystal_structure, a=lattice_constant, cubic=True
    )
    sym = get_symmetry(structure=basis)
    eqvdirs = np.unique(
        np.matmul(sym.rotations[:], (np.array(step_orientation))), axis=0
    )
    eqvdirk = np.unique(
        np.matmul(sym.rotations[:], (np.array(kink_orientation))), axis=0
    )
    eqvdirs_ind = np.where(np.dot(np.squeeze(eqvdirs), terrace_orientation) == 0)[0]
    eqvdirk_ind = np.where(np.dot(np.squeeze(eqvdirk), terrace_orientation) == 0)[0]
    if len(eqvdirs_ind) == 0:
        raise ValueError("Step orientation vector should lie in terrace.\
        For the given choice I could not find any symmetrically equivalent vector that lies in the terrace.\
        please change the stepOrientation and try again")
    if len(eqvdirk_ind) == 0:
        raise ValueError("Kink orientation vector should lie in terrace.\
        For the given choice I could not find any symmetrically equivalent vector that lies in the terrace.\
        please change the kinkOrientation and try again")
    temp = (
        (np.cross(np.squeeze(eqvdirk[eqvdirk_ind[0]]), np.squeeze(eqvdirs)))
        .tolist()
        .index(terrace_orientation)
    )
    fin_kink_orientation = eqvdirk[eqvdirk_ind[0]]
    fin_step_orientation = eqvdirs[temp]
    vec1 = (np.asanyarray(fin_step_orientation).dot(length_step)) + (
        np.asanyarray(fin_kink_orientation).dot(length_kink)
    )
    vec2 = (np.asanyarray(fin_kink_orientation).dot(length_terrace)) + step_down_vector
    high_index_surface = np.cross(np.asanyarray(vec1), np.asanyarray(vec2))
    high_index_surface = np.array(
        high_index_surface / np.gcd.reduce(high_index_surface), dtype=int
    )

    return high_index_surface, fin_kink_orientation, fin_step_orientation


def high_index_surface(
    element: str,
    crystal_structure: str,
    lattice_constant: float,
    terrace_orientation: list | None = None,
    step_orientation: list | None = None,
    kink_orientation: list | None = None,
    step_down_vector: list | None = None,
    length_step: int = 3,
    length_terrace: int = 3,
    length_kink: int = 1,
    layers: int = 60,
    vacuum: int = 10,
) -> Atoms:
    """
    Gives a slab positioned at the bottom with the high index surface computed by high_index_surface_info().
    Args:
        element (str): The parent element eq. "N", "O", "Mg" etc.
        crystal_structure (str): The crystal structure of the lattice
        lattice_constant (float): The lattice constant
        terrace_orientation (list): The miller index of the terrace. default: [1,1,1]
        step_orientation (list): The miller index of the step. default: [1,1,0]
        kink_orientation (list): The miller index of the kink. default: [1,1,1]
        step_down_vector (list): The direction for stepping down from the step to next terrace. default: [1,1,0]
        length_terrace (int): The length of the terrace along the kink direction in atoms. default: 3
        length_step (int): The length of the step along the step direction in atoms. default: 3
        length_kink (int): The length of the kink along the kink direction in atoms. default: 1
        layers (int): Number of layers of the high_index_surface. default: 60
        vacuum (float): Thickness of vacuum on the top of the slab. default:10

    Returns:
        slab: ase.atoms.Atoms instance Required surface
    """
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

    basis = bulk(
        name=element, crystalstructure=crystal_structure, a=lattice_constant, cubic=True
    )
    high_index_surface, _, _ = get_high_index_surface_info(
        element=element,
        crystal_structure=crystal_structure,
        lattice_constant=lattice_constant,
        terrace_orientation=terrace_orientation,
        step_orientation=step_orientation,
        kink_orientation=kink_orientation,
        step_down_vector=step_down_vector,
        length_step=length_step,
        length_terrace=length_terrace,
        length_kink=length_kink,
    )
    surf = surface(basis, high_index_surface, layers, vacuum)
    slab = pymatgen_to_ase(
        SpacegroupAnalyzer(ase_to_pymatgen(structure=surf)).get_refined_structure()
    )
    slab.positions[:, 2] = slab.positions[:, 2] - np.min(slab.positions[:, 2])
    slab.set_pbc(True)
    return slab


def find_inplane_directions(
    basis,
    terrace_orientation=None,
    step_orientation=None,
    kink_orientation=None,
):
    """
    Author: based on code by Shyam Katnagallu
    Find symmetry-equivalent directions of step_direction and kink_direction lying in plane
    given by terrace_orientation
    Args:
        basis: some atomic structure
        terrace_orientation (list): The miller index of the terrace eg., [1,1,1]
        step_orientation (list): The miller index of the step eg., [1,1,0]
        kink_orientation (list): The miller index of the kink eg., [1,1,1]
    """
    terrace_orientation = list(terrace_orientation)
    step_orientation = list(step_orientation)
    kink_orientation = list(kink_orientation)
    sym = Symmetry(basis)
    eqvdirs = np.unique(
        np.matmul(sym.rotations[:], (np.array(step_orientation))), axis=0
    )
    eqvdirk = np.unique(
        np.matmul(sym.rotations[:], (np.array(kink_orientation))), axis=0
    )
    eqvdirs_ind = np.where(np.dot(np.squeeze(eqvdirs), terrace_orientation) == 0)[0]
    eqvdirk_ind = np.where(np.dot(np.squeeze(eqvdirk), terrace_orientation) == 0)[0]
    if len(eqvdirs_ind) == 0:
        raise ValueError("Step orientation vector should lie in terrace.\
        For the given choice I could not find any symmetrically equivalent vector that lies in the terrace.\
        please change the stepOrientation and try again")
    if len(eqvdirk_ind) == 0:
        raise ValueError("Kink orientation vector should lie in terrace.\
        For the given choice I could not find any symmetrically equivalent vector that lies in the terrace.\
        please change the kinkOrientation and try again")
    crossp = np.cross(np.squeeze(eqvdirk[eqvdirk_ind[0]]), np.squeeze(eqvdirs)).tolist()
    if terrace_orientation in crossp:
        # fast search
        temp = crossp.index(terrace_orientation)
    else:
        # check for same orientations (normalized)
        tn = terrace_orientation / np.linalg.norm(terrace_orientation)
        for i, cp in enumerate(crossp):
            cn = cp / np.linalg.norm(cp)
            if np.abs(np.dot(tn, cn) - 1.0) < 1e-8:
                temp = i
                break
    fin_kink_orientation = eqvdirk[eqvdirk_ind[0]]
    fin_step_orientation = eqvdirs[temp]
    return fin_step_orientation, fin_kink_orientation


def print_fractional_positions(atoms, S, K, T):
    """
    Print fractional atomic positions in a custom basis defined by:
        b1 = S, b2 = K, b3 = T
    where S, K, T are integer vectors (in the original cell basis: [s1, s2, s3] means s1*a1 + s2*a2 + s3*a3).

    The new cell is formed as: new_cell = cell @ [S, K, T] (matrix product).

    Output is sorted by x3 (depth), then x1, then x2.

    Parameters:
    -----------
    atoms : ase.Atoms
        Input crystal structure.
    S, K, T : array-like, shape (3,)
        Integer vectors (indices) in the original cell basis.
    """
    # Convert to numpy arrays
    S, K, T = np.array(S, dtype=int), np.array(K, dtype=int), np.array(T, dtype=int)

    # Pack into a 3x3 matrix: columns are S, K, T
    basis_indices = np.column_stack([S, K, T])

    # Compute new cell in Cartesian space: cell @ basis_indices
    cell = atoms.get_cell()
    new_cell = cell @ basis_indices

    # Compute inverse of new cell
    try:
        new_cell_inv = np.linalg.inv(new_cell)
    except np.linalg.LinAlgError:
        raise ValueError(
            "New cell matrix is singular — S, K, T are linearly dependent."
        )

    # Get atomic positions (Natoms x 3)
    positions = atoms.get_positions()

    # Compute fractional coordinates (Natoms x 3)
    frac_coords = positions @ new_cell_inv

    # Extract x1, x2, x3 (columns)
    x1, x2, x3 = frac_coords[:, 0], frac_coords[:, 1], frac_coords[:, 2]

    # Create list of (x3, x1, x2, atom_index) and sort
    sorted_data = sorted(
        zip(x3, x1, x2, range(len(atoms)), strict=True),
        key=lambda x: (x[0], x[1], x[2]),
    )

    # Print results
    print("Fractional positions (x1, x2, x3) in basis (S, K, T), sorted by x3, x1, x2:")
    print("x1     x2     x3     atom_index")
    print("-" * 40)
    for x3_val, x1_val, x2_val, idx in sorted_data:
        print(
            f"{x1_val:6.3f}  {x2_val:6.3f}  {x3_val:6.3f}  {atoms.symbols[idx]:2s}  {idx:4d}"
        )


# ----------------------------------------------------------------------
# Helper: obtain HNF and the left‑transform U  (P = U·H)
# ----------------------------------------------------------------------
def _hnf_and_U(P):
    """
    Return (H, U) where H is the Hermite normal form of the integer matrix P
    and U is unimodular such that P = U·H.
    """
    from sympy import Matrix

    P_sym = Matrix(P.tolist())
    # The functional interface works for all recent SymPy releases.
    from sympy.matrices.normalforms import hermite_normal_form

    H_sym = hermite_normal_form(P_sym)
    H = np.array(H_sym, dtype=int)
    U = np.asarray(H_sym.inv() @ P_sym, dtype=int)

    return H, U


# ----------------------------------------------------------------------
# auxiliary functions: test for cubic and fcc/bcc lattices
# ----------------------------------------------------------------------
def _is_cubic(cell: np.ndarray | ase.cell.Cell) -> bool:
    """
    Check if a cell has cubic symmetry.

    Returns True if the cell's symmetry group has 48 operations and no translational components.
    """
    sym = Symmetry(
        Atoms(
            symbols="H",
            cell=cell,
            positions=[
                3 * [0],
            ],
        )
    )
    return bool(len(sym.rotations) == 48 and np.linalg.norm(sym.translations) < 1e-6)


def _is_cubic_nonsimple(cell: np.ndarray | ase.cell.Cell) -> bool:
    """
    Check if a cubic cell is non-simple (e.g., FCC or BCC).

    Returns True if the cell is cubic and has non-orthogonal lattice vectors.
    """
    if not _is_cubic(cell):
        return False
    for i in range(3):
        if np.dot(cell[i], cell[i - 1]) > 1e-6 * np.linalg.norm(
            cell[i]
        ) * np.linalg.norm(cell[i - 1]):
            return True
    return False


# ----------------------------------------------------------------------
# Main routine
# ----------------------------------------------------------------------
def make_supercell(primitive: Atoms, P: np.ndarray) -> Atoms:
    """
    Build a bulk super‑cell from a primitive cell using the integer
    transformation matrix ``P`` (the same matrix that ASE’s built‑in
    ``make_supercell`` expects).  The routine does **not** rotate the cell;
    the new lattice vectors are exactly ``primitive.cell @ P``.

    Parameters
    ----------
    primitive
        ASE ``Atoms`` object that contains the primitive cell.
    P
        3×3 integer matrix (columns are the new basis vectors expressed in the
        primitive basis).  ``P`` must be non‑singular.

    Returns
    -------
    supercell : ase.Atoms
        The enlarged cell with the requested ordering.

    atom order:
        ``'atom-major'`` – loop over primitive atoms first, then over the
        translations.

    """

    # ------------------------------------------------------------------
    # 1.  Primitive‑cell data
    # ------------------------------------------------------------------
    cell_prim = primitive.get_cell()  # (3,3) Cartesian
    symbols_prim = primitive.get_chemical_symbols()
    frac_prim = primitive.get_scaled_positions()  # (N0,3) in [0,1)

    # ------------------------------------------------------------------
    # 2.  Check the transformation matrix
    # ------------------------------------------------------------------
    P = np.asarray(P, dtype=int)
    if P.shape != (3, 3):
        raise ValueError("Transformation matrix P must be 3×3.")
    det = int(round(np.linalg.det(P)))
    if det == 0:
        raise ValueError("Transformation matrix P is singular.")
    n_cells = abs(det)  # number of copies
    print(f"ncells={n_cells}")

    # ------------------------------------------------------------------
    # 3.  Hermite normal form → easy enumeration of translation vectors
    # ------------------------------------------------------------------
    H, U = _hnf_and_U(P)  # H is upper‑triangular, U unimodular
    # diagonal entries of H give the limits of the three nested loops
    h11, h22, h33 = H[0, 0], H[1, 1], H[2, 2]

    # ------------------------------------------------------------------
    # 4.  Build the list of integer translation vectors (in the primitive basis)
    # ------------------------------------------------------------------
    trans = []
    for i in range(h11):
        for j in range(h22):
            for k in range(h33):
                # t = U·[i, j, k]^T  (still integer because U is unimodular)
                t = np.array([i, j, k], dtype=int) @ U
                trans.append(t)
    trans = np.asarray(trans, dtype=int)  # (n_cells, 3)

    # ------------------------------------------------------------------
    # 5.  New lattice vectors (Cartesian)
    # ------------------------------------------------------------------
    # print(P)
    cell_super = P @ cell_prim  # (3,3)

    # print(cell_super)
    # ------------------------------------------------------------------
    # 6.  Fractional coordinates in the super‑cell
    # ------------------------------------------------------------------
    # Broadcast: (N0,1,3) + (1,n_cells,3) → (N0,n_cells,3)
    new_frac = frac_prim[:, None, :] + trans[None, :, :]
    new_frac = new_frac.reshape(-1, 3)  # (N0*n_cells, 3)
    new_pos = new_frac @ cell_prim  # (N,3)

    new_frac = new_pos @ np.linalg.inv(cell_super)
    new_frac %= 1.0  # bring into [0,1)
    new_pos = new_frac @ cell_super

    # ------------------------------------------------------------------
    # 7.  Cartesian positions
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # 8.  Symbol list according to the requested ordering
    # ------------------------------------------------------------------
    # for each primitive atom we repeat all translations
    symbols = np.repeat(symbols_prim, n_cells)

    # ------------------------------------------------------------------
    # 9.  Assemble the ASE Atoms object
    # ------------------------------------------------------------------
    supercell = Atoms(
        symbols=symbols.tolist(),
        positions=new_pos,
        cell=cell_super,
        pbc=True,
    )

    return supercell


def bulk_supercell_with_mapping(
    atoms, S, K, T, n, m, t, l, kink_length=1, step_depth=1
):
    """
    Build a bulk super‑cell for a poly‑atomic crystal and return a deterministic
    mapping from each atom in the super‑cell to the atom index in the original
    primitive cell.

    Parameters
    ----------
    atoms : ase.Atoms
        Primitive bulk structure.
    S, K, T : array‑like, shape (3,)
        Integer direction vectors expressed in the original lattice basis.
    n, m, t, l : int
        Terrace size (n × m), step‑down offset (t) and number of layers (l).

    Returns
    -------
    supercell      : ase.Atoms
        Bulk super‑cell with lattice vectors
            b1 = n·S + K
            b2 = t·S + m·K + D   ( D = t·S – T )
            b3 = l·T
    mapping_cell   : (3, 3) ndarray
        Cartesian lattice vectors of the *mapping* cell
            b1′ = n·S + K
            b2′ = t·S + m·K
            b3′ = T
    frac_coords    : (N, 3) ndarray
        Fractional coordinates of every atom of ``supercell`` expressed in the
        mapping cell (0 ≤ x1,x2 < 1, 0 ≤ x3 < l).
    orig_index     : (N,) ndarray of int
        For each atom in ``supercell`` the index (0‑based) of the atom in the
        original ``atoms`` object from which it originated.
    """

    # ------------------------------------------------------------------ #
    # 1.  Integer direction vectors
    # ------------------------------------------------------------------ #
    S = np.asarray(S, dtype=int)
    K = np.asarray(K, dtype=int)
    T = np.asarray(T, dtype=int)

    # ------------------------------------------------------------------ #
    # 2.  Super‑cell transformation matrix (columns expressed in the old basis)
    # ------------------------------------------------------------------ #
    v1 = n * S + kink_length * K  # b1
    D = t * S - step_depth * T  # step‑down vector
    v2 = m * K + D  # b2 = t S + m·K – T
    v3 = l * T  # b3
    P_super = np.vstack([v1, v2, v3])  # 3 × 3 integer matrix

    # ------------------------------------------------------------------ #
    # 3.  Build the bulk super‑cell with a deterministic ordering
    # ------------------------------------------------------------------ #
    supercell = make_supercell(atoms, P_super)

    # ------------------------------------------------------------------ #
    # 4.  Mapping cell (used only for fractional coordinates)
    # ------------------------------------------------------------------ #
    vm1 = n * S
    vm2 = m * K
    vm3 = T
    P_map = np.vstack([vm1, vm2, vm3])
    mapping_cell = P_map @ atoms.get_cell()  # Cartesian vectors

    # ------------------------------------------------------------------ #
    # 5.  Fractional coordinates of the super‑cell atoms in the mapping cell
    # ------------------------------------------------------------------ #
    pos_cart = supercell.get_positions()  # (N,3)
    inv_map = np.linalg.inv(mapping_cell)  # Cartesian → fractional
    frac = pos_cart @ inv_map  # (N,3)
    # ------------------------------------------------------------------ #
    # 6.  Construct the index mapping
    # ------------------------------------------------------------------ #
    n_prim = len(atoms)  # atoms in the primitive cell
    n_cells = int(round(abs(np.linalg.det(P_super))))  # number of translated cells

    # atom‑major: for each primitive atom we append all its images
    orig_index = np.repeat(np.arange(n_prim, dtype=int), n_cells)

    # ------------------------------------------------------------------ #
    # 7.  Return everything
    # ------------------------------------------------------------------ #
    return supercell, mapping_cell, frac, orig_index


def apply_fractional_filter(supercell, frac_coords, orig_index, filter_func):
    """
    Remove atoms from a super‑cell according to a user‑provided filter.

    Parameters
    ----------
    supercell : ase.Atoms
        Full super‑cell (as returned by ``bulk_supercell_with_mapping``).

    frac_coords : (N, 3) ndarray
        Fractional coordinates of each atom in the *mapping* cell
        (vectors n S+K, t S+m K, T).

    orig_index : (N,) ndarray of int
        Index of the atom in the original primitive cell for every atom
        of ``supercell``.

    filter_func : callable
        Signature ``keep = filter_func(frac, element, original_index)``.
        *frac* – length‑3 array (x1, x2, x3) in the mapping cell.
        *element* – chemical symbol string (e.g. ``'Al'``).
        *original_index* – integer index of the atom in the primitive cell.
        Must return ``True`` for atoms that should be kept.

    Returns
    -------
    filtered_sc : ase.Atoms
        Super‑cell after the filter has been applied.

    filtered_orig_index : (N_keep,) ndarray of int
        Original‑cell indices corresponding to the atoms that remain.
    """
    # Boolean mask built with a list‑comprehension + zip
    keep_mask = np.array(
        [
            filter_func(frac, atom.symbol, idx)
            for frac, atom, idx in zip(frac_coords, supercell, orig_index, strict=True)
        ],
        dtype=bool,
    )

    # ASE supports boolean indexing directly
    filtered_sc = supercell[keep_mask]
    filtered_orig_index = orig_index[keep_mask]

    return filtered_sc, filtered_orig_index


def make_slab(supercell: Atoms, step_direction: np.ndarray, vacuum: float) -> Atoms:
    """
    Build a slab from a bulk super‑cell.

    Parameters
    ----------
    supercell : ase.Atoms
        Bulk super‑cell (already constructed).

    step_direction : array‑like, shape (3,)
        Step direction expressed in the *current* Cartesian frame of the
        supercell.  The vector does not need to be normalised.

    vacuum : float
        Thickness of the vacuum layer (Å) that will be added on the top side
        of the slab.

    Returns
    -------
    slab : ase.Atoms
        The rotated, vacuum‑padded slab with atoms wrapped into the new cell.
    """
    # ------------------------------------------------------------------ #
    # 1.  Gather current lattice vectors and normalise the step vector
    # ------------------------------------------------------------------ #
    a1, a2, a3 = supercell.get_cell()  # rows, shape (3,3)
    step = np.asarray(step_direction, dtype=float)

    if np.linalg.norm(step) < 1e-12:
        raise ValueError("Step direction vector is zero.")

    # ------------------------------------------------------------------ #
    # 2.  Construct the orthonormal target basis (x̂, ŷ, ẑ)
    # ------------------------------------------------------------------ #
    # 2.1  New ẑ – normal of the (a1, a2) plane
    z_vec = np.cross(a1, a2)
    if np.linalg.norm(z_vec) < 1e-12:
        raise ValueError(
            "Lattice vectors a1 and a2 are collinear; cannot define a surface normal."
        )
    z_hat = z_vec / np.linalg.norm(z_vec)
    if np.dot(z_hat, a3) < 0.0:
        z_hat = -z_hat

    # 2.2  New ŷ – step direction, orthogonalised to ẑ
    y_hat = step / np.linalg.norm(step)
    # Remove any component parallel to ẑ (numerical safety)
    y_hat = y_hat - np.dot(y_hat, z_hat) * z_hat
    if np.linalg.norm(y_hat) < 1e-12:
        raise ValueError("Step direction is parallel to the surface normal.")
    y_hat = y_hat / np.linalg.norm(y_hat)

    # 2.3  New x̂ – completes a right‑handed set
    x_hat = np.cross(y_hat, z_hat)  # already unit length because ŷ ⟂ ẑ

    # Ensure a right‑handed coordinate system (determinant +1)
    if np.linalg.det(np.column_stack([x_hat, y_hat, z_hat])) < 0:
        x_hat = -x_hat

    # ------------------------------------------------------------------ #
    # 3.  Assemble the rotation matrix (columns = new basis vectors)
    # ------------------------------------------------------------------ #
    R = np.column_stack([x_hat, y_hat, z_hat])  # 3×3 orthogonal matrix

    # ------------------------------------------------------------------ #
    # 4.  Apply the rotation manually (cell rows and atomic positions)
    # ------------------------------------------------------------------ #
    #   rows of the cell matrix are multiplied on the right by R.T
    cell_rotated = supercell.get_cell() @ R
    #   atomic positions are also rows → multiply on the right by R.T
    pos_rotated = supercell.get_positions() @ R

    # Update the Atoms object (do not scale atoms – we already have the new positions)
    slab = supercell.copy()
    slab.set_cell(cell_rotated, scale_atoms=False)
    slab.set_positions(pos_rotated)

    # ------------------------------------------------------------------ #
    # 5.  Insert vacuum: zero x‑ and y‑components of the third lattice vector
    #     and add the requested vacuum thickness to its z‑component
    # ------------------------------------------------------------------ #
    cell = slab.get_cell()
    cell[2, :2] = 0.0  # a3_{x,y}=0
    cell[:2, 2] = 0.0  # a{1,2}_z =0
    cell[2, 2] += float(vacuum)  # extend along z

    slab.set_cell(cell, scale_atoms=False)

    # ------------------------------------------------------------------ #
    # 6.  Wrap atoms back into the new cell
    # ------------------------------------------------------------------ #
    slab.wrap()  # ASE method that brings all positions into [0,1) in the new cell

    # ------------------------------------------------------------------ #
    # 7.  Return the slab
    # ------------------------------------------------------------------ #
    return slab


def create_slab(
    bulk_str: Atoms,
    vacuum_size: float,
    terrace_orientation: Sequence[int] | np.ndarray,
    thickness: int,
    step_orientation: Sequence[int] | np.ndarray,
    step_length: int,
    kink_orientation: Sequence[int] | np.ndarray,
    terrace_width: int,
    kink_length: int,
    kink_shift: int,
    step_depth: int,
    filter_function: (
        Callable[[Sequence[float] | np.ndarray, str, int], bool] | None
    ) = None,
) -> tuple[Atoms, dict[int, int]]:
    """
    Build a surface slab from a bulk crystal structure with optional steps and kinks.

    Parameters
    ----------
    bulk_str : ase.Atoms
        Bulk crystal structure from which the slab will be cut.

    vacuum_size : float
        Thickness of the vacuum layer (Å) that will be appended to the slab after the
        surface normal direction is identified.

    terrace_orientation : Sequence[int] | np.ndarray
        Miller index that defines the direction of the terrace

    thickness : float
        Desired slab thickness in layers measured along the surface‑normal (terrace) direction .

    step_orientation : Sequence[int] | np.ndarray
        Miller index along the step edge.  Should be orthogonal to the terrace normal.
        If not, it will be replaced by a symmetry equivalent direction.

    step_length : int
        Number of repetitions along the step

    kink_orientation : Sequence[int] | np.ndarray
        Miller index for direction away from step within the terrace.  Should be orthogonal to the terrace normal.
        If not, it will be replaced by a symmetry equivalent direction.

    terrace_width : int
        Width of the terrace along the kink orientation.

    kink_length : int
        Length of the kink segment along ``kink_orientation``.

    kink_shift : int
        Lateral shift of the kink relative to the step edge.  Positive values move
        the kink in the direction of ``step_orientation``.
        This parameter allows to control the in-plane cell-shape.

    step_depth : int
        Depth of the step measured along the terrace normal.

    filter_function : Callable[[Sequence[float] | np.ndarray, str, int], bool],
        Function ``f(frac,symbol,index) -> bool`` that receives the fractional position of an
        atom (``frac``), the chemical symbol, and the index within the bulk structure
        and returns ``True`` if the atom should be kept in the final slab.

    Returns
    -------
    slab : ase.Atoms
        The final slab structure, rotated into the desired orientation and padded with
        a vacuum region of ``vacuum_size`` Å.

    idxmap : dict[int, int]
        Mapping from the atom indices of ``slab`` back to the original indices in
        ``bulk_str`` (after the fractional filter).  Useful for tracking which bulk
        atoms survived the construction process.

    Notes
    -----

    Example
    -------
    from ase.build import bulk
    from structuretoolkit.build.surface import create_slab
    cu = bulk('Cu', 'fcc', a=3.6)
    terrace = [1, 1, 1]          # (111) normal
    step    = [-1, 1, 0]         # in‑plane step direction
    kink    = [1, 1, -2]         # kink direction
    slab, idxmap = create_slab(
         bulk_str=cu,
         vacuum_size=15,
         terrace_orientation=terrace,
         thickness=2,
         step_orientation=step,
         step_length=3,
         kink_orientation=kink,
         terrace_width=4,
         kink_length=1,
         kink_shift=0,
         step_depth=1,
    )
    slab.get_volume()
    """
    if filter_function is None:

        def filter_function(*args) -> bool:
            return True

    # --------------------------------------------------------------------- #
    # The implementation is unchanged – only the signature now carries
    # explicit type information.
    # --------------------------------------------------------------------- #
    S, K = find_inplane_directions(
        bulk_str, terrace_orientation, step_orientation, kink_orientation
    )

    if np.dot(step_orientation, terrace_orientation) == 0:
        S = step_orientation
    if np.dot(kink_orientation, terrace_orientation) == 0:
        K = kink_orientation

    # create an appropriate bulk supercell
    supercell, mapcell, frac, idxmap = bulk_supercell_with_mapping(
        bulk_str,
        S,
        K,
        terrace_orientation,
        step_length,
        terrace_width,
        kink_shift,
        thickness,
        kink_length,
        step_depth,
    )

    # clean up the surface from unwanted atoms
    slab, idxmapf = apply_fractional_filter(supercell, frac, idxmap, filter_function)

    # rotate cell into final orientation
    slab = make_slab(slab, S @ bulk_str.cell, vacuum_size)

    return slab, idxmap


def make_stepped_surface(
    bulk_str: Atoms,
    vacuum_size: float = 15,
    terrace_orientation: Sequence[int] | np.ndarray = (0, 0, 1),
    thickness: int = 4,
    step_orientation: Sequence[int] | np.ndarray = (1, 0, 0),
    step_length: int = 3,
    terrace_width: int = 3,
    filter_function: (
        Callable[[Sequence[float] | np.ndarray, str, int], bool] | None
    ) = None,
) -> Atoms:
    """
    Generate a stepped surface slab from a bulk crystal structure with a single step edge.

    This function constructs a surface slab with a defined terrace and a step, where the step
    is oriented along a specified direction and the terrace extends in a perpendicular direction.
    The kink direction is automatically computed as the cross product of the terrace and step
    orientations, ensuring orthogonality.

    Parameters
    ----------
    bulk_str : ase.Atoms
        The bulk crystal structure from which the slab will be derived.

    vacuum_size : float, optional
        Thickness of the vacuum layer (in Å) to be added above and below the slab along the
        surface normal direction. Default is 15 Å.

    terrace_orientation : Sequence[int] | np.ndarray, optional
        Miller index defining the surface normal (i.e., the direction perpendicular to the
        terrace plane). Default is (0, 0, 1), corresponding to a (001) surface.

    thickness : int, optional
        Number of atomic layers in the slab along the terrace normal direction. Default is 4.

    step_orientation : Sequence[int] | np.ndarray, optional
        Miller index defining the direction along the step edge (in-plane). Must be orthogonal
        to the terrace normal. If not, a symmetry-equivalent direction will be used. Default is (1, 0, 0).

    step_length : int, optional
        Number of repetitions of the step along the step_orientation direction. Controls the
        length of the step in the unit cell. Default is 3.

    terrace_width : int, optional
        Width of the terrace in atomic layers along the kink direction (perpendicular to the step).
        Default is 3.

    filter_function : Callable[[Sequence[float] | np.ndarray, str, int], bool] | None, optional
        A function that determines which atoms from the bulk are retained in the final slab.
        It takes three arguments: the fractional coordinates of the atom, its chemical symbol,
        and its index in the bulk. Returns True to keep the atom, False to discard it.
        If None, all atoms are kept. Default is None.

    Returns
    -------
    slab : ase.Atoms
        The resulting stepped surface slab, rotated to align with the specified terrace
        orientation, with vacuum padding and the step geometry applied.

    Example
    -------
    >>> from ase.build import bulk
    >>> from structuretoolkit.build.surface import make_stepped_surface
    >>> cu = bulk('Cu', 'fcc', a=3.6)
    >>> slab = make_stepped_surface(
    ...     bulk_str=cu,
    ...     vacuum_size=15,
    ...     terrace_orientation=(1, 1, 1),
    ...     thickness=2, # (111) direction has 3 sublayers => 6 sublayers
    ...     step_orientation=(1, -1, 0),
    ...     step_length=3,
    ...     terrace_width=4
    ... )
    >>> print(slab.get_volume())
    """
    if filter_function is None:

        def filter_function(*args) -> bool:
            return True

    if _is_cubic_nonsimple(bulk_str.cell):
        print("=== WARNING ===")
        print("This seems to be a cubic lattice, but not a conventional (cubic) cell.")
        print("- maybe a fcc or bcc primitive cell?")
        print(
            "The Miller indices you provide refer to the cell as given, NOT to the conventional one!"
        )
        print("===============")
    kink_orientation = np.cross(terrace_orientation, step_orientation)
    step_xyz = step_orientation @ bulk_str.cell
    kink_xyz = kink_orientation @ bulk_str.cell
    cosangle = np.dot(step_xyz, kink_xyz) / (
        np.linalg.norm(step_xyz) * np.linalg.norm(kink_xyz)
    )
    print(
        f"terrace={terrace_orientation} step={step_orientation} kink={kink_orientation}"
    )
    return create_slab(
        bulk_str,
        vacuum_size,
        terrace_orientation,
        thickness,
        step_orientation,
        step_length,
        kink_orientation,
        terrace_width,
        0,
        -int(cosangle * terrace_width),
        1,
        filter_function,
    )[0]
