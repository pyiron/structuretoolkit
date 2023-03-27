import numpy as np
from ase.build import bulk, surface
from structuretoolkit.analyse.symmetry import get_symmetry
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.io.ase import AseAtomsAdaptor


def high_index_surface_info(
        element,
        crystal_structure,
        lattice_constant,
        terrace_orientation=None,
        step_orientation=None,
        kink_orientation=None,
        step_down_vector=None,
        length_step=3,
        length_terrace=3,
        length_kink=1,
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
    step_orientation = (
        step_orientation if step_orientation is not None else [1, 1, 0]
    )
    kink_orientation = (
        kink_orientation if kink_orientation is not None else [1, 1, 1]
    )
    step_down_vector = (
        step_down_vector if step_down_vector is not None else [1, 1, 0]
    )
    basis = bulk(
        name=element,
        crystalstructure=crystal_structure,
        a=lattice_constant,
        cubic=True
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
        raise ValueError(
            "Step orientation vector should lie in terrace.\
        For the given choice I could not find any symmetrically equivalent vector that lies in the terrace.\
        please change the stepOrientation and try again"
        )
    if len(eqvdirk_ind) == 0:
        raise ValueError(
            "Kink orientation vector should lie in terrace.\
        For the given choice I could not find any symmetrically equivalent vector that lies in the terrace.\
        please change the kinkOrientation and try again"
        )
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
    vec2 = (
               np.asanyarray(fin_kink_orientation).dot(length_terrace)
           ) + step_down_vector
    high_index_surface = np.cross(np.asanyarray(vec1), np.asanyarray(vec2))
    high_index_surface = np.array(
        high_index_surface / np.gcd.reduce(high_index_surface), dtype=int
    )

    return high_index_surface, fin_kink_orientation, fin_step_orientation


def high_index_surface(
        element,
        crystal_structure,
        lattice_constant,
        terrace_orientation=None,
        step_orientation=None,
        kink_orientation=None,
        step_down_vector=None,
        length_step=3,
        length_terrace=3,
        length_kink=1,
        layers=60,
        vacuum=10,
):
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
        slab: pyiron_atomistics.atomistics.structure.atoms.Atoms instance Required surface
    """
    basis = bulk(
        name=element,
        crystalstructure=crystal_structure,
        a=lattice_constant,
        cubic=True
    )
    high_index_surface, _, _ = high_index_surface_info(
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
    adapter = AseAtomsAdaptor()
    sga = SpacegroupAnalyzer(adapter.get_structure(atoms=surf))
    pmg_refined = sga.get_refined_structure()
    slab = adapter.get_atoms(structure=pmg_refined)
    slab.positions[:, 2] = slab.positions[:, 2] - np.min(slab.positions[:, 2])
    slab.set_pbc(True)
    return slab
