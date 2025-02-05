import itertools
import random
import warnings
from collections.abc import Iterable
from multiprocessing import cpu_count
from typing import Optional, Union

import numpy as np
from ase.atoms import Atoms
from ase.data import atomic_numbers


def chemical_formula(atoms: Atoms) -> str:
    """
    Generate the chemical formula of an Atoms object.

    Args:
        atoms (Atoms): The Atoms object representing the structure.

    Returns:
        str: The chemical formula of the structure.

    """

    def group_symbols():
        for species, same in itertools.groupby(atoms.get_chemical_symbols()):
            num_same = len(list(same))
            yield species if num_same == 1 else f"{species}{num_same}"

    return "".join(group_symbols())


def map_dict(f, d: dict) -> dict:
    """
    Apply a function to each value in a dictionary.

    Args:
        f: The function to apply.
        d (Dict): The dictionary to apply the function to.

    Returns:
        Dict: The dictionary with the function applied to each value.

    """
    return {k: f(v) for k, v in d.items()}


def mole_fractions_to_composition(
    mole_fractions: dict[str, float], num_atoms: int
) -> dict[str, int]:
    """
    Convert mole fractions to composition.

    Args:
        mole_fractions (Dict[str, float]): The mole fractions of each species.
        num_atoms (int): The total number of atoms.

    Returns:
        Dict[str, int]: The composition of each species.

    Raises:
        ValueError: If the sum of mole fractions is not within the range (1 - 1/num_atoms, 1 + 1/num_atoms).

    """
    if not (1.0 - 1 / num_atoms) < sum(mole_fractions.values()) < (1.0 + 1 / num_atoms):
        raise ValueError(
            f"mole-fractions must sum up to one: {sum(mole_fractions.values())}"
        )

    composition = map_dict(lambda x: x * num_atoms, mole_fractions)
    # check to avoid partial occupation -> x_i * num_atoms is not an integer number
    if any(
        not float.is_integer(round(occupation, 1))
        for occupation in composition.values()
    ):
        # at least one of the specified species exhibits fractional occupation, we try to fix it by rounding
        composition_ = map_dict(lambda occ: int(round(occ)), composition)
        warnings.warn(
            f"The current mole-fraction specification cannot be applied to {num_atoms} atoms, "
            "as it would lead to fractional occupation. Hence, I have changed it from "
            f'"{mole_fractions}" -> "{map_dict(lambda n: n / num_atoms, composition_)}"',
            stacklevel=2,
        )
        composition = composition_

    # due to rounding errors there might be a difference of one atom
    actual_atoms = sum(composition.values())
    diff = actual_atoms - num_atoms
    if abs(diff) == 1:
        # it is not possible to distribute atoms equally e.g x_a = x_b = x_c = 1/3 on 32 atoms
        # we remove one randomly bet we inform the user
        removed_species = random.choice(tuple(composition))
        composition[removed_species] -= 1
        warnings.warn(
            f'It is not possible to distribute the species properly. Therefore one "{removed_species}" atom was removed. '
            "This changes the input mole-fraction specification. "
            f'"{mole_fractions}" -> "{map_dict(lambda n: n / num_atoms, composition)}"',
            stacklevel=2,
        )
    elif abs(diff) > 1:
        # something else is wrong with the mole-fractions input
        raise ValueError(f"Cannot interpret mole-fraction dict {mole_fractions}")

    return composition


def remap_sro(species: Iterable[str], array: np.ndarray) -> dict[str, list]:
    """
    Remap computed short-range order parameters to the style of sqsgenerator=v0.0.5.

    Args:
        species (Iterable[str]): The species in the structure.
        array (np.ndarray): The computed short-range order parameters.

    Returns:
        Dict[str, list]: The remapped short-range order parameters.

    """
    species = tuple(sorted(species, key=lambda abbr: atomic_numbers[abbr]))
    return {
        f"{si}-{sj}": array[:, i, j].tolist()
        for (i, si), (j, sj) in itertools.product(
            enumerate(species), enumerate(species)
        )
        if j >= i
    }


def remap_sqs_results(
    result: dict[str, Union[Atoms, np.ndarray]],
) -> tuple[Atoms, dict[str, list]]:
    """
    Remap the results of SQS optimization.

    Args:
        result (Dict[str, Union[Atoms, np.ndarray]]): The result of SQS optimization.

    Returns:
        Tuple[Atoms, Dict[str, list]]: The remapped structure and short-range order parameters.

    """
    return result["structure"], remap_sro(
        set(result["structure"].get_chemical_symbols()), result["parameters"]
    )


def transpose(it: Iterable[Iterable]) -> Iterable[tuple]:
    """
    Transpose an iterable of iterables.

    Args:
        it (Iterable[Iterable]): The iterable to transpose.

    Returns:
        Iterable[tuple]: The transposed iterable.

    """
    return zip(*it)


def sqs_structures(
    structure: Atoms,
    mole_fractions: dict[str, Union[float, int]],
    weights: Optional[dict[int, float]] = None,
    objective: Union[float, np.ndarray] = 0.0,
    iterations: Union[float, int] = 1e6,
    output_structures: int = 10,
    mode: str = "random",
    num_threads: Optional[int] = None,
    prefactors: Optional[Union[float, np.ndarray]] = None,
    pair_weights: Optional[np.ndarray] = None,
    rtol: Optional[float] = None,
    atol: Optional[float] = None,
    which: Optional[Iterable[int]] = None,
    shell_distances: Optional[Iterable[int]] = None,
    minimal: Optional[bool] = True,
    similar: Optional[bool] = True,
    return_statistics: Optional[bool] = False,
) -> Union[Atoms, tuple[Atoms, dict[str, list], int, float]]:
    """
    Generate SQS structures.

    Args:
        structure (Atoms): The initial structure.
        mole_fractions (Dict[str, Union[float, int]]): The mole fractions of each species.
        weights (Optional[Dict[int, float]]): The weights for each shell.
        objective (Union[float, np.ndarray]): The target objective value.
        iterations (Union[float, int]): The number of iterations.
        output_structures (int): The number of output structures.
        mode (str): The mode for selecting configurations.
        num_threads (Optional[int]): The number of threads to use.
        prefactors (Optional[Union[float, np.ndarray]]): The prefactors for each shell.
        pair_weights (Optional[np.ndarray]): The pair weights.
        rtol (Optional[float]): The relative tolerance.
        atol (Optional[float]): The absolute tolerance.
        which (Optional[Iterable[int]]): The indices of the shells to optimize.
        shell_distances (Optional[Iterable[int]]): The distances for each shell.
        minimal (Optional[bool]): Whether to minimize the objective function.
        similar (Optional[bool]): Whether to generate similar structures.
        return_statistics (Optional[bool]): Whether to return additional statistics.

    Returns:
        Union[Atoms, Tuple[Atoms, Dict[str, list], int, float]]: The generated structures or a tuple containing the structures, short-range order parameters breakdown, number of iterations, and average cycle time.

    """
    from sqsgenerator import sqs_optimize

    composition = mole_fractions_to_composition(mole_fractions, len(structure))

    settings = {
        "atol": atol,
        "rtol": rtol,
        "mode": mode,
        "which": which,
        "structure": structure,
        "prefactors": prefactors,
        "shell_weights": weights,
        "iterations": iterations,
        "composition": composition,
        "pair_weights": pair_weights,
        "target_objective": objective,
        "shell_distances": shell_distances,
        "threads_per_rank": num_threads or cpu_count(),
        "max_output_configurations": output_structures,
    }
    # not specifying a parameter in settings causes sqsgenerator to choose a "sensible" default,
    # hence we remove all entries with a None value

    results, timings = sqs_optimize(
        {param: value for param, value in settings.items() if value is not None},
        minimal=minimal,
        similar=similar,
        make_structures=True,
        structure_format="ase",
    )

    structures, sro_breakdown = transpose(map(remap_sqs_results, results.values()))
    num_iterations = iterations
    cycle_time = np.average(list(map_dict(np.average, timings).values()))
    if not return_statistics:
        return structures
    else:
        return structures, sro_breakdown, num_iterations, cycle_time
