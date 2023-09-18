from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.analysis.local_env import VoronoiNN
from pymatgen.core import Structure, Element
import numpy as np
import pandas as pd

def get_stats(property_list, property_str):
    """
    Calculate statistical properties of a list of values.
    Parameters:
    property_list (list): A list of numerical values for which statistics are calculated.
    property_str (str): A string prefix to be used in the resulting statistical property names.
    Returns:
    dict: A dictionary containing statistical properties with keys in the format:
          "{property_str}_{statistic}" where statistic can be "std" (standard deviation),
          "mean" (mean), "min" (minimum), and "max" (maximum).
    Example:
    >>> values = [1, 2, 3, 4, 5]
    >>> get_stats(values, "example")
    {'example_std': 1.4142135623730951,
     'example_mean': 3.0,
     'example_min': 1,
     'example_max': 5}
    """
    return {
        f"{property_str}_std": np.std(property_list),
        f"{property_str}_mean": np.mean(property_list),
        f"{property_str}_min": np.min(property_list),
        f"{property_str}_max": np.max(property_list)
    }

def VoronoiSiteFeaturiser(structure, site):
    """
    Calculate various Voronoi-related features for a specific site in a crystal structure.
    Parameters:
    structure (ase.Atoms or pymatgen.Structure): The crystal structure.
    site (int): The index of the site in the crystal structure.
    Returns:
    pandas.DataFrame: A DataFrame containing computed Voronoi features for the specified site.
                      Columns include VorNN_CoordNo, VorNN_tot_vol, VorNN_tot_area, as well as
                      statistics for volumes, vertices, areas, and distances.
    Example:
    >>> from pymatgen import Structure
    >>> structure = Structure.from_file("example.cif")
    >>> VoronoiSiteFeaturiser(structure, 0)
                VorNN_CoordNo  VorNN_tot_vol  VorNN_tot_area  volumes_std  volumes_mean  ...
    0  7.0  34.315831  61.556747   10.172586     34.315831  ...
    """
    structure = AseAtomsAdaptor().get_structure(structure)
    coord_no = VoronoiNN().get_cn(structure=structure, n=site)
    site_info_dict = VoronoiNN().get_voronoi_polyhedra(structure, site)
    volumes = [site_info_dict[polyhedra]["volume"] for polyhedra in site_info_dict]
    vertices = [site_info_dict[polyhedra]["n_verts"] for polyhedra in site_info_dict]
    distances = [site_info_dict[polyhedra]["face_dist"] for polyhedra in site_info_dict]
    areas = [site_info_dict[polyhedra]["area"] for polyhedra in site_info_dict]

    total_area = np.sum(areas)
    total_volume = np.sum(volumes)

    data = {
        "VorNN_CoordNo": coord_no,
        "VorNN_tot_vol": total_volume,
        "VorNN_tot_area": total_area
    }

    data_str_list = ["volumes", "vertices", "areas", "distances"]

    for i, value_list in enumerate([volumes, vertices, areas, distances]):
        stats = get_stats(value_list, f"VorNN_{data_str_list[i]}")
        data.update(stats)

    df = pd.DataFrame(data, index=[site])
    return df