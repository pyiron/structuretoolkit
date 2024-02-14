def calculate_soap_descriptor_per_atom(
    structure,
    r_cut=None,
    n_max=None,
    l_max=None,
    sigma=1.0,
    rbf="gto",
    weighting=None,
    average="off",
    compression={"mode": "off", "species_weighting": None},
    species=None,
    periodic=True,
    sparse=False,
    dtype="float64",
    centers=None,
    n_jobs=1,
    only_physical_cores=False,
    verbose=False,
):
    from dscribe.descriptors import SOAP

    if species is None:
        species = list(set(structure.get_chemical_symbols()))
    periodic_soap = SOAP(
        r_cut=r_cut,
        n_max=n_max,
        l_max=l_max,
        sigma=sigma,
        rbf=rbf,
        weighting=weighting,
        average=average,
        compression=compression,
        species=species,
        periodic=periodic,
        sparse=sparse,
        dtype=dtype,
    )
    return periodic_soap.create(
        system=structure,
        centers=centers,
        n_jobs=n_jobs,
        only_physical_cores=only_physical_cores,
        verbose=verbose,
    )
