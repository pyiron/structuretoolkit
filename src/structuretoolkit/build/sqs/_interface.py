from __future__ import annotations
from ase.atoms import Atoms
from threading import Event, Thread
from typing import overload, Literal, TypeVar, Generic, Any, Iterator

from ._types import (
    Composition,
    ShellWeights,
    ShellRadii,
    SublatticeMode,
    IterationMode,
    Prec,
    SqsResultInteract,
    SqsResultSplit,
    LogLevel,
)

R = TypeVar("R", SqsResultInteract, SqsResultSplit)
T = TypeVar("T")


class _SqsResultProxy(Generic[R]):
    def __init__(self, result: R):
        self._result = result

    def atoms(self) -> Atoms:
        from sqsgenerator import to_ase

        return to_ase(self._result.structure())

    def __getattr__(self, item: str) -> Any:
        return getattr(self._result, item)

    def sublattices(self) -> list[_SqsResultProxy[SqsResultInteract]]:
        from sqsgenerator.core import SqsResultSplitDouble, SqsResultSplitFloat

        if isinstance(self._result, (SqsResultSplitDouble, SqsResultSplitFloat)):
            return [
                _SqsResultProxy[SqsResultInteract](r)
                for r in self._result.sublattices()
            ]
        else:
            raise AttributeError(
                f"{type(self._result).__name__} has no attribute 'sublattices'"
            )


class SqsResultPack(Generic[R]):
    def __init__(self, pack):
        self._pack = pack

    def __len__(self) -> int:
        return len(self._pack)

    def best(self) -> _SqsResultProxy[R]:
        return _SqsResultProxy(self._pack.best())

    def num_objectives(self) -> int:
        return self._pack.num_objectives()

    def num_results(self) -> int:
        return self._pack.num_results()

    def __iter__(self) -> Iterator[_SqsResultProxy[R]]:
        for _, results in self._pack:
            for result in results:
                yield _SqsResultProxy(result)


def _ensure_list(v: T | list[T] | None) -> list[T] | None:
    if v is not None:
        return v if isinstance(v, list) else [v]
    else:
        return None


@overload
def sqs_structures(
    structure: Atoms,
    composition: Composition,
    supercell: tuple[int, int, int] | None = None,
    shell_weights: ShellWeights | None = None,
    shell_radii: ShellRadii | None = None,
    objective: float | None = None,
    iterations: int = 1_000_000,
    atol: float | None = None,
    rtol: float | None = None,
    sublattice_mode: Literal["interact"] = "interact",
    iteration_mode: IterationMode = "random",
    num_threads: int | None = None,
    precision: Prec = "single",
    max_results_per_objective: int = 10,
    log_level: LogLevel = "warn",
    **kwargs: Any,
) -> SqsResultPack[SqsResultInteract]: ...


@overload
def sqs_structures(
    structure: Atoms,
    composition: list[Composition],
    supercell: tuple[int, int, int] | None = None,
    shell_weights: list[ShellWeights] | None = None,
    shell_radii: list[ShellRadii] | None = None,
    objective: list[float] | None = None,
    iterations: int = 1_000_000,
    atol: float | None = None,
    rtol: float | None = None,
    sublattice_mode: Literal["split"] = "split",
    iteration_mode: IterationMode = "random",
    num_threads: int | None = None,
    precision: Prec = "single",
    max_results_per_objective: int = 10,
    log_level: LogLevel = "warn",
    **kwargs: Any,
) -> SqsResultPack[SqsResultSplit]: ...


def sqs_structures(
    structure: Atoms,
    composition: list[Composition] | Composition,
    supercell: tuple[int, int, int] | None = None,
    shell_weights: list[ShellWeights] | ShellWeights | None = None,
    shell_radii: list[ShellRadii] | ShellRadii | None = None,
    objective: list[float] | float | None = None,
    iterations: int = 1_000_000,
    atol: float | None = None,
    rtol: float | None = None,
    sublattice_mode: SublatticeMode = "interact",
    iteration_mode: IterationMode = "random",
    num_threads: int | None = None,
    precision: Prec = "single",
    max_results_per_objective: int = 10,
    log_level: LogLevel = "warn",
    **kwargs: Any,
) -> SqsResultPack[SqsResultSplit] | SqsResultPack[SqsResultInteract]:
    """
    Generate special quasirandom structures (SQS) using the sqsgenerator package.
    The function can handle both single and multiple sublattices,

    Args:
        structure: (ase.atoms.Atoms) The initial structure to optimize.
        composition (dict[str, int] | list[dict[str, int]]): The target composition(s) for the optimization.
            Each dictionary should map element symbols to their desired counts. If a list is provided, each dictionary
            corresponds to a different sublattice. A list is expected if sublattice_mode is set to "split".
            Use "sites" key to specify the sites that belong to a sublattice,
            e.g. {"Cu": 8, "Au": 8, "sites": [0, 1, ..., 15]}. In case you use "sites" key with atomic species,
            e.g. {"Cu": 8, "Au": 8, "sites": "Al"}, the "Al" sites refer to the atoms of the {structure} argument.
        supercell (tuple[int, int, int] | None): The supercell size to use for the optimization.
            If None, the original cell is used.
        shell_weights (dict[int, float] | list[dict[int, float]] | None): The weights for each shell in the objective
            function. The keys should be the shell numbers (starting from 1) and the values should be the corresponding
            weights. If a list is provided, each dictionary corresponds to a different sublattice ("split" mode).
        shell_radii (list[float] | list[list[float]] | None): The radii for each shell. Use to manually define
            the coordination shell radii. The list should contain the radii for each shell, starting from the first
            shell. If a list of lists is provided, each inner list corresponds to a different sublattice ("split" mode).
            If set to None 0.0 (=random) will be used in interact and [0.0, 0.0, ...] in split mode.
        objective: (float | list[float]) The target objective value(s) for the optimization. If a list is provided,
        each value corresponds to a different sublattice ("split" mode). In split mode diverging objectives are
        supported, e.g. [0, 1], to enable clustering, ordering, partial ordering or randomization for each sublattice.
        iterations: (int) The maximum number of iterations to perform during the optimization. In case iteration_mode
            is set to "systematic", this parameter is ignored.
        atol (float | None): The absolute tolerance for shell radii detection. If None, no absolute tolerance is used.
        rtol (float | None): The relative tolerance for shell radii detection. If None, no relative tolerance is used.
        sublattice_mode (str): The mode to use for handling sublattices. Can be either "interact" or "split".
            In "interact" mode, the whole cell is treated as a whole. In "split" mode, the cell is split into
            sublattices according to the "sites" key in the composition dictionaries, and the optimization is performed
            separately for each sublattice. "split" mode does not support iteration_mode "systematic".
        iteration_mode (str): The mode to use for iterating through the configuration space.
            Can be either "random" or "systematic".
        num_threads (int | None): The number of threads to use for the optimization.
            If None, the optimization will use the number of hardware threads it detects.
        precision (str): The precision to use for the optimization. Can be either "single" or "double".
        max_results_per_objective (int): The maximum number of results to return for each objective value.
            If the optimization finds more results with the same objective value, only at most
            {max_results_per_objective} results will be kept.
        log_level (str): The log level to use for the optimization. Can be either "trace", "debug", "info", "warn",
            or "error". The log level controls the verbosity of the output during optimization.
        **kwargs (Any): Additional keyword arguments to pass to the sqsgenerator optimization function.

    Returns:
        SqsResultPack: A pack of optimization results. The type of the results (SqsResultInteract or SqsResultSplit)
        depends on the sublattice_mode used for the optimization.

    """

    from sqsgenerator import parse_config
    from sqsgenerator.core import (
        ParseError,
        LogLevel as SqsLogLevel,
        SqsCallbackContext,
        optimize as sqs_optimize,
    )

    config = dict(
        prec=precision,
        iteration_mode=iteration_mode,
        sublattice_mode=sublattice_mode,
        structure=dict(
            lattice=structure.cell.array.tolist(),
            coords=structure.get_scaled_positions().tolist(),
            species=structure.get_atomic_numbers().tolist(),
        ),
        iterations=iterations,
        max_results_per_objective=max_results_per_objective,
    )
    if atol is not None:
        config["atol"] = atol
    if rtol is not None:
        config["rtol"] = rtol
    if supercell is not None:
        if all(n > 0 for n in supercell):
            config["structure"]["supercell"] = supercell
        else:
            raise ValueError(
                f"Invalid supercell: {supercell}. All dimensions must be positive integers."
            )

    def _preprocess_for_mode(v: T | list[T] | None) -> list[T] | None:
        match sublattice_mode:
            case "interact":
                return v
            case "split":
                return _ensure_list(v)
            case _:
                raise ValueError(
                    f"Invalid sublattice mode: {sublattice_mode}. Use 'interact' or 'split'."
                )

    if (composition := _preprocess_for_mode(composition)) is not None:
        config["composition"] = composition
    if (shell_weights := _preprocess_for_mode(shell_weights)) is not None:
        config["shell_weights"] = shell_weights
    if (shell_radii := _preprocess_for_mode(shell_radii)) is not None:
        config["shell_radii"] = shell_radii
    if objective is None:
        objective = 0.0 if sublattice_mode == "interact" else [0.0] * len(composition)
    config["target_objective"] = _preprocess_for_mode(objective)

    if num_threads is not None:
        if num_threads > 0:
            config["thread_config"] = num_threads
        else:
            raise ValueError(
                f"Invalid num_threads: {num_threads}. Must be a positive integer."
            )

    for kwarg, val in kwargs.items():
        if val is not None:
            config[kwarg] = val

    config = parse_config(config)
    if isinstance(config, ParseError):
        raise ValueError(
            f"Failed to parse config: parameter {config.key} -  {config.msg}"
        )

    stop_gracefully: bool = False
    stop_event = Event()

    def _callback(ctx: SqsCallbackContext) -> None:
        nonlocal stop_gracefully
        if stop_gracefully:
            ctx.stop()

    optimization_result: SqsResultPack | None = None

    match log_level:
        case "warn":
            level = SqsLogLevel.warn
        case "info":
            level = SqsLogLevel.info
        case "debug":
            level = SqsLogLevel.debug
        case "error":
            level = SqsLogLevel.error
        case "trace":
            level = SqsLogLevel.trace
        case _:
            raise ValueError(
                f"Invalid log level: {log_level}. Use 'trace', 'debug', 'info', 'warn', or 'error'."
            )

    def _optimize():
        result_local = sqs_optimize(config, log_level=level, callback=_callback)
        stop_event.set()
        nonlocal optimization_result
        optimization_result = result_local

    t = Thread(target=_optimize)
    t.start()
    try:
        while t.is_alive() and not stop_event.is_set():
            stop_event.wait(timeout=1.0)
    except (KeyboardInterrupt, EOFError):
        stop_gracefully = True
    finally:
        try:
            t.join(timeout=5.0)
        except TimeoutError:
            raise RuntimeError(
                "Optimization thread did not finish within the timeout period after requesting it to stop. "
                "The optimization may still be running in the background. Try to decrease chunk_size by passing it as a "
                "keyword argument to sqs_structures to make the optimization more responsive to stop requests."
            )

    if optimization_result is None:
        raise RuntimeError("Optimization failed to produce a result.")
    else:
        return SqsResultPack(optimization_result)
