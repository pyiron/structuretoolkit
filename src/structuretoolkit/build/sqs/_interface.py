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
    objective: float = 0.0,
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
) -> SqsResultPack[SqsResultInteract]:
    pass


@overload
def sqs_structures(
    structure: Atoms,
    composition: list[Composition],
    supercell: tuple[int, int, int] | None = None,
    shell_weights: list[ShellWeights] | None = None,
    shell_radii: list[ShellRadii] | None = None,
    objective: list[float] = 0.0,
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
) -> SqsResultPack[SqsResultSplit]:
    pass


def sqs_structures(
    structure: Atoms,
    composition: list[Composition] | Composition,
    supercell: tuple[int, int, int] | None = None,
    shell_weights: list[ShellWeights] | ShellWeights | None = None,
    shell_radii: list[ShellRadii] | ShellRadii | None = None,
    objective: list[float] | float = 0.0,
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
) -> list[SqsResultSplit]:
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
            stop_event.wait()
    except (KeyboardInterrupt, EOFError):
        stop_gracefully = True
    finally:
        t.join()

    if optimization_result is None:
        raise RuntimeError("Optimization failed to produce a result.")
    else:
        return SqsResultPack(optimization_result)
