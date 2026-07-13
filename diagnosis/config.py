from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


# --- Core config dataclasses -------------------------------------------------


@dataclass
class InputConfig:
    """
    Configuration for the falsified requirement and traces.
    """

    requirement_file: str
    traces_file: str
    output_dir: str = "outputs"


@dataclass
class EvaluationConfig:
    """
    Configuration for candidate property evaluation.
    """

    trace_check_timeout_sec: int = 3600
    cache_enabled: bool = False
    engine: str = "subprocess"
    parallel_workers: int = 1
    # Optional stable location for the verdict cache. When set, the cache is
    # written here instead of inside the (timestamped) run directory, so it can
    # be preserved across runs to resume an interrupted run cheaply.
    cache_path: Optional[str] = None


@dataclass
class GAConfig:
    """
    Genetic algorithm configuration parameters.
    """

    population_size: int = 50
    generations: int = 50
    crossover_rate: float = 0.9
    mutation_rate: float = 0.1
    elitism: int = 1
    seed: Optional[int] = None
    target_sats: int = 2
    stopping: "GAStoppingConfig" = field(default_factory=lambda: GAStoppingConfig())


@dataclass
class GAStoppingConfig:
    """
    Optional adaptive stopping criteria.
    """

    mode: str = "count"
    pr_threshold: float = 0.95
    check_every_generations: int = 1
    patience: int = 2
    min_samples: int = 0
    max_samples: Optional[int] = None


@dataclass
class MutationConfig:
    """
    Configuration for mutations.
    """

    max_mutations: int = 1

    enable_numeric_perturbation: bool = True
    enable_relop_flip: bool = True
    enable_logical_flip: bool = True
    enable_quantifier_flip: bool = True

    # Positions in the flattened AST (preorder indices)
    allowed_positions: Optional[List[int]] = None

    # Unified per-position constraints:
    # idx -> {"numeric": [lo, hi], "relational": [...], "equals": [...], "logical": [...], "arith": [...], "quantifier": [...]}
    allowed_changes: Dict[int, Dict[str, object]] = field(default_factory=dict)


@dataclass
class IntervalInferenceConfig:
    """
    Opt-in banded interval inference (Feature 2). OFF by default.
    """

    enabled: bool = False
    mode: str = "guide"              # "guide" | "label"
    empirical_validation_k: int = 3  # confirming solves before trusting a direction
    min_gap: float = 1e-6            # relative bracket width below which interval shrinking stops


@dataclass
class TwoTierTimeoutConfig:
    """
    Opt-in two-tier solver timeout (Feature 3). OFF by default.
    """

    enabled: bool = False
    low_sec: int = 60
    high_sec: int = 600
    escalation: str = "once_per_formula"


@dataclass
class AdaptiveRangeConfig:
    """
    Opt-in adaptive mutation range (Feature A). OFF by default.

    SEARCH-BEHAVIOUR CHANGE: when ``enabled`` this alters candidate generation
    for the monotone numeric position -- results obtained with
    ``adaptive_range=true`` require re-validation against expert ground truth.
    All defaults live here; the ``heuristics.adaptive_range`` config block only
    overrides them.
    """

    enabled: bool = False
    exploration_fraction: float = 0.15   # share of draws from the FULL configured range
    endpoint_init: bool = True           # evaluate range endpoints before generation 0
    on_one_class: str = "report_and_stop"  # "report_and_stop" | "widen" | "continue"
    widen_factor: float = 1.5            # geometric range expansion (only for "widen")
    max_widenings: int = 4


@dataclass
class TimeQuantizationConfig:
    """
    Opt-in sample-aligned time-window quantization (Sprint 7). OFF by default.

    SEARCH-BEHAVIOUR CHANGE: when ``enabled`` the verdict cache is consulted with
    a *canonical* key in which every quantizable time token is replaced by its
    sample-class index ``floor(value / period)`` -- distinct raw bounds inside one
    inter-sample interval then collide on a single solve. The formula sent to the
    solver on a miss is still the original, un-canonicalized one; only the key
    changes. Reported time boundaries become sample-aligned, so results obtained
    with ``time_quantization=true`` require re-validation against expert ground
    truth. All defaults live here; the ``heuristics.time_quantization`` block only
    overrides them.
    """

    enabled: bool = False
    validate_every_n_hits: int = 50   # periodic same-class double-solve validation
    period: Optional[float] = None    # optional override (see quantize.quantizability)
    force_period: bool = False        # validation-only: skip the period cross-check


@dataclass
class HeuristicsConfig:
    """
    Opt-in search-pruning heuristics. Everything OFF by default so a config with
    no ``heuristics`` block behaves exactly like the perf-update baseline.
    """

    interval_inference: IntervalInferenceConfig = field(
        default_factory=IntervalInferenceConfig
    )
    two_tier_timeout: TwoTierTimeoutConfig = field(
        default_factory=TwoTierTimeoutConfig
    )
    adaptive_range: AdaptiveRangeConfig = field(
        default_factory=AdaptiveRangeConfig
    )
    time_quantization: TimeQuantizationConfig = field(
        default_factory=TimeQuantizationConfig
    )


@dataclass
class Config:
    """
    Top-level configuration object for a diagnosis run.
    """

    input: InputConfig = field(default_factory=InputConfig)
    ga: GAConfig = field(default_factory=GAConfig)
    mutation: MutationConfig = field(default_factory=MutationConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    heuristics: HeuristicsConfig = field(default_factory=HeuristicsConfig)


# --- Loader utilities --------------------------------------------------------


class ConfigError(RuntimeError):
    """Raised when there is a problem loading or validating a configuration."""


def _as_dict(obj: Any) -> Dict[str, Any]:
    if not isinstance(obj, dict):
        raise ConfigError(f"Expected dict at top level, got {type(obj)!r}")
    return obj


def _load_json(path: Path) -> Dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"Failed to read config file {path!s}: {exc}") from exc

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ConfigError(f"Invalid JSON in config file {path!s}: {exc}") from exc

    return _as_dict(data)

_ALLOWED_CHANGE_KEYS = {"numeric", "logical", "relational", "equals", "quantifier", "arith"}

def _parse_allowed_changes(mut_data: Dict[str, Any], path: Path) -> Dict[int, Dict[str, object]]:
    raw = mut_data.get("allowed_changes", {})
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ConfigError(
            f"Expected 'mutation.allowed_changes' to be an object/dict in {path!s}, got {type(raw)!r}"
        )

    allowed_changes: Dict[int, Dict[str, object]] = {}

    for k, spec in raw.items():
        try:
            idx = int(k)  # JSON keys are strings -> int
        except Exception as exc:
            raise ConfigError(
                f"Invalid key in mutation.allowed_changes: {k!r} (expected an integer-like string)"
            ) from exc

        if not isinstance(spec, dict):
            raise ConfigError(
                f"mutation.allowed_changes[{k!r}] must be an object/dict, got {type(spec)!r}"
            )

        parsed_spec: Dict[str, object] = {}
        for family, payload in spec.items():
            if family not in _ALLOWED_CHANGE_KEYS:
                raise ConfigError(
                    f"Unknown allowed_changes family {family!r} at position {idx}. "
                    f"Expected one of: {sorted(_ALLOWED_CHANGE_KEYS)}"
                )

            if family == "numeric":
                if not (isinstance(payload, (list, tuple)) and len(payload) == 2):
                    raise ConfigError(
                        f"mutation.allowed_changes[{idx}].numeric must be [lo, hi], got: {payload!r}"
                    )
                lo, hi = payload
                parsed_spec["numeric"] = [float(lo), float(hi)]
                continue

            # operator families: must be non-empty list[str]
            if not (isinstance(payload, list) and payload and all(isinstance(x, str) for x in payload)):
                raise ConfigError(
                    f"mutation.allowed_changes[{idx}].{family} must be a non-empty list of strings, got: {payload!r}"
                )
            parsed_spec[family] = payload

        allowed_changes[idx] = parsed_spec

    return allowed_changes


def _parse_stopping(ga_data: Dict[str, Any], path: Path) -> GAStoppingConfig:
    raw = ga_data.get("stopping", {})
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ConfigError(
            f"Expected 'ga.stopping' to be an object/dict in {path!s}, got {type(raw)!r}"
        )
    mode = str(raw.get("mode", GAStoppingConfig.mode))
    if mode not in ("count", "cv_pr", "tree_stable"):
        raise ConfigError(
            f"Invalid 'ga.stopping.mode' {mode!r} in {path!s}: "
            "expected 'count', 'cv_pr', or 'tree_stable'"
        )
    max_samples = raw.get("max_samples", GAStoppingConfig.max_samples)
    return GAStoppingConfig(
        mode=mode,
        pr_threshold=float(raw.get("pr_threshold", GAStoppingConfig.pr_threshold)),
        check_every_generations=max(
            1,
            int(raw.get(
                "check_every_generations",
                GAStoppingConfig.check_every_generations,
            )),
        ),
        patience=max(1, int(raw.get("patience", GAStoppingConfig.patience))),
        min_samples=max(0, int(raw.get("min_samples", GAStoppingConfig.min_samples))),
        max_samples=None if max_samples is None else max(0, int(max_samples)),
    )


def _parse_heuristics(
    data: Dict[str, Any],
    trace_check_timeout_sec: int,
    path: Path,
) -> HeuristicsConfig:
    raw = data.get("heuristics", {})
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ConfigError(
            f"Expected 'heuristics' to be an object/dict in {path!s}, got {type(raw)!r}"
        )

    # --- interval_inference ---
    ii_raw = raw.get("interval_inference", {}) or {}
    if not isinstance(ii_raw, dict):
        raise ConfigError(
            f"Expected 'heuristics.interval_inference' to be an object in {path!s}"
        )
    mode = str(ii_raw.get("mode", IntervalInferenceConfig.mode))
    if mode not in ("guide", "label"):
        raise ConfigError(
            f"Invalid 'heuristics.interval_inference.mode' {mode!r} in {path!s}: "
            "expected 'guide' or 'label'"
        )
    k = int(ii_raw.get("empirical_validation_k", IntervalInferenceConfig.empirical_validation_k))
    if k < 0:
        raise ConfigError(
            f"'heuristics.interval_inference.empirical_validation_k' must be >= 0, got {k}"
        )
    min_gap = float(ii_raw.get("min_gap", IntervalInferenceConfig.min_gap))
    if min_gap <= 0:
        raise ConfigError(
            f"'heuristics.interval_inference.min_gap' must be > 0, got {min_gap}"
        )
    interval_cfg = IntervalInferenceConfig(
        enabled=bool(ii_raw.get("enabled", IntervalInferenceConfig.enabled)),
        mode=mode,
        empirical_validation_k=k,
        min_gap=min_gap,
    )

    # --- two_tier_timeout ---
    tt_raw = raw.get("two_tier_timeout", {}) or {}
    if not isinstance(tt_raw, dict):
        raise ConfigError(
            f"Expected 'heuristics.two_tier_timeout' to be an object in {path!s}"
        )
    escalation = str(tt_raw.get("escalation", TwoTierTimeoutConfig.escalation))
    if escalation != "once_per_formula":
        raise ConfigError(
            f"Invalid 'heuristics.two_tier_timeout.escalation' {escalation!r} in {path!s}: "
            "only 'once_per_formula' is supported"
        )
    low_sec = int(tt_raw.get("low_sec", TwoTierTimeoutConfig.low_sec))
    high_sec = int(tt_raw.get("high_sec", TwoTierTimeoutConfig.high_sec))
    two_tier_enabled = bool(tt_raw.get("enabled", TwoTierTimeoutConfig.enabled))
    if low_sec <= 0 or high_sec <= 0:
        raise ConfigError(
            f"'heuristics.two_tier_timeout' low_sec/high_sec must be > 0 in {path!s}"
        )
    if high_sec < low_sec:
        raise ConfigError(
            f"'heuristics.two_tier_timeout.high_sec' ({high_sec}) must be >= "
            f"low_sec ({low_sec}) in {path!s}"
        )
    # Interaction rule: the high tier must compose with the hard trace-check cap.
    if two_tier_enabled and high_sec > trace_check_timeout_sec:
        raise ConfigError(
            f"'heuristics.two_tier_timeout.high_sec' ({high_sec}) must be <= "
            f"'evaluation.trace_check_timeout_sec' ({trace_check_timeout_sec}) in {path!s}"
        )
    two_tier_cfg = TwoTierTimeoutConfig(
        enabled=two_tier_enabled,
        low_sec=low_sec,
        high_sec=high_sec,
        escalation=escalation,
    )

    # --- adaptive_range (Feature A) ---
    ar_raw = raw.get("adaptive_range", {}) or {}
    if not isinstance(ar_raw, dict):
        raise ConfigError(
            f"Expected 'heuristics.adaptive_range' to be an object in {path!s}"
        )
    on_one_class = str(ar_raw.get("on_one_class", AdaptiveRangeConfig.on_one_class))
    if on_one_class not in ("report_and_stop", "widen", "continue"):
        raise ConfigError(
            f"Invalid 'heuristics.adaptive_range.on_one_class' {on_one_class!r} in {path!s}: "
            "expected 'report_and_stop', 'widen', or 'continue'"
        )
    exploration_fraction = float(
        ar_raw.get("exploration_fraction", AdaptiveRangeConfig.exploration_fraction)
    )
    if not (0.0 <= exploration_fraction <= 1.0):
        raise ConfigError(
            f"'heuristics.adaptive_range.exploration_fraction' must be in [0, 1], "
            f"got {exploration_fraction}"
        )
    widen_factor = float(ar_raw.get("widen_factor", AdaptiveRangeConfig.widen_factor))
    if widen_factor <= 1.0:
        raise ConfigError(
            f"'heuristics.adaptive_range.widen_factor' must be > 1, got {widen_factor}"
        )
    max_widenings = int(ar_raw.get("max_widenings", AdaptiveRangeConfig.max_widenings))
    if max_widenings < 0:
        raise ConfigError(
            f"'heuristics.adaptive_range.max_widenings' must be >= 0, got {max_widenings}"
        )
    adaptive_cfg = AdaptiveRangeConfig(
        enabled=bool(ar_raw.get("enabled", AdaptiveRangeConfig.enabled)),
        exploration_fraction=exploration_fraction,
        endpoint_init=bool(ar_raw.get("endpoint_init", AdaptiveRangeConfig.endpoint_init)),
        on_one_class=on_one_class,
        widen_factor=widen_factor,
        max_widenings=max_widenings,
    )

    # --- time_quantization (Sprint 7) ---
    tq_raw = raw.get("time_quantization", {}) or {}
    if not isinstance(tq_raw, dict):
        raise ConfigError(
            f"Expected 'heuristics.time_quantization' to be an object in {path!s}"
        )
    validate_every = int(
        tq_raw.get("validate_every_n_hits", TimeQuantizationConfig.validate_every_n_hits)
    )
    if validate_every < 0:
        raise ConfigError(
            f"'heuristics.time_quantization.validate_every_n_hits' must be >= 0, "
            f"got {validate_every}"
        )
    tq_period = tq_raw.get("period", TimeQuantizationConfig.period)
    if tq_period is not None:
        tq_period = float(tq_period)
        if tq_period <= 0:
            raise ConfigError(
                f"'heuristics.time_quantization.period' must be > 0, got {tq_period}"
            )
    time_quant_cfg = TimeQuantizationConfig(
        enabled=bool(tq_raw.get("enabled", TimeQuantizationConfig.enabled)),
        validate_every_n_hits=validate_every,
        period=tq_period,
        force_period=bool(tq_raw.get("force_period", TimeQuantizationConfig.force_period)),
    )

    return HeuristicsConfig(
        interval_inference=interval_cfg,
        two_tier_timeout=two_tier_cfg,
        adaptive_range=adaptive_cfg,
        time_quantization=time_quant_cfg,
    )


def load_config(path: str | Path) -> Config:
    """
    Load a configuration from a JSON file.

    Expected high-level structure:

    {
      "input": {
        "requirement_file": "tool/requirements.hls",
        "traces_file": "tool/example_traces.txt",
        "output_dir": "outputs"
      },
      "ga": {
        "population_size": 80,
        "generations": 60,
        "crossover_rate": 0.9,
        "mutation_rate": 0.1,
        "elitism": 2,
        "seed": 42
      },
      "mutation": {
        "max_mutations": 1,
        "enable_numeric_perturbation": true,
        "enable_relop_flip": false,
        "enable_logical_flip": false,
        "enable_quantifier_flip": false,
        "allowed_positions": [14],
        "allowed_changes": {
          "14": {"numeric": [100.0, 140.0]}
        }
      }
    }
    """
    path = Path(path)
    data = _load_json(path)

    input_data = _as_dict(data.get("input", {}))
    ga_data = _as_dict(data.get("ga", {}))
    mut_data = _as_dict(data.get("mutation", {}))
    evaluation_data = _as_dict(data.get("evaluation", {}))

    try:
        input_cfg = InputConfig(
            requirement_file=input_data["requirement_file"],
            traces_file=input_data["traces_file"],
            output_dir=input_data.get("output_dir", "outputs"),
        )
    except KeyError as exc:
        missing = exc.args[0]
        raise ConfigError(
            f"Missing required field 'input.{missing}' in config file {path!s}"
        ) from exc

    ga_cfg = GAConfig(
        population_size=int(ga_data.get("population_size", GAConfig.population_size)),
        generations=int(ga_data.get("generations", GAConfig.generations)),
        crossover_rate=float(ga_data.get("crossover_rate", GAConfig.crossover_rate)),
        mutation_rate=float(ga_data.get("mutation_rate", GAConfig.mutation_rate)),
        elitism=int(ga_data.get("elitism", GAConfig.elitism)),
        seed=ga_data.get("seed"),
        target_sats=ga_data.get("target_sats"),
        stopping=_parse_stopping(ga_data, path),
    )

    # --- mutation section ---
    allowed_changes = _parse_allowed_changes(mut_data, path)

    mut_cfg = MutationConfig(
        max_mutations=int(mut_data.get("max_mutations", 1)),
        enable_numeric_perturbation=bool(mut_data.get("enable_numeric_perturbation", True)),
        enable_relop_flip=bool(mut_data.get("enable_relop_flip", True)),
        enable_logical_flip=bool(mut_data.get("enable_logical_flip", True)),
        enable_quantifier_flip=bool(mut_data.get("enable_quantifier_flip", True)),
        allowed_positions=mut_data.get("allowed_positions"),
        allowed_changes=allowed_changes,
    )

    engine = str(evaluation_data.get("engine", EvaluationConfig.engine))
    if engine not in ("subprocess", "worker"):
        raise ConfigError(
            f"Invalid 'evaluation.engine' {engine!r} in {path!s}: "
            "expected 'subprocess' or 'worker'"
        )

    evaluation_cfg = EvaluationConfig(
        trace_check_timeout_sec=int(
            evaluation_data.get(
                "trace_check_timeout_sec",
                EvaluationConfig.trace_check_timeout_sec,
            )
        ),
        cache_enabled=bool(
            evaluation_data.get("cache_enabled", EvaluationConfig.cache_enabled)
        ),
        engine=engine,
        parallel_workers=max(
            1,
            int(evaluation_data.get("parallel_workers", EvaluationConfig.parallel_workers)),
        ),
        cache_path=evaluation_data.get("cache_path", EvaluationConfig.cache_path),
    )

    heuristics_cfg = _parse_heuristics(
        data, evaluation_cfg.trace_check_timeout_sec, path
    )

    return Config(
        input=input_cfg,
        ga=ga_cfg,
        mutation=mut_cfg,
        evaluation=evaluation_cfg,
        heuristics=heuristics_cfg,
    )
