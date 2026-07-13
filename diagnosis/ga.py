import os
import math
import time
import json
import random
import datetime
import subprocess
import queue
import threading
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from copy import deepcopy
from typing import Optional

from pathlib import Path

from . import treenode
from .defs import (
    CROSSOVER_RATE,
    MUTATION_RATE,
    POPULATION_SIZE,
    CHROMOSOME_TO_PRESERVE
    )
from .individual import (
    Individual,
    QUANTIFIERS,
    RELATIONALS,
    EQUALS,
    ARITHMETICS,
    MULDIV,
    EXP,
    LOGICALS,
    NEG,
    FUNC,
)
from .diagnostics.j48 import run_j48
from .diagnostics.summary import parse_j48_out
from .diagnostics.arff import write_dataset_all, write_dataset_qty, write_dataset_sat_unsat

from .lang.python_printer import formula_to_python_expr
from .lang.internal_parser import parse_internal_obj
from .lang.internal_encoder import FormulaLayout
from .lang.ast import Formula, Not

from .harness import run_property_script, Verdict
from .cache import VerdictCache
from .worker import (
    SolverWorker,
    WorkerCrash,
    V_ERROR,
    V_SATISFIED,
    V_UNDECIDED,
    V_VIOLATED,
)

from .fitness import Fitness, SmithWatermanFitness

from .mutation import MutationConfig
from .inference import HeuristicsController, Plan
from .lang.polarity import Monotonicity

# Prints check-in and checkout timings
CHECK_PRINT = False
PROPERTY_ASSERTION_MARKER = "z3solver.add(Not("


def replace_property_assertion(
    lines: list[str], nline: str, z3_timeout_ms: int | None = None
) -> list[str]:
    """
    Replace the unique negated property assertion, preserving trace constraints.

    When ``z3_timeout_ms`` is given, a ``z3solver.set("timeout", ms)`` line is
    emitted immediately before the assertion so the subprocess engine honours a
    per-candidate z3 timeout (used by the two-tier timeout heuristic). When it is
    ``None`` the output is byte-identical to the baseline.
    """
    marker_indexes = [
        idx for idx, line in enumerate(lines)
        if PROPERTY_ASSERTION_MARKER in line
    ]
    if len(marker_indexes) != 1:
        raise RuntimeError(
            f"Expected exactly one {PROPERTY_ASSERTION_MARKER!r} property line; "
            f"found {len(marker_indexes)}"
        )

    marker_idx = marker_indexes[0]
    new_lines = list(lines)
    line = lines[marker_idx]
    indent = line[: len(line) - len(line.lstrip())]
    if z3_timeout_ms is None:
        new_lines[marker_idx] = f"{indent}z3solver.add({nline})\n"
    else:
        new_lines[marker_idx] = (
            f'{indent}z3solver.set("timeout", {int(z3_timeout_ms)})\n'
            f"{indent}z3solver.add({nline})\n"
        )
    return new_lines


def write_candidate_script(
    src: str | Path, dst: str | Path, nline: str, z3_timeout_ms: int | None = None
) -> None:
    """Write a property script with the candidate assertion injected."""
    with open(src, "r", encoding="utf-8") as f:
        lines = f.readlines()
    new_lines = replace_property_assertion(lines, nline, z3_timeout_ms)
    with open(dst, "w", encoding="utf-8") as f:
        f.writelines(new_lines)

class GA(object):
    """GA over requirement formulas with AST-based mutation and harness-backed evaluation."""
    def __init__(
        self,
        init_form,
        target_sats: int = 2,
        population_size: int | None = None,
        max_generations: int | None = None,
        crossover_rate: float | None = None,
        mutation_rate: float | None = None,
        seed: int | None = None,
        fitness: Fitness | None = None,
        output_root: str | None = None,
        mutation_config: MutationConfig | None = None,
        property_path: str | None = None,
        formula_layout: FormulaLayout | None = None,
        trace_check_timeout_sec: int = 3600,
        cache_enabled: bool = False,
        engine: str = "subprocess",
        parallel_workers: int = 1,
        cache_path: str | None = None,
        stopping_config=None,
        heuristics_config=None,
        ):
        super(GA, self).__init__()

        # Log the timespaneach one of the tree steps in the approach
        self.checkin_start = {
            'mutation_timestamp': 0.0,
            'tracheck_timestamp': 0.0,
            'diagnosi_timestamp': 0.0
        }
        self.timespan_log = {
            'mutation_timestamp': 0.0,
            'tracheck_timestamp': 0.0,
            'diagnosi_timestamp': 0.0
        }

        # AST-level mutation configuration (from pipeline / config.json)
        self.mutation_config: MutationConfig | None = mutation_config

        # Seed the RNG (configurable)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        else:
            random.seed()

        # Fitness configuration (default: Smith–Waterman)
        if fitness is None:
            self.fitness: Fitness = SmithWatermanFitness()
        else:
            self.fitness = fitness

        # Population size from config, falling back to legacy constant
        if population_size is not None:
            self.size = int(population_size)
        else:
            self.size = POPULATION_SIZE

        # Max generations from config, falling back to legacy constant
        if max_generations is not None:
            self.max_generations = int(max_generations)
        else:
            self.max_generations = MAX_ALLOWABLE_GENERATIONS

        # Crossover rate from config, falling back to legacy constant
        if crossover_rate is not None:
            self.crossover_rate = crossover_rate
        else:
            self.crossover_rate = CROSSOVER_RATE
        
        if mutation_rate is not None:
            self.mutation_rate = mutation_rate
        else:
            self.mutation_rate = MUTATION_RATE
        
        self.highest_sat = None
        self.population = []
        self.now = datetime.datetime.now()
        self.output_root = Path(output_root)
        self.init_log(self.output_root)

        self.init_form = init_form

        # AST view of the seed formula
        self.seed_ast: Optional[Formula]
        try:
            self.seed_ast = parse_internal_obj(self.init_form)
        except Exception as exc:
            # Do not break GA if the AST view fails; just log and continue
            print(f"[diagnosis] Warning: failed to build seed AST: {exc}")
            self.seed_ast = None

        self.target_sats = int(target_sats)
        self.trace_check_timeout_sec = int(trace_check_timeout_sec)
        self.cache_enabled = bool(cache_enabled)
        self.cache_path = cache_path
        self.engine = engine
        self.parallel_workers = max(1, int(parallel_workers))
        self.worker: SolverWorker | None = None
        self.worker_restarts = 0
        self.worker_fallbacks = 0
        self.cache_lock = threading.Lock()
        self.stopping_config = stopping_config
        self.stopping_mode = getattr(stopping_config, "mode", "count")
        self.stopping_checks = []
        self._stopping_success_streak = 0
        self._last_tree_hash = None
        # Latest adaptive-stopping check result, surfaced in the per-generation
        # terminal line for cv_pr / tree_stable modes.
        self._last_pr = None          # (generation, precision, recall, streak)
        self._last_tree_stable = None  # (generation, stable_bool, streak)

        self.property_path = property_path 
        self.base_property_script: str | None = None

        # Copy the original property script into this run's output dir
        # for traceability, and remember where it ended up.
        self.base_property_script = self.copy_temp_file()

        if self.base_property_script:
            print(f'Running script {self.base_property_script}')
            with open(f'{self.path}/hypot.txt', 'a') as f:
                f.write(f'\t{self.base_property_script}\n')
        else:
            print('[diagnosis] WARNING: no property script copied (property_path not set?)')

        self.formula_layout = formula_layout
        cache_file = (
            Path(self.cache_path) if self.cache_path
            else Path(self.path) / "verdict_cache.sqlite"
        )
        self.verdict_cache = VerdictCache(
            cache_file,
            enabled=self.cache_enabled,
        )

        # Opt-in search-pruning heuristics (polarity-gated interval inference +
        # two-tier timeout). Defaults to an all-off config so the baseline path
        # is unchanged.
        if heuristics_config is None:
            from .config import HeuristicsConfig
            heuristics_config = HeuristicsConfig()
        self.heuristics_config = heuristics_config
        # Trace timestamp spacing backs the quantization period cross-check; the
        # requirement script inlines the trace, so it is the authoritative source.
        from .quantization import trace_period_from_property

        trace_period = (
            trace_period_from_property(self.property_path)
            if self.property_path
            else None
        )
        self.heuristics = HeuristicsController(
            seed_ast=self.seed_ast,
            heuristics_cfg=heuristics_config,
            trace_check_timeout_sec=self.trace_check_timeout_sec,
            run_dir=self.path,
            trace_period=trace_period,
        )

        self.init_population()
        self.execution_report = {'TOTAL': 0}
        self._adaptive_early_stop = False
        self._adaptive_early_stop_message: str | None = None

        self.hypots = []
        self.sats = []
        self.unsats = []
        self.unknown = []
        self.entire_dataset = []  # collects sats/unsats/unknown for diagnostics

        # --- Instrumentation: lightweight run statistics (for sensitivity experiments) ---
        self.stats = {
            "generations_completed": 0,
            "individuals_evaluated": 0,
            "best_fitness": None,
            "best_fitness_by_gen": [],
        }

    def _report_cache_stats(self):
        if self.cache_enabled:
            self.execution_report.update(self.verdict_cache.stats())
        if self.engine == "worker":
            self.execution_report["worker_restarts"] = self.worker_restarts
            self.execution_report["worker_fallbacks"] = self.worker_fallbacks
        if self.parallel_workers > 1:
            self.execution_report["parallel_workers"] = self.parallel_workers
        if self.heuristics.any_on or self.heuristics.quant_on:
            self.execution_report.update(self.heuristics.report())

    def _update_stats_after_evaluate(self):
        """
        Update self.stats after self.evaluate() has assigned fitness values.
        Assumes larger fitness is better (consistent with reverse=True sorting).
        """
        try:
            pop = getattr(self, "population", None) or []
            n = len(pop)
            self.stats["individuals_evaluated"] += n

            if n > 0:
                # Some individuals may have None fitness if evaluation errored; ignore them
                fitness_vals = [c.fitness for c in pop if getattr(c, "fitness", None) is not None]
                if fitness_vals:
                    gen_best = max(fitness_vals)
                    self.stats["best_fitness_by_gen"].append(gen_best)

                    best_so_far = self.stats["best_fitness"]
                    if best_so_far is None or gen_best > best_so_far:
                        self.stats["best_fitness"] = gen_best

            # generations_completed tracks how many GA generations have been *produced* so far
            self.stats["generations_completed"] = int(getattr(self, "generation_counter", 0))
        except Exception:
            # Never let instrumentation break the GA
            pass

    def _progress(self, iterable, desc: str = ""):
        """
        Wrap an iterable in a tqdm progress bar if available; otherwise return
        the iterable unchanged. Used to avoid spamming 'evaluating  0' logs.
        """
        if tqdm is not None:
            return tqdm(iterable, desc=desc, leave=False)
        else:
            if desc:
                print(desc)
            return iterable

    def print_seed_ast(self):
        """Print the AST representation of the seed formula, if available."""
        if self.seed_ast is None:
            print("No seed AST available.")
        else:
            print(self.seed_ast)

    def get_max_score(self):
        tokens = self.replace_token(list(self.seed))
        self.max_score = self.fitness.compute_max_score(tokens=tokens)

    def init_population(self):
        self.population = []
        root = treenode.parse(self.init_form)
        self.seed = root
        terminators = list(set(treenode.get_terminators(root)))

        # Seed individual has both tree and AST
        self.seed_ch = deepcopy(Individual(root, terminators, self.seed_ast))
        # self.seed_ch.show_idx()
        #print(f"terminators = {terminators}")

        self.checkin("mutation_timestamp")
        for i in range(0, self.size):
            # Each chromosome starts from the same tree + AST seed
            chromosome = deepcopy(Individual(root, terminators, self.seed_ast))

            n = random.randrange(len(root))

            if self.mutation_config is not None:
                chromosome.mutate(1, mutation_config=self.mutation_config)
                ar = getattr(self.heuristics_config, "adaptive_range", None)
                if not (ar and ar.enabled and ar.endpoint_init):
                    self._maybe_adapt_range(chromosome)

            self.population.append(deepcopy(chromosome))
        self.checkout("mutation_timestamp")
        print(f"Population initialized. Size = {self.size}")

    def init_log(self, parent_dir):
        directory = str(self.now)
        self.path = os.path.join(parent_dir, directory)
        self.path = self.path.replace(' ', '_')
        self.path = self.path.replace(':', '_')
        print(self.path)
        try:
            os.mkdir(self.path)
            pass
        except OSError as error:
            print(error)
        # Per-generation population logs live under generations/ to keep the run
        # root uncluttered (one NN.txt per generation would otherwise dominate it).
        try:
            os.mkdir(os.path.join(self.path, "generations"))
        except OSError:
            pass

    def copy_temp_file(self) -> str:
        """
        Copy the original property script (self.property_path) into this run's
        output directory for traceability.

        Returns the full path to the copied file, or "" if nothing was copied.
        """
        if not self.property_path:
            return ""

        src = Path(self.property_path).resolve()
        filename = src.name
        dst = Path(self.path) / filename

        try:
            with src.open('r', encoding='utf-8') as infile, dst.open('w', encoding='utf-8') as outfile:
                for line in infile:
                    outfile.write(line)
        except FileNotFoundError as exc:
            print(f"[diagnosis] WARNING: could not copy property script {src}: {exc}")
            return ""

        return str(dst)

    def write_population(self, generation):
        self._report_cache_stats()
        with open('{}/generations/{:0>2}.txt'.format(self.path, generation), 'w') as f:
            f.write('Formula\tFitness\tSatisfied\n')
            if self.highest_sat:
                f.write('HC: ')
                f.write(str(self.highest_sat))
                f.write(f'\t{self.highest_sat.fitness}')
                f.write(f'\t{self.highest_sat.madeit}')
                f.write('\n')
            for i, chromosome in enumerate(self.population):
                f.write('{:0>2}'.format(i)+': ')
                # print(chromosome.format())
                f.write(str(chromosome))
                f.write(f'\t{chromosome.fitness}')
                f.write(f'\t{chromosome.madeit}')
                f.write('\n')
        self._write_report()

    def _write_report(self):
        self._report_cache_stats()
        json_object = json.dumps(self.execution_report, indent=4)
        with open(f"{self.path}/report.json", "w") as outfile:
            outfile.write(json_object)

    def generate_dataset_qty(self):
        res = []
        [res.append(x) for x in self.population if x not in res]
        # In guide mode, inferred individuals carry include_in_arff=False and are
        # excluded from the diagnostics dataset (the DT trains on real verdicts
        # only). With heuristics off every individual has include_in_arff=True,
        # so this is a no-op and the baseline dataset is unchanged.
        [self.unknown.append(x) for x in self.population if (x not in self.unknown) and (x.madeit == 'Unknown') and getattr(x, 'include_in_arff', True) and not getattr(x, 'vacuous', False)]
        [self.unsats.append(x) for x in self.population if (x not in self.unsats) and (x.madeit == 'False') and getattr(x, 'include_in_arff', True) and not getattr(x, 'vacuous', False)]
        [self.sats.append(x) for x in self.population if (x not in self.sats) and (x.madeit == 'True') and getattr(x, 'include_in_arff', True) and not getattr(x, 'vacuous', False)]

        # --- current (per-population) view ---
        pop = self.population
        cur_sat = sum(1 for x in pop if x.madeit == "True")
        cur_unsat = sum(1 for x in pop if x.madeit == "False")
        cur_unk = sum(1 for x in pop if x.madeit == "Unknown")
        cur_total = len(pop)

        # --- cumulative (ever-seen) view (used by check_evolution) ---
        cum_sat = len(self.sats)
        cum_unsat = len(self.unsats)
        cum_unk = len(self.unknown)
        cum_total = cum_sat + cum_unsat + cum_unk

        # Skip the "pre-eval" view (it is typically all-Unknown).
        if cur_total > 0 and cur_unk == cur_total:
            return

        def pct(part, whole):
            return (100.0 * part / whole) if whole else 0.0

        def tri_bar(sat, unsat, unk, total, width=20):
            if not total:
                return "░" * width
            sat_w = int(round(width * sat / total))
            unsat_w = int(round(width * unsat / total))

            # Clamp to avoid rounding overflow
            sat_w = max(0, min(width, sat_w))
            unsat_w = max(0, min(width - sat_w, unsat_w))
            unk_w = width - sat_w - unsat_w

            return ("█" * sat_w) + ("▓" * unsat_w) + ("░" * unk_w)

        BAR_W = 20
        BAR_COL = 110  # <- pick a column that fits your terminal width (100–130 is typical)

        cur_bar = tri_bar(cur_sat, cur_unsat, cur_unk, cur_total, width=BAR_W)
        cum_bar = tri_bar(cum_sat, cum_unsat, cum_unk, cum_total, width=BAR_W)

        gen = getattr(self, "generation_counter", None)
        gen_str = f"gen={gen:02d}" if isinstance(gen, int) else "gen=NA"

        target = getattr(self, "target_sats", None)
        stop_str = self._stop_criteria_line(cum_sat, cum_unsat, target)

        def fmt_counts(total, sat, unsat, unk) -> str:
            return (
                f"total={total:5d}  "
                f"sat={sat:5d}({pct(sat,total):5.1f}%)  "
                f"unsat={unsat:5d}({pct(unsat,total):5.1f}%)  "
                f"unk={unk:5d}({pct(unk,total):5.1f}%)"
            )

        # Left parts (no bar yet)
        cur_left = f"[eval] {gen_str}  cur {fmt_counts(cur_total, cur_sat, cur_unsat, cur_unk)}"
        cum_left = f"            cum {fmt_counts(cum_total, cum_sat, cum_unsat, cum_unk)}"
        stop_left = f"            {stop_str}"

        # Pad to a fixed column, then append the bar
        print(cur_left.ljust(BAR_COL) + f"|{cur_bar}|")
        print(cum_left.ljust(BAR_COL) + f"|{cum_bar}|")
        print(stop_left)



    def store_dataset_all(self):
        return write_dataset_all(
            path=self.path,
            now=self.now,
            seed=self.seed,
            population=self.population,
            seed_ch=self.seed_ch,
            unknown=self.unknown,
            unsats=self.unsats,
            sats=self.sats,
            entire_dataset=self.entire_dataset,
            layout=self.formula_layout
        )

    def roulette_wheel_selection(self):
        population_fitness = sum([chromosome.fitness for chromosome in self.population])
        if population_fitness == 0:
            chromosome_probabilities = [1/len(self.population) for chromosome in self.population]
        else:
            chromosome_probabilities = [chromosome.fitness/population_fitness for chromosome in self.population]
        
        return deepcopy(np.random.choice(self.population, p=chromosome_probabilities))

    def check_evolution(self):
        if self.stopping_mode != "count":
            return self._check_adaptive_stopping()
        evolved = (len(self.sats) >= self.target_sats) and (len(self.unsats) >= self.target_sats)
        return (evolved)

    def _stop_criteria_line(self, cum_sat, cum_unsat, target) -> str:
        """Per-generation stop-criteria line, tailored to the stopping mode.

        In ``cv_pr`` mode it reports the current cross-validated precision/recall
        (the "PR" being evaluated) against the threshold instead of the raw
        sat/unsat counts; in ``tree_stable`` mode it reports tree stability.
        """
        if self.stopping_mode == "cv_pr":
            thr = getattr(self.stopping_config, "pr_threshold", 0.95)
            patience = getattr(self.stopping_config, "patience", 2)
            if self._last_pr is None:
                return (
                    f"stop criteria [cv_pr>= {thr:.3f}]: PR not yet evaluated "
                    f"(sat {cum_sat}, unsat {cum_unsat})"
                )
            gen, prec, rec, streak = self._last_pr
            prec_s = "n/a" if prec is None else f"{prec:.4f}"
            rec_s = "n/a" if rec is None else f"{rec:.4f}"
            return (
                f"stop criteria [cv_pr>= {thr:.3f}]: precision={prec_s} "
                f"recall={rec_s}  (gen {gen}, streak {streak}/{patience})"
            )
        if self.stopping_mode == "tree_stable":
            patience = getattr(self.stopping_config, "patience", 2)
            if self._last_tree_stable is None:
                return "stop criteria [tree_stable]: tree not yet evaluated"
            gen, stable, streak = self._last_tree_stable
            return (
                f"stop criteria [tree_stable]: stable={stable}  "
                f"(gen {gen}, streak {streak}/{patience})"
            )
        return f"stop criteria: sat {cum_sat}/{target} & unsat {cum_unsat}/{target}"

    def _adaptive_decision_from_stats(self, stats: dict) -> bool:
        mode = self.stopping_mode
        if mode == "cv_pr":
            precision = stats.get("cv_precision")
            recall = stats.get("cv_recall")
            threshold = getattr(self.stopping_config, "pr_threshold", 0.95)
            ok = precision is not None and recall is not None
            ok = ok and precision >= threshold and recall >= threshold
        elif mode == "tree_stable":
            tree_hash = stats.get("tree_hash")
            ok = tree_hash is not None and tree_hash == self._last_tree_hash
            self._last_tree_hash = tree_hash
        else:
            ok = False

        if ok:
            self._stopping_success_streak += 1
        else:
            self._stopping_success_streak = 0
        return self._stopping_success_streak >= getattr(self.stopping_config, "patience", 2)

    def _check_adaptive_stopping(self) -> bool:
        total = len(self.sats) + len(self.unsats)
        max_samples = getattr(self.stopping_config, "max_samples", None)
        if max_samples is not None and total >= max_samples:
            self.stopping_checks.append({
                "generation": getattr(self, "generation_counter", 0),
                "samples": total,
                "stop": True,
                "reason": "max_samples",
            })
            return True
        if len(self.sats) == 0 or len(self.unsats) == 0:
            return False
        if total < getattr(self.stopping_config, "min_samples", 0):
            return False
        generation = getattr(self, "generation_counter", 0)
        every = getattr(self.stopping_config, "check_every_generations", 1)
        if generation % every != 0:
            return False

        # Single, overwritten stopping-check workspace: the ARFF, J48 model and
        # .out are reused every check instead of one directory per generation.
        # The per-generation history is kept in memory (self.stopping_checks) and
        # rendered to pr_growth.png at the end of the run.
        checks_dir = Path(self.path) / "stopping_checks"
        checks_dir.mkdir(parents=True, exist_ok=True)
        arff_path = write_dataset_sat_unsat(
            str(checks_dir),
            self.now,
            self.seed,
            self.seed_ch,
            self.sats,
            self.unsats,
            self.formula_layout,
            "current",
        )
        out_path = run_j48(arff_path, 1.0, str(checks_dir))
        try:
            out_text = Path(out_path).read_text(encoding="utf-8", errors="replace")
        except Exception:
            out_text = ""
        stats = parse_j48_out(out_text, include_stopping_metrics=True)
        stop = self._adaptive_decision_from_stats(stats)
        if self.stopping_mode == "cv_pr":
            self._last_pr = (
                generation,
                stats.get("cv_precision"),
                stats.get("cv_recall"),
                self._stopping_success_streak,
            )
        elif self.stopping_mode == "tree_stable":
            self._last_tree_stable = (
                generation,
                self._stopping_success_streak > 0,
                self._stopping_success_streak,
            )
        self.stopping_checks.append({
            "generation": generation,
            "samples": total,
            "stop": stop,
            **stats,
        })
        return stop

    def _write_pr_plot(self) -> None:
        """Plot cross-validated precision/recall vs generation for cv_pr runs.

        Saves ``pr_growth.png`` in the run directory. The per-generation history
        comes from the in-memory ``self.stopping_checks`` records, so no extra
        per-generation files are needed. A no-op unless in cv_pr mode with data.
        """
        if self.stopping_mode != "cv_pr":
            return
        points = [
            c for c in self.stopping_checks
            if c.get("cv_precision") is not None or c.get("cv_recall") is not None
        ]
        if not points:
            return
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as exc:  # matplotlib missing/broken: skip gracefully
            print(f"[diagnosis] PR plot skipped (matplotlib unavailable): {exc}")
            return

        gens = [c.get("generation") for c in points]
        prec = [c.get("cv_precision") for c in points]
        rec = [c.get("cv_recall") for c in points]
        threshold = getattr(self.stopping_config, "pr_threshold", None)

        # Auto-zoom the y-axis to the observed values (they usually cluster near
        # the top, e.g. ~0.98), keeping the threshold line in view, so the growth
        # is legible instead of a flat band under a full 0..1 axis.
        observed = [v for v in (prec + rec) if v is not None]
        if threshold is not None:
            observed.append(threshold)
        lo, hi = min(observed), max(observed)
        span = hi - lo
        pad = max(0.01, span * 0.15)
        ymin = max(0.0, lo - pad)
        ymax = min(1.005, hi + pad)
        if ymax - ymin < 0.04:  # avoid over-zooming when everything is ~equal
            ymin = max(0.0, hi - 0.04)
            ymax = min(1.005, hi + 0.01)

        try:
            fig, ax = plt.subplots(figsize=(8, 4.5))
            ax.plot(gens, prec, marker="o", ms=3, label="CV precision")
            ax.plot(gens, rec, marker="s", ms=3, label="CV recall")
            if threshold is not None:
                ax.axhline(threshold, color="grey", ls="--", lw=1,
                           label=f"threshold {threshold:.3f}")
            ax.set_xlabel("generation")
            ax.set_ylabel("weighted CV precision / recall")
            ax.set_title("Precision/Recall growth over generations")
            ax.set_ylim(ymin, ymax)
            ax.grid(True, alpha=0.3)
            ax.legend(loc="lower right")
            fig.tight_layout()
            out = Path(self.path) / "pr_growth.png"
            fig.savefig(out, dpi=120)
            plt.close(fig)
            print(f"[diagnosis] wrote PR growth plot: {out}")
        except Exception as exc:
            print(f"[diagnosis] PR plot failed: {exc}")

    def checkin(self, logtype: str):
        self.checkin_start[logtype] = time.time()
        if CHECK_PRINT: print(f'Check in: {logtype} {self.checkin_start[logtype]} seconds')

    def checkout(self, logtype: str):
        self.checkin_start[logtype] = time.time() - self.checkin_start[logtype]
        if CHECK_PRINT: print(f'Check out: {logtype} {self.checkin_start[logtype]} seconds')
        self.timespan_log[logtype] = self.timespan_log[logtype] + self.checkin_start[logtype]
        if CHECK_PRINT: print(f'Timespan: {logtype} {self.timespan_log[logtype]} seconds')

    def write_timespan_log(self):
        json_object = json.dumps(self.timespan_log, indent=4)
        with open(f"{self.path}/timespan.json", "w") as outfile:
            outfile.write(json_object)

    def evolve(self):
        with open('{}/hypot.txt'.format(self.path), 'a') as f:
            for hypot in self.hypots:
                f.write(f'\t{hypot[1]}\n')
        # loop
        self.generation_counter = 0

        if self._run_adaptive_endpoint_initialization():
            self._write_report()
            self.close()
            return {}
        self._adapt_initial_population_after_endpoint_init()
       
        self.generate_dataset_qty()
        s100 = self.store_dataset_all()
        
        self.checkin('tracheck_timestamp')
        self.evaluate()
        self._update_stats_after_evaluate()
        self.checkout('tracheck_timestamp')
        
        while (not self.check_evolution()) and (self.generation_counter < self.max_generations):
            self.checkin('mutation_timestamp')
            self.population.sort(key=lambda x: x.fitness, reverse=True)
            self.write_population(self.generation_counter)
            self.generate_dataset_qty()
            s100 = self.store_dataset_all()

            new_population = self.population[:CHROMOSOME_TO_PRESERVE]

            terminators = list(set(treenode.get_terminators(self.seed)))
            
            population_counter = CHROMOSOME_TO_PRESERVE
            while(population_counter < self.size):
                offspring1 = self.roulette_wheel_selection()
                offspring2 = self.roulette_wheel_selection()

                self.crossover(offspring1, offspring2)

                # Offspring 1 mutation
                if self.mutation_config is not None:
                    offspring1.mutate(
                        self.mutation_rate,
                        mutation_config=self.mutation_config,
                    )

                # Offspring 2 mutation
                if self.mutation_config is not None:
                    offspring2.mutate(
                        self.mutation_rate,
                        mutation_config=self.mutation_config,
                    )

                self._maybe_adapt_range(offspring1)
                self._maybe_adapt_range(offspring2)

                new_population.append(offspring1)
                new_population.append(offspring2)

                # Reset fitness 
                offspring1.reset()
                offspring2.reset()

                # Reset fitness 
                offspring1.reset()
                offspring2.reset()

                population_counter += 2
            self.generation_counter += 1
            self.population = new_population
            self.checkout('mutation_timestamp')

            # self.diagnosis()

            # write population before trace checker
            self.generate_dataset_qty()
            s100 = self.store_dataset_all()

            ## score population
            self.checkin('tracheck_timestamp')
            self.evaluate()
            self._update_stats_after_evaluate()

            s100 = write_dataset_qty(self.path, self.now, self.seed, self.seed_ch, self.sats, self.unsats, self.unknown, self.formula_layout, per_cut=1.0)
            s025 = write_dataset_qty(self.path, self.now, self.seed, self.seed_ch, self.sats, self.unsats, self.unknown, self.formula_layout, per_cut=.25)
            s020 = write_dataset_qty(self.path, self.now, self.seed, self.seed_ch, self.sats, self.unsats, self.unknown, self.formula_layout, per_cut=.20)
            s015 = write_dataset_qty(self.path, self.now, self.seed, self.seed_ch, self.sats, self.unsats, self.unknown, self.formula_layout, per_cut=.15)
            s010 = write_dataset_qty(self.path, self.now, self.seed, self.seed_ch, self.sats, self.unsats, self.unknown, self.formula_layout, per_cut=.10)
            self.store_dataset_all()
            self.checkout('tracheck_timestamp')

            self.write_timespan_log()

        self.checkin('diagnosi_timestamp')
        s100 = write_dataset_qty(self.path, self.now, self.seed, self.seed_ch, self.sats, self.unsats, self.unknown, self.formula_layout, per_cut=1.0)
        s025 = write_dataset_qty(self.path, self.now, self.seed, self.seed_ch, self.sats, self.unsats, self.unknown, self.formula_layout, per_cut=.25)
        s020 = write_dataset_qty(self.path, self.now, self.seed, self.seed_ch, self.sats, self.unsats, self.unknown, self.formula_layout, per_cut=.20)
        s015 = write_dataset_qty(self.path, self.now, self.seed, self.seed_ch, self.sats, self.unsats, self.unknown, self.formula_layout, per_cut=.15)
        s010 = write_dataset_qty(self.path, self.now, self.seed, self.seed_ch, self.sats, self.unsats, self.unknown, self.formula_layout, per_cut=.10)
        self.checkout('diagnosi_timestamp')
        self.write_timespan_log()
        self._write_pr_plot()

        # Return the ARFF paths for the diagnostics layer (pipeline) to consume.
        self.close()
        return {
            1.0: s100,
            0.25: s025,
            0.20: s020,
            0.15: s015,
            0.10: s010,
        }



    def replace_token(self, tk_list):
        l = list()
        for tk in tk_list:
            if tk in [">=", "<="]:
                l.append(tk.value[0])
                l.append(tk.value[1])
            else:
                l.append(tk)
        return l

    def save_file(self, nline: str):
        """
        Create self.path/temp.py by copying the original property script
        (self.property_path) and replacing the property z3solver.add(Not(ForAll(...)))
        line with the given expression.
        """
        src = self.property_path or self.base_property_script
        if not src:
            raise RuntimeError("[diagnosis] save_file: no property_path/base_property_script set")

        dst = f"{self.path}/temp.py"

        write_candidate_script(src, dst, nline)

    def _madeit_from_result(self, result):
        err = ""
        if result.verdict == Verdict.SAT:
            madeit = "True"
        elif result.verdict == Verdict.UNSAT:
            madeit = "False"
        else:
            out = (result.stdout or "") + (result.stderr or "")
            if "UNDECIDED" in out:
                madeit = "Unknown"
                err = "REQUIREMENT UNDECIDED"
            else:
                print(result.stdout)
                print(result.stderr)
                madeit = "Problem"
        return madeit, err

    def _madeit_from_worker_verdict(self, verdict: str) -> tuple[str, str]:
        if verdict == V_SATISFIED:
            return "True", ""
        if verdict == V_VIOLATED:
            return "False", ""
        if verdict == V_UNDECIDED:
            return "Unknown", "REQUIREMENT UNDECIDED"
        if verdict == V_ERROR:
            return "Problem", ""
        return "Problem", ""

    def _ensure_worker(self) -> SolverWorker:
        if self.worker is None:
            self.worker = SolverWorker(
                self.property_path or self.base_property_script,
                log_path=Path(self.path) / "worker.log",
            )
        if not self.worker.is_alive():
            self.worker.start()
        return self.worker

    def _worker_check(self, expression: str) -> tuple[str, float]:
        worker = self._ensure_worker()
        return worker.check(expression, self.trace_check_timeout_sec * 1000)

    def _solve_once(
        self,
        nline: str,
        script_path,
        timeout_sec: int,
        worker: "SolverWorker | None" = None,
    ) -> tuple[str, str, float, int, int]:
        """Solve one candidate with a per-call z3 timeout of ``timeout_sec``.

        Returns ``(madeit, err, solve_seconds, worker_restarts, worker_fallbacks)``.
        A z3 timeout is only injected when the two-tier heuristic is active; when
        it is off, ``timeout_sec`` equals ``trace_check_timeout_sec`` and the
        behaviour matches the baseline (no injection).
        """
        inject = self.heuristics.two_tier
        z3_timeout_ms = int(timeout_sec) * 1000 if inject else None
        src = self.property_path or self.base_property_script
        if not src:
            raise RuntimeError("[diagnosis] solve: no property_path/base_property_script set")
        # Subprocess wall backstop: a little over the z3 cap when injecting.
        wall = self.trace_check_timeout_sec if not inject else min(
            self.trace_check_timeout_sec, int(timeout_sec) + 60
        )
        worker_restarts = 0
        worker_fallbacks = 0
        start = time.time()

        if self.engine == "worker":
            w = worker if worker is not None else self._ensure_worker()
            try:
                verdict, solve_seconds = w.check(nline, int(timeout_sec) * 1000)
                madeit, err = self._madeit_from_worker_verdict(verdict)
                return madeit, err, solve_seconds, worker_restarts, worker_fallbacks
            except WorkerCrash:
                worker_restarts += 1
                try:
                    w.restart()
                except WorkerCrash:
                    pass
                worker_fallbacks += 1
                # fall through to subprocess fallback

        write_candidate_script(src, script_path, nline, z3_timeout_ms=z3_timeout_ms)
        result = run_property_script(script_path, timeout=wall)
        solve_seconds = time.time() - start
        madeit, err = self._madeit_from_result(result)
        return madeit, err, solve_seconds, worker_restarts, worker_fallbacks

    def _run_tiers(
        self, nline: str, script_path, timeouts: list[int], worker=None
    ) -> tuple[str, str, float, list[str], int, int]:
        """Run the ordered timeout tiers, stopping at the first decided verdict."""
        tier_verdicts: list[str] = []
        madeit, err, total_seconds = "Problem", "", 0.0
        restarts = fallbacks = 0
        for t in timeouts:
            madeit, err, secs, wr, wf = self._solve_once(nline, script_path, t, worker=worker)
            tier_verdicts.append(madeit)
            total_seconds += secs
            restarts += wr
            fallbacks += wf
            if madeit != "Unknown":
                break
        return madeit, err, total_seconds, tier_verdicts, restarts, fallbacks

    def close(self):
        if self.worker is not None:
            self.worker.stop()
            self.worker = None
        if self.heuristics.any_on:
            self.heuristics.persist()
            self.heuristics.write_sidecar(self.path)

    def _apply_madeit(self, chromosome, madeit: str, err: str):
        chromosome.madeit = madeit

        if chromosome.madeit == 'Problem':
            # When we cannot evaluate the chromosome set fitness to zero and evaluate next chromosome
            chromosome.sw_score = 0
            chromosome.fitness = 0
            return

        if err in self.execution_report.keys():
            self.execution_report[err] += 1
        else:
            self.execution_report[err] = 1
        self.execution_report['TOTAL'] += 1

        ## running sw
        seed_tokens = self.replace_token(list(self.seed))
        chrom_tokens = self.replace_token(list(chromosome))

        result_seed = self.fitness.score(seed_tokens, chrom_tokens)

        # Keep sw_score for backwards compatibility / ARFF
        chromosome.sw_score = result_seed

        # Normalized fitness in [0, 100] based on max_score
        self.get_max_score()
        if self.max_score > 0:
            chromosome.fitness = int(100 * (result_seed) / self.max_score)
        else:
            chromosome.fitness = 0

        # Track the best satisfied chromosome seen so far for logging
        if chromosome.madeit == 'True':
            if self.highest_sat is None or chromosome.fitness > self.highest_sat.fitness:
                self.highest_sat = chromosome

    def _flag_vacuous(self, chromosome) -> None:
        """Flag a candidate whose mutable time window is empty (Sprint 7).

        A vacuous ForAll is trivially SATISFIED, so the row must not feed the tree
        as an ordinary SAT; the ``vacuous`` flag is consumed by the ARFF-inclusion
        filter. A no-op when quantization is off (flag stays ``False``).
        """
        if chromosome is None:
            return
        chromosome.vacuous = self.heuristics.quant_vacuous(getattr(chromosome, "ast", None))

    def _ckey(self, cand_ast, nline: str) -> str:
        """Canonical verdict-cache key for a candidate.

        Equals ``nline`` when time quantization is off, so the cache behaves
        exactly as before; otherwise quantizable time tokens collapse to their
        sample-class indices.
        """
        return self.heuristics.quant_cache_expr(cand_ast, nline)

    def _raw_solve(self, nline: str) -> str:
        """Solve a raw candidate ignoring the cache (used by the quant validator)."""
        src = self.property_path or self.base_property_script
        script_path = Path(self.path) / "temp_quant_validate.py"
        write_candidate_script(src, script_path, nline)
        result = run_property_script(script_path, timeout=self.trace_check_timeout_sec)
        madeit, _err = self._madeit_from_result(result)
        return madeit

    def _quant_validate_after_hit(self, cand_ast, ckey: str, nline: str, cached_verdict: str) -> None:
        """Run the periodic same-class double-solve check behind a cache hit.

        Every ``validate_every_n_hits`` canonical hits on a class, re-solve the new
        raw value and compare; a mismatch disables quantization for the position,
        purges the class's cache entries, and logs a witness (violations must be 0
        in a valid run).
        """
        if not self.heuristics.quant_on_lookup(ckey, nline, hit=True):
            return
        real = self._raw_solve(nline)
        purge = self.heuristics.quant_validate(cand_ast, ckey, cached_verdict, real)
        if purge:
            with self.cache_lock:
                self.verdict_cache.purge_expressions(purge)

    def _pending_evaluations(self):
        pending = []
        for idx, chromosome in enumerate(self.population):
            if chromosome.fitness != -1:
                continue
            wrapped: Formula = Not(chromosome.ast)
            nline = formula_to_python_expr(wrapped)
            pending.append((idx, chromosome, nline))
        return pending

    def _evaluate_parallel_candidate(self, idx: int, chromosome_ast, nline: str, worker_pool):
        ckey = self._ckey(chromosome_ast, nline)
        with self.cache_lock:
            cached = self.verdict_cache.get(ckey)
        if cached is not None:
            madeit, _solve_seconds = cached
            self._quant_validate_after_hit(chromosome_ast, ckey, nline, madeit)
            return {
                "idx": idx,
                "madeit": madeit,
                "err": "",
                "worker_restarts": 0,
                "worker_fallbacks": 0,
                "real": False,
                "inferred": False,
                "include_in_arff": True,
                "plan": None,
                "tier_verdicts": [],
            }

        src = self.property_path or self.base_property_script
        if not src:
            raise RuntimeError("[diagnosis] evaluate: no property_path/base_property_script set")
        script_path = Path(self.path) / f"temp_{idx}.py"

        # --- heuristic plan-driven path (plans read a per-generation snapshot) ---
        if self.heuristics.any_on:
            plan = self.heuristics.plan(chromosome_ast)
            if plan.inferred is not None:
                return {
                    "idx": idx,
                    "madeit": plan.inferred,
                    "err": plan.err,
                    "worker_restarts": 0,
                    "worker_fallbacks": 0,
                    "real": False,
                    "inferred": True,
                    "include_in_arff": plan.include_in_arff,
                    "plan": plan,
                    "tier_verdicts": [],
                }
            worker = worker_pool.get() if self.engine == "worker" else None
            try:
                madeit, err, solve_seconds, tier_verdicts, wr, wf = self._run_tiers(
                    nline, script_path, plan.timeouts, worker=worker
                )
            finally:
                if worker is not None:
                    worker_pool.put(worker)
            self.heuristics.quant_on_lookup(ckey, nline, hit=False)
            with self.cache_lock:
                self.verdict_cache.put(ckey, madeit, solve_seconds)
            return {
                "idx": idx,
                "madeit": madeit,
                "err": err,
                "worker_restarts": wr,
                "worker_fallbacks": wf,
                "real": True,
                "inferred": False,
                "include_in_arff": True,
                "plan": plan,
                "tier_verdicts": tier_verdicts,
            }

        start = time.time()
        worker_restarts = 0
        worker_fallbacks = 0

        if self.engine == "worker":
            worker = worker_pool.get()
            try:
                try:
                    verdict, solve_seconds = worker.check(
                        nline,
                        self.trace_check_timeout_sec * 1000,
                    )
                    madeit, err = self._madeit_from_worker_verdict(verdict)
                except WorkerCrash:
                    worker_restarts += 1
                    try:
                        worker.restart()
                    except WorkerCrash:
                        pass
                    worker_fallbacks += 1
                    write_candidate_script(src, script_path, nline)
                    result = run_property_script(
                        script_path,
                        timeout=self.trace_check_timeout_sec,
                    )
                    solve_seconds = time.time() - start
                    madeit, err = self._madeit_from_result(result)
            finally:
                worker_pool.put(worker)
        else:
            write_candidate_script(src, script_path, nline)
            result = run_property_script(script_path, timeout=self.trace_check_timeout_sec)
            solve_seconds = time.time() - start
            madeit, err = self._madeit_from_result(result)

        self.heuristics.quant_on_lookup(ckey, nline, hit=False)
        with self.cache_lock:
            self.verdict_cache.put(ckey, madeit, solve_seconds)
        return {
            "idx": idx,
            "madeit": madeit,
            "err": err,
            "worker_restarts": worker_restarts,
            "worker_fallbacks": worker_fallbacks,
            "real": True,
            "inferred": False,
            "include_in_arff": True,
            "plan": None,
            "tier_verdicts": [],
        }

    def _evaluate_parallel(self, pending):
        worker_pool = queue.Queue()
        workers = []
        if self.engine == "worker":
            for worker_idx in range(self.parallel_workers):
                worker = SolverWorker(
                    self.property_path or self.base_property_script,
                    log_path=Path(self.path) / f"worker_{worker_idx}.log",
                )
                worker.start()
                workers.append(worker)
                worker_pool.put(worker)

        results = {}
        try:
            with ThreadPoolExecutor(max_workers=self.parallel_workers) as executor:
                future_to_idx = {
                    executor.submit(
                        self._evaluate_parallel_candidate,
                        idx,
                        chromosome.ast,
                        nline,
                        worker_pool,
                    ): idx
                    for idx, chromosome, nline in pending
                }
                for future in self._progress(
                    as_completed(future_to_idx),
                    desc=f"Evaluating gen {getattr(self, 'generation_counter', 0)}",
                ):
                    item = future.result()
                    results[item["idx"]] = item
                    self.worker_restarts += int(item["worker_restarts"])
                    self.worker_fallbacks += int(item["worker_fallbacks"])
        finally:
            for worker in workers:
                worker.stop()

        for idx, chromosome, _nline in sorted(pending, key=lambda item: item[0]):
            item = results[idx]
            # Region updates are applied here, in deterministic idx order, so a
            # generation's plans all saw the same pre-generation region snapshot.
            if item.get("real") and item.get("plan") is not None:
                self.heuristics.record_solve(
                    item["plan"], item["tier_verdicts"], item["madeit"]
                )
            chromosome.inferred = bool(item.get("inferred", False))
            chromosome.include_in_arff = bool(item.get("include_in_arff", True))
            self._flag_vacuous(chromosome)
            if self.heuristics.any_on:
                self.heuristics.record_arff_row(
                    chromosome.arrf_str(), item["madeit"], chromosome.inferred
                )
            self._apply_madeit(chromosome, item["madeit"], item["err"])

    def evaluate(self):
        if self.parallel_workers > 1:
            pending = self._pending_evaluations()
            self._evaluate_parallel(pending)
            self._report_cache_stats()
            return

        for idx, chromosome in enumerate(self._progress(self.population, desc=f"Evaluating gen {getattr(self, 'generation_counter', 0)}")
):
            if chromosome.fitness != -1:
                continue

            # Wrap the ForAll(...) formula back into a Not(...) for trace checking
            wrapped: Formula = Not(chromosome.ast)
            nline = formula_to_python_expr(wrapped)
            ckey = self._ckey(chromosome.ast, nline)
            self._flag_vacuous(chromosome)

            self.save_file(nline)

            script_path = Path(self.path) / "temp.py"

            cached = self.verdict_cache.get(ckey)
            if cached is not None:
                madeit, solve_seconds = cached
                self._mark_real(chromosome)
                self._quant_validate_after_hit(chromosome.ast, ckey, nline, madeit)
                self._apply_madeit(chromosome, madeit, "")
            elif self.heuristics.any_on:
                self._evaluate_with_heuristics(chromosome, nline, script_path)
            else:
                start = time.time()
                if self.engine == "worker":
                    try:
                        verdict, solve_seconds = self._worker_check(nline)
                        madeit, err = self._madeit_from_worker_verdict(verdict)
                    except WorkerCrash:
                        self.worker_restarts += 1
                        try:
                            if self.worker is not None:
                                self.worker.restart()
                        except WorkerCrash:
                            pass
                        self.worker_fallbacks += 1
                        result = run_property_script(
                            script_path,
                            timeout=self.trace_check_timeout_sec,
                        )
                        solve_seconds = time.time() - start
                        madeit, err = self._madeit_from_result(result)
                else:
                    result = run_property_script(script_path, timeout=self.trace_check_timeout_sec)
                    solve_seconds = time.time() - start
                    madeit, err = self._madeit_from_result(result)
                self.heuristics.quant_on_lookup(ckey, nline, hit=False)
                self.verdict_cache.put(ckey, madeit, solve_seconds)
                self._apply_madeit(chromosome, madeit, err)

        self._report_cache_stats()

    def _mark_real(self, chromosome) -> None:
        chromosome.inferred = False
        chromosome.include_in_arff = True

    def _run_adaptive_endpoint_initialization(self) -> bool:
        ar = getattr(self.heuristics_config, "adaptive_range", None)
        if not (ar and ar.enabled and ar.endpoint_init):
            return False
        if self.seed_ast is None or self.mutation_config is None:
            return False

        positions = self.heuristics.numeric_range_positions(
            self.mutation_config.allowed_changes
        )
        for position in positions:
            bounds = self.mutation_config.allowed_changes[position].get("numeric")
            if not (isinstance(bounds, (list, tuple)) and len(bounds) == 2):
                continue
            lo, hi = float(bounds[0]), float(bounds[1])
            if lo > hi:
                lo, hi = hi, lo
            if self._probe_adaptive_endpoint_range(position, (lo, hi)):
                return True
        return False

    def _adapt_initial_population_after_endpoint_init(self) -> None:
        ar = getattr(self.heuristics_config, "adaptive_range", None)
        if not (ar and ar.enabled and ar.endpoint_init):
            return
        for chromosome in self.population:
            self._maybe_adapt_range(chromosome)

    def _probe_adaptive_endpoint_range(
        self, position: int, bounds: tuple[float, float]
    ) -> bool:
        ar = self.heuristics_config.adaptive_range
        current = bounds
        attempts = 0
        while True:
            witnesses = []
            verdicts = []
            for role, value in (("lo", current[0]), ("hi", current[1])):
                madeit = self._evaluate_endpoint_probe(position, value, role)
                witness = {"role": role, "value": float(value), "verdict": madeit}
                witnesses.append(witness)
                verdicts.append(madeit)

            decisive = all(v in ("True", "False") for v in verdicts)
            if not decisive:
                print(
                    f"[diagnosis][adaptive_range] endpoint probe for position "
                    f"{position} was non-decisive; continuing GA"
                )
                return False
            if verdicts[0] != verdicts[1]:
                return False

            action = ar.on_one_class
            # A quantization-managed window bound that is one-class over its range
            # only means its sample-aligned boundary lies outside the sampled range
            # (the window never flips the verdict here). That is a recorded finding,
            # not a reason to halt or widen the whole run -- the other knobs still
            # drive the search, and the endpoint probes above already seeded this
            # knob's class-bracket observations. Force "continue" for such knobs so
            # promoting bounds to monotone (only with quantization on) never turns a
            # productive multi-knob run into a generation-0 stop.
            if self.heuristics.quant_knob(position):
                action = "continue"
            if action == "continue":
                self.heuristics.record_one_class_space(
                    position, current, verdicts[0], witnesses, "continue"
                )
                print(
                    f"[diagnosis][adaptive_range] position {position} is "
                    f"one-class in [{current[0]}, {current[1]}]; continuing GA"
                )
                return False

            if action == "widen" and attempts < ar.max_widenings:
                widened = self._widen_adaptive_bounds(position, current)
                self.heuristics.record_widening(
                    position, current, widened, ar.widen_factor
                )
                print(
                    f"[diagnosis][adaptive_range] widened position {position} "
                    f"from [{current[0]}, {current[1]}] to "
                    f"[{widened[0]}, {widened[1]}]"
                )
                current = widened
                attempts += 1
                continue

            finding = self.heuristics.record_one_class_space(
                position, current, verdicts[0], witnesses, "report_and_stop"
            )
            self._adaptive_early_stop = True
            self._adaptive_early_stop_message = (
                f"mutating position {position} in "
                f"[{current[0]}, {current[1]}] cannot flip the verdict; "
                "widen the range or choose other tokens"
            )
            self.execution_report["adaptive_range_stop"] = self._adaptive_early_stop_message
            self.execution_report["one_class_space"] = [finding]
            print(f"[diagnosis][adaptive_range] {self._adaptive_early_stop_message}")
            return True

    def _widen_adaptive_bounds(
        self, position: int, bounds: tuple[float, float]
    ) -> tuple[float, float]:
        lo, hi = bounds
        span = max(hi - lo, 1.0)
        extra = span * (float(self.heuristics_config.adaptive_range.widen_factor) - 1.0)
        direction = self.heuristics._all_directions.get(position)
        if direction is Monotonicity.DECREASING:
            return (lo - extra, hi)
        return (lo, hi + extra)

    def _evaluate_endpoint_probe(self, position: int, value: float, role: str) -> str:
        ast = self.seed_ast
        if ast is None:
            return "Problem"
        from .inference import set_numeric_at_position

        endpoint_ast = set_numeric_at_position(ast, position, value)
        wrapped: Formula = Not(endpoint_ast)
        nline = formula_to_python_expr(wrapped)
        ckey = self._ckey(endpoint_ast, nline)

        cached = self.verdict_cache.get(ckey)
        if cached is not None:
            madeit, _solve_seconds = cached
            self.heuristics.record_endpoint_probe(position, value, madeit, role)
            return madeit

        planned = self.heuristics.plan(endpoint_ast)
        timeouts = planned.timeouts or [self.trace_check_timeout_sec]
        plan = Plan(
            timeouts=timeouts,
            vector=planned.vector,
            positions=planned.positions,
            position=None,
            value=None,
            decision=planned.decision,
        )

        script_path = Path(self.path) / f"endpoint_{position}_{role}.py"
        madeit, err, solve_seconds, tier_verdicts, wr, wf = self._run_tiers(
            nline, script_path, plan.timeouts
        )
        self.worker_restarts += wr
        self.worker_fallbacks += wf
        self.heuristics.record_solve(plan, tier_verdicts, madeit)
        self.heuristics.record_endpoint_probe(position, value, madeit, role)
        self.verdict_cache.put(ckey, madeit, solve_seconds)
        if err:
            self.execution_report[err] = self.execution_report.get(err, 0) + 1
        return madeit

    def _maybe_adapt_range(self, chromosome) -> None:
        """Apply opt-in adaptive-range sampling to a single-position mutant."""
        if not self.heuristics.any_on or chromosome is None:
            return
        if self.mutation_config is None:
            return
        ast = getattr(chromosome, "ast", None)
        if ast is None:
            return
        adapted = self.heuristics.adapt_candidate(
            ast,
            self.mutation_config.allowed_changes,
            random,
        )
        if adapted is not None:
            chromosome.ast = adapted
            chromosome._sync_root_from_ast()

    def _evaluate_with_heuristics(self, chromosome, nline: str, script_path) -> None:
        """Plan-driven evaluation: infer / two-tier solve, then bookkeeping.

        Inferred verdicts are not written to the verdict cache so that repeats
        are re-planned and re-marked (keeps guide-mode ARFF exclusion consistent);
        real solves are cached exactly as before.
        """
        plan = self.heuristics.plan(chromosome.ast)
        if plan.inferred is not None:
            chromosome.inferred = True
            chromosome.include_in_arff = plan.include_in_arff
            self.heuristics.record_arff_row(
                chromosome.arrf_str(), plan.inferred, True
            )
            self._apply_madeit(chromosome, plan.inferred, plan.err)
            return

        madeit, err, solve_seconds, tier_verdicts, wr, wf = self._run_tiers(
            nline, script_path, plan.timeouts
        )
        self.worker_restarts += wr
        self.worker_fallbacks += wf
        self.heuristics.record_solve(plan, tier_verdicts, madeit)
        self._mark_real(chromosome)
        self.heuristics.record_arff_row(chromosome.arrf_str(), madeit, False)
        ckey = self._ckey(chromosome.ast, nline)
        self.heuristics.quant_on_lookup(ckey, nline, hit=False)
        self.verdict_cache.put(ckey, madeit, solve_seconds)
        self._apply_madeit(chromosome, madeit, err)


    def crossover(self, parent1: Individual, parent2: Individual):
        offsprings = [parent1, parent2]

        # Guard: if either offspring is too short, skip crossover.
        min_len = min(len(offsprings[0]), len(offsprings[1]))
        if min_len <= 2:
            # Nothing sensible to cross; return parents as-is.
            return offsprings

        # Choose a cut index that is valid for both trees.
        cut_idx = random.randint(1, min_len - 2)

        try:
            off0_tree, off0_sub, node0 = offsprings[0].root.cut_tree(cut_idx)
            off1_tree, off1_sub, node1 = offsprings[1].root.cut_tree(cut_idx)
        except Exception:
            # If cut_tree fails for any reason, fall back to unmodified parents.
            return offsprings

        node0 += off1_sub
        node1 += off0_sub

        return offsprings
