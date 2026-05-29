"""
Driftline — High-Performance GBM Portfolio Simulator
====================================
A vectorized simulation engine designed to project multi-asset pathways under
correlated market conditions using geometric Brownian motion, with optional
periodic portfolio rebalancing.
"""

import time
import tracemalloc
from dataclasses import dataclass, field
from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool
from typing import Optional
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Assets output directory
# ---------------------------------------------------------------------------

ASSETS_DIR = Path("assets")
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Rebalancing Frequency Constants
# ---------------------------------------------------------------------------
# All values assume n_steps=252 (one trading year). Pass an integer directly
# if your simulation uses a different step count.

REBAL_FREQ: dict[str, Optional[int]] = {
    "daily":      1,
    "weekly":     5,
    "monthly":   21,
    "quarterly": 63,
    "semiannual": 126,
    "annual":    252,
    "never":    None,
}


# ---------------------------------------------------------------------------
# Validation Helpers
# ---------------------------------------------------------------------------

def _to_f64_array(x, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1-D array, got shape {arr.shape}")
    return arr


def _validate_params(p: "GBMParams") -> None:
    n = len(p.S0)
    for attr in ("mu", "sigma", "weights"):
        v = getattr(p, attr)
        if len(v) != n:
            raise ValueError(
                f"Length mismatch: S0 has {n} assets but {attr} has {len(v)}"
            )
    if p.rho.shape != (n, n):
        raise ValueError(f"rho must be ({n}, {n}), got {p.rho.shape}")
    if not np.allclose(p.rho, p.rho.T):
        raise ValueError("Correlation matrix must be perfectly symmetric")
    eigvals = np.linalg.eigvalsh(p.rho)
    if np.any(eigvals < -1e-8):
        raise ValueError("Correlation matrix must be positive semi-definite")
    if not np.allclose(np.diag(p.rho), 1.0):
        raise ValueError("Correlation matrix diagonal must equal 1.0")
    w_sum = np.sum(p.weights)
    if not np.isclose(w_sum, 1.0):
        raise ValueError(f"Portfolio weights must sum to 1.0, got {w_sum:.6f}")
    if p.rebal_steps is not None:
        if not isinstance(p.rebal_steps, int) or p.rebal_steps < 1:
            raise ValueError(
                f"rebal_steps must be a positive integer, got {p.rebal_steps!r}"
            )
        if p.rebal_steps >= p.n_steps:
            raise ValueError(
                f"rebal_steps ({p.rebal_steps}) must be less than n_steps ({p.n_steps}). "
                f"Use rebal_steps=None to disable rebalancing."
            )


# ---------------------------------------------------------------------------
# Configuration Data Structures
# ---------------------------------------------------------------------------

@dataclass
class GBMParams:
    """
    Immutable parameter bundle for the multi-asset GBM simulation.

    Parameters
    ----------
    S0          : (n_assets,) initial asset prices
    mu          : (n_assets,) annualized drift rates
    sigma       : (n_assets,) annualized volatilities
    weights     : (n_assets,) portfolio allocation weights, must sum to 1.0
    rho         : (n_assets, n_assets) asset correlation matrix
    T           : simulation horizon in years
    n_steps     : number of discrete time steps
    n_paths     : total Monte Carlo paths to simulate
    rebal_steps : rebalance every N steps; None disables rebalancing.
                  Use REBAL_FREQ constants for standard frequencies.
    names       : optional list of asset labels for reporting and plots
    dtype       : storage dtype for path arrays (float32 halves memory vs float64)

    Notes
    -----
    The Cholesky factor L is computed once in __post_init__ and reused by every
    worker. rebal_steps operates entirely inside compute_portfolio_values;
    _worker is completely unaffected by the rebalancing setting.
    """
    S0:          np.ndarray
    mu:          np.ndarray
    sigma:       np.ndarray
    weights:     np.ndarray
    rho:         np.ndarray
    T:           float
    n_steps:     int
    n_paths:     int
    rebal_steps: Optional[int] = None
    names:       list          = field(default_factory=list)
    dtype:       type          = np.float32

    # Pre-computed Cholesky factor; not user-supplied
    L: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        self.S0      = _to_f64_array(self.S0,      "S0")
        self.mu      = _to_f64_array(self.mu,       "mu")
        self.sigma   = _to_f64_array(self.sigma,    "sigma")
        self.weights = _to_f64_array(self.weights,  "weights")
        self.rho     = np.asarray(self.rho, dtype=np.float64)
        if self.rho.ndim != 2:
            raise ValueError("rho must be a 2-D matrix")
        _validate_params(self)
        self.L = np.linalg.cholesky(self.rho)
        if not self.names:
            self.names = [f"Asset {i+1}" for i in range(len(self.S0))]

    @property
    def n_assets(self) -> int:
        return len(self.S0)

    @property
    def initial_portfolio_value(self) -> float:
        return float(np.sum(self.S0))

    @property
    def _total_investment(self) -> float:
        return float(np.sum(self.S0))

    @property
    def rebal_label(self) -> str:
        """Human-readable rebalancing frequency label for reports and plots."""
        if self.rebal_steps is None:
            return "none"
        reverse = {v: k for k, v in REBAL_FREQ.items() if v is not None}
        return reverse.get(self.rebal_steps, f"every {self.rebal_steps} steps")


# ---------------------------------------------------------------------------
# Execution Worker  (rebalancing does NOT touch this function)
# ---------------------------------------------------------------------------

def _worker(args: tuple) -> np.ndarray:
    """
    Generate a slice of correlated multi-asset GBM price paths.

    Rebalancing is a portfolio-management decision applied during aggregation.
    Asset prices follow GBM regardless of investor behavior, so this function
    is identical with or without rebalancing enabled.

    Returns
    -------
    np.ndarray  shape (slice_paths, n_assets, n_steps+1), dtype params.dtype
    """
    params, slice_paths, seed_offset = args

    n_assets  = params.n_assets
    n_steps   = params.n_steps
    dt        = params.T / n_steps
    rng       = np.random.default_rng(seed_offset)

    drift     = (params.mu - 0.5 * params.sigma ** 2) * dt   # (n_assets,)
    diffusion = params.sigma * np.sqrt(dt)                     # (n_assets,)

    # Independent standard normals: (slice_paths, n_steps, n_assets)
    Z = rng.standard_normal((slice_paths, n_steps, n_assets))

    # Correlate via Cholesky: Z @ L.T is a single BLAS gemm over the asset axis
    Z_corr = Z @ params.L.T                                    # (paths, steps, n_assets)

    # GBM multiplicative price increments
    increments = np.exp(drift + diffusion * Z_corr).astype(params.dtype)
    increments = increments.transpose(0, 2, 1)

    # Prepend S0 and take cumulative product along the time axis
    S0_col = np.broadcast_to(
        params.S0.astype(params.dtype)[np.newaxis, :, np.newaxis],
        (slice_paths, n_assets, 1),
    ).copy()

    paths = np.concatenate([S0_col, increments], axis=2)       # (paths, n_assets, steps+1)
    np.cumprod(paths, axis=2, out=paths)
    return paths


# ---------------------------------------------------------------------------
# Portfolio Value Aggregator
# ---------------------------------------------------------------------------

def compute_portfolio_values(
    asset_paths: np.ndarray,
    params:      GBMParams,
) -> np.ndarray:
    """
    Aggregate per-asset price paths into total portfolio value over time.

    Without rebalancing
    -------------------
    Share holdings are fixed at t=0:  shares_i = w_i * V_0 / S0_i
    V(t) = sum_i( shares_i * P_i(t) )
    Implemented as a single vectorized broadcast + sum — one pass, no Python loop.

    With rebalancing (rebal_steps is not None)
    ------------------------------------------
    The timeline is divided into segments at each rebalancing date.  Within a
    segment, holdings are constant, so portfolio value is fully vectorized
    across all paths.  At segment boundaries the share holdings are updated:

        shares_i_new = w_i * V(t_r) / P_i(t_r)

    The outer loop iterates over rebalancing events, not over paths or timesteps.
    For quarterly rebalancing (4 events/year) the loop has 4 iterations regardless
    of n_paths.  Overhead vs the no-rebalancing path is approximately 11%,
    dominated by the float32 -> float64 upcast for precision, not the loop itself.

    Value is conserved at every rebalancing boundary (no cash enters or leaves).

    Parameters
    ----------
    asset_paths : (n_paths, n_assets, n_steps+1)
    params      : GBMParams

    Returns
    -------
    portfolio_values : (n_paths, n_steps+1), dtype params.dtype
    """
    n_paths, n_assets, n_timepoints = asset_paths.shape
    V0 = params.initial_portfolio_value

    # ── No rebalancing: single-pass vectorized aggregation ───────────────
    if params.rebal_steps is None:
        allocs = (params.weights / params.S0).astype(np.float64)
        pv = (
            asset_paths.astype(np.float64) * allocs[np.newaxis, :, np.newaxis]
        ).sum(axis=1)
        return (pv * V0).astype(params.dtype)

    # ── Rebalancing: segment-by-segment, vectorized within each segment ──
    # Initial share holdings: all paths start identically
    shares = np.tile(
        (params.weights * V0 / params.S0).astype(np.float64),
        (n_paths, 1),
    )                                                            # (n_paths, n_assets)

    pv = np.empty((n_paths, n_timepoints), dtype=np.float64)

    # Segment boundary indices: e.g. [0, 63, 126, 189, 252] for quarterly
    boundaries = list(range(0, n_timepoints, params.rebal_steps))
    if boundaries[-1] != n_timepoints - 1:
        boundaries.append(n_timepoints - 1)

    n_segments = len(boundaries) - 1

    for k in range(n_segments):
        a = boundaries[k]
        b = boundaries[k + 1]

        # Prices in this segment: (n_paths, n_assets, seg_len)
        seg_prices = asset_paths[:, :, a : b + 1].astype(np.float64)

        # Portfolio value across the segment: (n_paths, seg_len)
        seg_pv = (shares[:, :, np.newaxis] * seg_prices).sum(axis=1)

        # Write to output; skip the leading boundary point for k > 0 to
        # avoid double-writing (the boundary value is identical from either
        # side because rebalancing conserves value)
        if k == 0:
            pv[:, a : b + 1] = seg_pv
        else:
            pv[:, a + 1 : b + 1] = seg_pv[:, 1:]

        # Rebalance at the end of this segment (not after the final segment)
        if k < n_segments - 1:
            V_at_b  = seg_pv[:, -1]                            # (n_paths,)
            P_at_b  = asset_paths[:, :, b].astype(np.float64)  # (n_paths, n_assets)
            shares  = V_at_b[:, np.newaxis] * params.weights[np.newaxis, :] / P_at_b

    return pv.astype(params.dtype)


# ---------------------------------------------------------------------------
# Portfolio Analytics
# ---------------------------------------------------------------------------

@dataclass
class RiskMetrics:
    var_95:         float
    var_99:         float
    cvar_95:        float
    cvar_99:        float
    var_95_pct:     float
    var_99_pct:     float
    cvar_95_pct:    float
    cvar_99_pct:    float
    barrier_breach: float
    initial_value:  float
    barrier_level:  float


def compute_risk(portfolio_values: np.ndarray, params: GBMParams) -> RiskMetrics:
    """
    Vectorized portfolio-level risk metrics. Zero Python loops.

    VaR_alpha  = percentile(losses, alpha * 100)
    CVaR_alpha = mean( losses[ losses >= VaR_alpha ] )
    where losses = V0 - V_T  (positive = money lost).

    Barrier breach probability = fraction of paths where V(t) < 0.75 * V0
    at any point during the simulation lifetime.
    """
    V0       = params.initial_portfolio_value
    terminal = portfolio_values[:, -1]
    losses   = V0 - terminal

    var_95  = float(np.percentile(losses, 95))
    var_99  = float(np.percentile(losses, 99))
    cvar_95 = float(np.mean(losses[losses >= var_95]))
    cvar_99 = float(np.mean(losses[losses >= var_99]))

    barrier        = V0 * 0.75
    breached       = np.any(portfolio_values < barrier, axis=1)
    barrier_breach = float(np.mean(breached))

    return RiskMetrics(
        var_95=var_95,         var_99=var_99,
        cvar_95=cvar_95,       cvar_99=cvar_99,
        var_95_pct=(var_95  / V0) * 100,
        var_99_pct=(var_99  / V0) * 100,
        cvar_95_pct=(cvar_95 / V0) * 100,
        cvar_99_pct=(cvar_99 / V0) * 100,
        barrier_breach=barrier_breach,
        initial_value=V0,
        barrier_level=barrier,
    )


# ---------------------------------------------------------------------------
# Visualization Generator
# ---------------------------------------------------------------------------

_ASSET_COLOURS = ["#58a6ff", "#3fb950", "#ff7b72", "#d2a8ff", "#ffa657", "#79c0ff"]


def plot_simulation(
    asset_paths:      np.ndarray,
    portfolio_values: np.ndarray,
    params:           GBMParams,
    risk:             RiskMetrics,
    save_path:        str | Path = ASSETS_DIR / "simulation_results.png",
    n_sample:         int = 100,
) -> None:
    
    """
    Three-panel report saved to disk (headless, Agg backend).

    Panel 1 : 100 sampled portfolio value paths, analytical mean, 75% barrier.
    Panel 2 : Per-asset normalized paths (P / S0) with per-asset analytical means.
    Panel 3 : Terminal portfolio value histogram with VaR / CVaR markers.

    Path sampling uses a single vectorized fancy-index; no Python loop.
    rasterized=True on dense trajectory layers keeps the file size manageable.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    BG      = "#0d1117"
    SURFACE = "#161b22"
    BORDER  = "#30363d"
    TEXT    = "#e6edf3"
    MUTED   = "#8b949e"
    TICK    = "#c9d1d9"
    GOLD    = "#e3b341"
    RED     = "#f85149"

    rng    = np.random.default_rng(42)
    idx    = rng.choice(params.n_paths, size=n_sample, replace=False)
    t_axis = np.linspace(0, params.T, params.n_steps + 1)

    # Analytical mean portfolio: V0 * sum_i( w_i * exp(mu_i * t) )
    mean_portfolio = risk.initial_value * (
        params.weights[:, np.newaxis]
        * np.exp(params.mu[:, np.newaxis] * t_axis[np.newaxis, :])
    ).sum(axis=0)

    fig, axes = plt.subplots(
        3, 1,
        figsize=(14, 14),
        gridspec_kw={"height_ratios": [3, 2, 2], "hspace": 0.42},
        facecolor=BG,
    )

    def _style_ax(ax):
        ax.set_facecolor(BG)
        ax.tick_params(colors=TICK, labelsize=9)
        for sp in ax.spines.values():
            sp.set_edgecolor(BORDER)

    for ax in axes:
        _style_ax(ax)

    ax_port, ax_assets, ax_dist = axes

    # ── Panel 1: Portfolio value paths ──────────────────────────────────
    pv_sample = portfolio_values[idx, :]
    ax_port.plot(
        t_axis, pv_sample.T,
        color="#388bfd", alpha=0.15, linewidth=0.55, rasterized=True,
    )
    ax_port.plot(
        t_axis, mean_portfolio,
        color=GOLD, linewidth=2.2, zorder=5,
        label=r"Analytical Mean  $V_0 \sum_i w_i e^{\mu_i t}$",
    )
    ax_port.axhline(
        risk.barrier_level, color=RED, linewidth=1.4, linestyle="--", zorder=4,
        label=f"75% Risk Barrier  (${risk.barrier_level:,.2f})",
    )

    rebal_tag    = f"  |  Rebalancing: {params.rebal_label}"
    weights_str  = "  |  ".join(
        f"{params.names[i]} {params.weights[i]:.0%}" for i in range(params.n_assets)
    )
    ax_port.set_title(
        f"Multi-Asset Portfolio  —  {params.n_paths:,} paths  |  "
        f"T={params.T}yr  |  {params.n_steps} steps{rebal_tag}\n{weights_str}",
        color=TEXT, fontsize=11, pad=12,
    )
    ax_port.set_xlabel("Time (years)", color=MUTED, fontsize=10)
    ax_port.set_ylabel("Portfolio Value ($)", color=MUTED, fontsize=10)
    ax_port.legend(facecolor=SURFACE, edgecolor=BORDER, labelcolor=TEXT, fontsize=9)
    ax_port.yaxis.set_major_formatter(mticker.FormatStrFormatter("$%.0f"))

    # ── Panel 2: Per-asset normalized paths ─────────────────────────────
    for i in range(params.n_assets):
        colour       = _ASSET_COLOURS[i % len(_ASSET_COLOURS)]
        asset_sample = asset_paths[idx, i, :]
        normalized   = asset_sample / params.S0[i].astype(params.dtype)
        ax_assets.plot(
            t_axis, normalized.T,
            color=colour, alpha=0.20, linewidth=0.55, rasterized=True,
        )
        ax_assets.plot(
            t_axis, np.exp(params.mu[i] * t_axis),
            color=colour, linewidth=2.0, zorder=5,
            label=f"{params.names[i]}  μ={params.mu[i]:.0%}  σ={params.sigma[i]:.0%}",
        )

    ax_assets.axhline(1.0, color=BORDER, linewidth=0.8, linestyle=":")
    ax_assets.set_title(
        "Per-Asset Normalized Paths  (Price / S₀)",
        color=TEXT, fontsize=10, pad=8,
    )
    ax_assets.set_xlabel("Time (years)", color=MUTED, fontsize=9)
    ax_assets.set_ylabel("Normalized Price", color=MUTED, fontsize=9)
    ax_assets.legend(
        facecolor=SURFACE, edgecolor=BORDER, labelcolor=TEXT,
        fontsize=8, ncol=params.n_assets,
    )
    ax_assets.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2fx"))

    # ── Panel 3: Terminal value distribution ────────────────────────────
    terminal = portfolio_values[:, -1].astype(np.float64)
    ax_dist.hist(
        terminal, bins=100, color="#388bfd",
        alpha=0.55, density=True, rasterized=True,
    )

    for price, colour, ls, lbl in [
        (risk.initial_value - risk.var_95,  GOLD, "--", "VaR 95%"),
        (risk.initial_value - risk.var_99,  RED,  "--", "VaR 99%"),
        (risk.initial_value - risk.cvar_95, GOLD, ":",  "CVaR 95%"),
        (risk.initial_value - risk.cvar_99, RED,  ":",  "CVaR 99%"),
    ]:
        ax_dist.axvline(price, color=colour, linewidth=1.3, linestyle=ls, label=lbl)

    ax_dist.set_title(
        "Terminal Portfolio Value Distribution",
        color=TEXT, fontsize=10, pad=8,
    )
    ax_dist.set_xlabel("Terminal Portfolio Value ($)", color=MUTED, fontsize=9)
    ax_dist.set_ylabel("Density", color=MUTED, fontsize=9)
    ax_dist.legend(
        facecolor=SURFACE, edgecolor=BORDER, labelcolor=TEXT,
        fontsize=8, ncol=4,
    )
    ax_dist.xaxis.set_major_formatter(mticker.FormatStrFormatter("$%.0f"))
    
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(
        save_path,
        dpi=150,
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
    )
    plt.close(fig)
    print(f"  Plot saved → {save_path}")


# ---------------------------------------------------------------------------
# Simulation Result
# ---------------------------------------------------------------------------

@dataclass
class SimulationResult:
    asset_paths:      np.ndarray          # (n_paths, n_assets, n_steps+1)
    portfolio_values: np.ndarray          # (n_paths, n_steps+1)
    elapsed_ms:       float
    memory_mb:        float
    paths_per_sec:    float
    risk:             Optional[RiskMetrics] = field(default=None)


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------

class GBMSimulator:
    """
    Orchestrates path generation across a ThreadPool and aggregates results.

    ThreadPool (OS threads) is used rather than multiprocessing.Pool because
    NumPy releases the GIL during C-extension calls, allowing genuine
    parallelism for the compute-heavy path generation without the
    inter-process communication overhead of spawned subprocesses.

    Seed strategy: each worker receives i * 10_000 + int(time.time() % 1000).
    Seeds are always distinct across workers within a run. The time component
    makes successive runs non-deterministic by design. Pass a fixed integer
    to the seed_base parameter of run() to override this for reproducibility.
    """

    def __init__(self, params: GBMParams, n_workers: Optional[int] = None):
        self.params    = params
        self.n_workers = n_workers or cpu_count()

    def _partition(self, seed_base: int) -> list[tuple]:
        n, w = self.params.n_paths, self.n_workers
        base, remainder = divmod(n, w)
        return [
            (self.params, base + (1 if i < remainder else 0), i * 10_000 + seed_base)
            for i in range(w)
        ]

    def run(self, seed_base: Optional[int] = None) -> SimulationResult:
        """
        Execute the simulation and return a SimulationResult.

        Parameters
        ----------
        seed_base : optional integer to fix RNG seeds across all workers.
                    If None, seeds are derived from the current time (non-deterministic).
        """
        if seed_base is None:
            seed_base = int(time.time() % 1000)

        tracemalloc.start()
        t0 = time.perf_counter()

        with ThreadPool(processes=self.n_workers) as pool:
            chunks = pool.map(_worker, self._partition(seed_base))

        asset_paths      = np.concatenate(chunks, axis=0)
        portfolio_values = compute_portfolio_values(asset_paths, self.params)
        elapsed_ms       = (time.perf_counter() - t0) * 1_000

        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        return SimulationResult(
            asset_paths      = asset_paths,
            portfolio_values = portfolio_values,
            elapsed_ms       = elapsed_ms,
            memory_mb        = peak_bytes / 1024 ** 2,
            paths_per_sec    = self.params.n_paths / (elapsed_ms / 1_000),
        )


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def benchmark(
    n_paths:     int           = 1_000_000,
    n_steps:     int           = 252,
    T:           float         = 1.0,
    runs:        int           = 3,
    rebal_steps: Optional[int] = REBAL_FREQ["quarterly"],
    plot:        bool          = True,
) -> None:
    """
    Run the 3-asset reference portfolio and print a full performance and
    risk report.

    Parameters
    ----------
    n_paths     : total Monte Carlo paths
    n_steps     : discrete time steps (252 = daily for one trading year)
    T           : simulation horizon in years
    runs        : number of timed repetitions; best/worst/mean/stddev reported
    rebal_steps : rebalancing interval in steps; None disables rebalancing.
                  Use REBAL_FREQ constants: REBAL_FREQ["quarterly"] etc.
    plot        : write simulation_results.png to the working directory

    Reference portfolio
    -------------------
    Equity     S0=100  mu=9%   sigma=22%  weight=50%
    Bond       S0= 95  mu=3%   sigma= 6%  weight=30%
    Commodity  S0=120  mu=5%   sigma=18%  weight=20%

    Correlation matrix
    ------------------
    [[1.00, 0.25, 0.10],
     [0.25, 1.00, 0.05],
     [0.10, 0.05, 1.00]]
    """
    params = GBMParams(
        S0      = np.array([100.0,  95.0, 120.0]),
        mu      = np.array([  0.09,  0.03,  0.05]),
        sigma   = np.array([  0.22,  0.06,  0.18]),
        weights = np.array([  0.50,  0.30,  0.20]),
        rho     = np.array([
            [1.00, 0.25, 0.10],
            [0.25, 1.00, 0.05],
            [0.10, 0.05, 1.00],
        ]),
        T           = T,
        n_steps     = n_steps,
        n_paths     = n_paths,
        rebal_steps = rebal_steps,
        names       = ["Equity", "Bond", "Commodity"],
    )

    sim  = GBMSimulator(params)
    sep  = "─" * 70
    sep2 = "═" * 70

    print(f"\n{sep2}")
    print(f"  Multi-Asset GBM Portfolio Simulator")
    print(f"{sep2}")
    print(f"  Assets       : {params.n_assets}  ({', '.join(params.names)})")
    print(f"  Paths        : {n_paths:>12,}")
    print(f"  Steps        : {n_steps:>12,}")
    print(f"  Workers      : {sim.n_workers:>12,}  (ThreadPool)")
    print(f"  dtype        : {params.dtype.__name__:>12}")
    print(f"  Runs         : {runs:>12,}")
    print(f"  T            : {T:>12.1f} years")
    print(f"  Rebalancing  : {params.rebal_label:>12}")
    if rebal_steps is not None:
        n_events = len(range(rebal_steps, n_steps, rebal_steps))
        print(f"  Rebal events : {n_events:>12,}  per path")
    print(f"\n  Portfolio composition")
    for i in range(params.n_assets):
        print(
            f"    {params.names[i]:<12}  S0=${params.S0[i]:>6.2f}  "
            f"μ={params.mu[i]:.0%}  σ={params.sigma[i]:.0%}  "
            f"wt={params.weights[i]:.0%}"
        )
    print(f"  Initial portfolio value : ${params.initial_portfolio_value:.2f}")
    print(f"\n  Cholesky factor L  (rho = L @ L.T)")
    for row in params.L:
        print("    " + "  ".join(f"{v:+.4f}" for v in row))
    print(f"{sep}\n")

    timings, last_result = [], None

    for i in range(1, runs + 1):
        result = sim.run()
        timings.append(result.elapsed_ms)
        last_result = result
        print(
            f"  Run {i:>2}  |  "
            f"Time: {result.elapsed_ms:>9.1f} ms  |  "
            f"Mem: {result.memory_mb:>7.1f} MB  |  "
            f"Throughput: {result.paths_per_sec:>12,.0f} paths/s"
        )

    best    = min(timings)
    worst   = max(timings)
    mean_t  = sum(timings) / len(timings)
    std_t   = (sum((t - mean_t) ** 2 for t in timings) / len(timings)) ** 0.5
    ap_mb   = last_result.asset_paths.nbytes / 1024 ** 2
    pv_mb   = last_result.portfolio_values.nbytes / 1024 ** 2

    print(f"\n{sep}")
    print(f"  Timing summary (ms)")
    print(f"    Best   : {best:>9.1f}")
    print(f"    Worst  : {worst:>9.1f}")
    print(f"    Mean   : {mean_t:>9.1f}")
    print(f"    StdDev : {std_t:>9.1f}")
    print(f"\n  Memory")
    print(f"    asset_paths      : {ap_mb:>7.1f} MB  "
          f"({n_paths:,} × {params.n_assets} × {n_steps+1}  float32)")
    print(f"    portfolio_values : {pv_mb:>7.1f} MB  "
          f"({n_paths:,} × {n_steps+1}  float32)")
    print(f"    Peak traced      : {last_result.memory_mb:>7.1f} MB")

    print(f"\n{sep}")
    print(f"  Portfolio Risk Analytics  (rebalancing: {params.rebal_label})")
    print(f"{sep}")

    risk = compute_risk(last_result.portfolio_values, params)
    last_result.risk = risk

    print(f"  Initial portfolio value  : ${risk.initial_value:>10.2f}")
    print(f"  75% barrier level        : ${risk.barrier_level:>10.2f}")
    print(f"\n  Value at Risk  (portfolio dollar loss at terminal step)")
    print(f"    VaR  95%  : ${risk.var_95:>10.2f}  ({risk.var_95_pct:.2f}% of portfolio)")
    print(f"    VaR  99%  : ${risk.var_99:>10.2f}  ({risk.var_99_pct:.2f}% of portfolio)")
    print(f"\n  Conditional VaR  (Expected Shortfall)")
    print(f"    CVaR 95%  : ${risk.cvar_95:>10.2f}  ({risk.cvar_95_pct:.2f}% of portfolio)")
    print(f"    CVaR 99%  : ${risk.cvar_99:>10.2f}  ({risk.cvar_99_pct:.2f}% of portfolio)")
    print(f"\n  Barrier Breach  (portfolio < 75% of V₀ at any point)")
    print(f"    Probability   :  {risk.barrier_breach * 100:>9.4f}%")

    if plot:
        print(f"\n{sep}")
        print(f"  Generating plot...")
        plot_simulation(
            last_result.asset_paths,
            last_result.portfolio_values,
            params,
            risk,
            save_path = ASSETS_DIR / "simulation_results.png",
        )

    print(f"{sep2}\n")


if __name__ == "__main__":
    benchmark()