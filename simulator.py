"""
Multi-Asset GBM Portfolio Simulator
===================================
A vectorized simulation engine designed to project multi-asset pathways under
correlated market conditions using geometric Brownian motion.
"""

import time
import tracemalloc
from dataclasses import dataclass, field
from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Rebalancing Frequency Constants
# ---------------------------------------------------------------------------
# All values assume n_steps=252 (one trading year).  Pass an integer directly
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
        
    if not np.all(np.diag(p.rho) == 1.0):
        raise ValueError("Correlation matrix diagonal must equal 1.0")
        
    w_sum = np.sum(p.weights)
    if not np.isclose(w_sum, 1.0):
        raise ValueError(f"Portfolio allocation weights must sum to 1.0, got {w_sum:.6f}")


# ---------------------------------------------------------------------------
# Configuration Data Structures
# ---------------------------------------------------------------------------

@dataclass
class GBMParams:
    """Holds parameters, target metrics, and structural configs for the simulation."""
    S0:      np.ndarray
    mu:      np.ndarray
    sigma:   np.ndarray
    weights: np.ndarray
    rho:     np.ndarray
    T:       float
    n_steps: int
    n_paths: int
    names:   list        = field(default_factory=list)
    dtype:   type        = np.float32

    # Pre-computed Cholesky factor matrix for generating correlations
    L:       np.ndarray  = field(init=False, repr=False)

    def __post_init__(self):
        self.S0      = _to_f64_array(self.S0,      "S0")
        self.mu      = _to_f64_array(self.mu,       "mu")
        self.sigma   = _to_f64_array(self.sigma,    "sigma")
        self.weights = _to_f64_array(self.weights,  "weights")
        self.rho     = np.asarray(self.rho, dtype=np.float64)
        
        if self.rho.ndim != 2:
            raise ValueError("rho must be a 2D matrix")
            
        _validate_params(self)
        
        # Factor the correlation matrix up front
        self.L = np.linalg.cholesky(self.rho)
        
        if not self.names:
            self.names = [f"Asset {i+1}" for i in range(len(self.S0))]

    @property
    def n_assets(self) -> int:
        return len(self.S0)

    @property
    def initial_portfolio_value(self) -> float:
        return float(self._total_investment)

    @property
    def _total_investment(self) -> float:
        return float(np.sum(self.S0))


# ---------------------------------------------------------------------------
# Execution Worker
# ---------------------------------------------------------------------------

def _worker(args: tuple) -> np.ndarray:
    """Generates a slice of correlated multi-asset price pathways."""
    params, slice_paths, seed_offset = args

    n_assets = params.n_assets
    n_steps  = params.n_steps
    dt       = params.T / n_steps
    rng      = np.random.default_rng(seed_offset)

    # Set up structural drift and diffusion vectors
    drift     = (params.mu - 0.5 * params.sigma ** 2) * dt
    diffusion = params.sigma * np.sqrt(dt)

    # Generate random baseline shocks
    Z = rng.standard_normal((slice_paths, n_steps, n_assets))

    # Apply correlations via matrix multiplication
    Z_corr = Z @ params.L.T

    # Calculate price increments over each time step
    increments = np.exp(drift + diffusion * Z_corr, dtype=params.dtype)
    increments = increments.transpose(0, 2, 1)

    # Map the starting prices to step zero
    S0_col = np.broadcast_to(
        params.S0.astype(params.dtype)[np.newaxis, :, np.newaxis],
        (slice_paths, n_assets, 1),
    ).copy()

    # Combine structures and calculate cumulative price paths
    paths = np.concatenate([S0_col, increments], axis=2)
    np.cumprod(paths, axis=2, out=paths)
    return paths


# ---------------------------------------------------------------------------
# Portfolio Value Aggregator
# ---------------------------------------------------------------------------

def compute_portfolio_values(asset_paths: np.ndarray, params: GBMParams) -> np.ndarray:
    """Aggregates per-asset paths into total portfolio value curves over time."""
    # Convert nominal weights to share units based on starting values
    normalized_allocations = params.weights / params.S0
    
    # Broadcast across paths and sum across assets
    portfolio_shares = (asset_paths * normalized_allocations[np.newaxis, :, np.newaxis]).sum(axis=1)
    
    # Scale by initial allocation to represent true absolute dollar values
    return portfolio_shares * params._total_investment


# ---------------------------------------------------------------------------
# Portfolio Analytics
# ---------------------------------------------------------------------------

@dataclass
class RiskMetrics:
    var_95:          float
    var_99:          float
    cvar_95:         float
    cvar_99:         float
    var_95_pct:      float
    var_99_pct:      float
    cvar_95_pct:     float
    cvar_99_pct:     float
    barrier_breach:  float
    initial_value:   float
    barrier_level:   float


def compute_risk(portfolio_values: np.ndarray, params: GBMParams) -> RiskMetrics:
    """Calculates distributional risk parameters from terminal portfolio states."""
    V0 = params.initial_portfolio_value

    terminal = portfolio_values[:, -1]
    losses   = V0 - terminal  # Express losses as positive values

    # Value at Risk thresholds
    var_95  = float(np.percentile(losses, 95))
    var_99  = float(np.percentile(losses, 99))
    
    # Expected Shortfall calculations
    cvar_95 = float(np.mean(losses[losses >= var_95]))
    cvar_99 = float(np.mean(losses[losses >= var_99]))

    # Check for risk floor exceptions across the entire timeline
    barrier = V0 * 0.75
    breached       = np.any(portfolio_values < barrier, axis=1)
    barrier_breach = float(np.mean(breached))

    return RiskMetrics(
        var_95=var_95,
        var_99=var_99,
        cvar_95=cvar_95,
        cvar_99=cvar_99,
        var_95_pct=(var_95 / V0) * 100,
        var_99_pct=(var_99 / V0) * 100,
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
    save_path:        str = "simulation_results.png",
    n_sample:         int = 100,
) -> None:
    """Generates a standard 3-panel visual report of the simulation outcomes."""
    import matplotlib
    matplotlib.use("Agg")  # Run headless without window context requirements
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    # Interface palette selections
    BG      = "#0d1117"
    SURFACE = "#161b22"
    BORDER  = "#30363d"
    TEXT    = "#e6edf3"
    MUTED   = "#8b949e"
    TICK    = "#c9d1d9"
    GOLD    = "#e3b341"
    RED     = "#f85149"

    # Pick a random subset of lines to plot for performance clarity
    rng        = np.random.default_rng()
    idx        = rng.choice(params.n_paths, size=n_sample, replace=False)
    t_axis     = np.linspace(0, params.T, params.n_steps + 1)

    # Compute expected tracking values
    mean_components = params.weights[:, np.newaxis] * np.exp(params.mu[:, np.newaxis] * t_axis[np.newaxis, :])
    mean_portfolio = risk.initial_value * mean_components.sum(axis=0)

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

    # -- PANEL 1: Total Portfolio Dollar Pathways
    pv_sample = portfolio_values[idx, :]
    ax_port.plot(t_axis, pv_sample.T, color="#388bfd", alpha=0.15, linewidth=0.55, rasterized=True)
    ax_port.plot(t_axis, mean_portfolio, color=GOLD, linewidth=2.2, zorder=5, label=r"Analytical Mean  $V_0 \sum_i w_i e^{\mu_i t}$")
    ax_port.axhline(risk.barrier_level, color=RED, linewidth=1.4, linestyle="--", zorder=4, label=f"75% Risk Barrier  (${risk.barrier_level:,.2f})")

    weights_str = "  |  ".join(f"{params.names[i]} {params.weights[i]:.0%}" for i in range(params.n_assets))
    ax_port.set_title(f"Multi-Asset Portfolio  —  {params.n_paths:,} paths  |  T={params.T}yr  |  {params.n_steps} steps\n{weights_str}", color=TEXT, fontsize=11, pad=12)
    ax_port.set_xlabel("Time (years)", color=MUTED, fontsize=10)
    ax_port.set_ylabel("Portfolio Value ($)", color=MUTED, fontsize=10)
    ax_port.legend(facecolor=SURFACE, edgecolor=BORDER, labelcolor=TEXT, fontsize=9)
    ax_port.yaxis.set_major_formatter(mticker.FormatStrFormatter("$%.0f"))

    # -- PANEL 2: Normalized Component Trends
    for i in range(params.n_assets):
        colour        = _ASSET_COLOURS[i % len(_ASSET_COLOURS)]
        asset_sample  = asset_paths[idx, i, :]
        normalised    = asset_sample / params.S0[i].astype(params.dtype)

        ax_assets.plot(t_axis, normalised.T, color=colour, alpha=0.20, linewidth=0.55, rasterized=True)
        ax_assets.plot(t_axis, np.exp(params.mu[i] * t_axis), color=colour, linewidth=2.0, zorder=5, label=f"{params.names[i]}  μ={params.mu[i]:.0%}  σ={params.sigma[i]:.0%}")

    ax_assets.axhline(1.0, color=BORDER, linewidth=0.8, linestyle=":")
    ax_assets.set_title("Per-Asset Normalised Paths  (Price / S₀)", color=TEXT, fontsize=10, pad=8)
    ax_assets.set_xlabel("Time (years)", color=MUTED, fontsize=9)
    ax_assets.set_ylabel("Normalised Price", color=MUTED, fontsize=9)
    ax_assets.legend(facecolor=SURFACE, edgecolor=BORDER, labelcolor=TEXT, fontsize=8, ncol=params.n_assets)
    ax_assets.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2fx"))

    # -- PANEL 3: Terminal Value Densities
    terminal = portfolio_values[:, -1].astype(np.float64)
    ax_dist.hist(terminal, bins=100, color="#388bfd", alpha=0.55, density=True, rasterized=True)

    var95_price  = risk.initial_value - risk.var_95
    var99_price  = risk.initial_value - risk.var_99
    cvar95_price = risk.initial_value - risk.cvar_95
    cvar99_price = risk.initial_value - risk.cvar_99

    for price, colour, ls, lbl in [
        (var95_price,  GOLD, "--", "VaR 95%"),
        (var99_price,  RED,  "--", "VaR 99%"),
        (cvar95_price, GOLD, ":",  "CVaR 95%"),
        (cvar99_price, RED,  ":",  "CVaR 99%"),
    ]:
        ax_dist.axvline(price, color=colour, linewidth=1.3, linestyle=ls, label=lbl)

    ax_dist.set_title("Terminal Portfolio Value Distribution", color=TEXT, fontsize=10, pad=8)
    ax_dist.set_xlabel("Terminal Portfolio Value ($)", color=MUTED, fontsize=9)
    ax_dist.set_ylabel("Density", color=MUTED, fontsize=9)
    ax_dist.legend(facecolor=SURFACE, edgecolor=BORDER, labelcolor=TEXT, fontsize=8, ncol=4)
    ax_dist.xaxis.set_major_formatter(mticker.FormatStrFormatter("$%.0f"))

    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Plot saved → {save_path}")


# ---------------------------------------------------------------------------
# Execution Handling Structure
# ---------------------------------------------------------------------------

@dataclass
class SimulationResult:
    asset_paths:      np.ndarray
    portfolio_values: np.ndarray
    elapsed_ms:       float
    memory_mb:        float
    paths_per_sec:    float
    risk:             Optional[RiskMetrics] = field(default=None)


class GBMSimulator:
    def __init__(self, params: GBMParams, n_workers: Optional[int] = None):
        self.params    = params
        self.n_workers = n_workers or cpu_count()

    def _partition(self) -> list[tuple]:
        """Calculates workload splits and anchors stable seeds for execution contexts."""
        n, w = self.params.n_paths, self.n_workers
        base, remainder = divmod(n, w)
        return [
            (self.params, base + (1 if i < remainder else 0), i * 10_000 + int(time.time() % 1000))
            for i in range(w)
        ]

    def run(self) -> SimulationResult:
        """Coordinates workload distribution across the pool and gathers results."""
        tracemalloc.start()
        t0 = time.perf_counter()

        with ThreadPool(processes=self.n_workers) as pool:
            chunks = pool.map(_worker, self._partition())

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
# Interface Execution Block
# ---------------------------------------------------------------------------

def benchmark(
    n_paths: int  = 1_000_000,
    n_steps: int  = 252,
    T:       float = 1.0,
    runs:    int  = 3,
    plot:    bool = True,
) -> None:
    # Build core metrics map
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
        T       = T,
        n_steps = n_steps,
        n_paths = n_paths,
        names   = ["Equity", "Bond", "Commodity"],
    )

    sim  = GBMSimulator(params)
    sep  = "─" * 70
    sep2 = "═" * 70

    print(f"\n{sep2}")
    print(f"  Multi-Asset GBM Portfolio Simulator")
    print(f"{sep2}")
    print(f"  Assets     : {params.n_assets}  ({', '.join(params.names)})")
    print(f"  Paths      : {n_paths:>12,}")
    print(f"  Steps      : {n_steps:>12,}")
    print(f"  Workers    : {sim.n_workers:>12,}  (Threaded Pool)")
    print(f"  dtype      : {params.dtype.__name__:>12}")
    print(f"  Runs       : {runs:>12,}")
    print(f"  T          : {T:>12.1f} years")
    print(f"\n  Portfolio composition")
    
    for i in range(params.n_assets):
        print(
            f"    {params.names[i]:<12}  S0=${params.S0[i]:>6.2f}  "
            f"μ={params.mu[i]:.0%}  σ={params.sigma[i]:.0%}  "
            f"wt={params.weights[i]:.0%}"
        )
    print(f"  Initial portfolio value base scale : ${params.initial_portfolio_value:.2f}")
    print(f"{sep}\n")

    timings     = []
    last_result = None

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

    # Risk metrics summary report
    print(f"\n{sep}")
    print(f"  Portfolio Risk Analytics")
    print(f"{sep}")

    risk = compute_risk(last_result.portfolio_values, params)
    last_result.risk = risk

    print(f"  Initial portfolio value  : ${risk.initial_value:>10.2f}")
    print(f"  75% barrier level        : ${risk.barrier_level:>10.2f}")
    print(f"\n  Value at Risk  (portfolio dollar loss)")
    print(f"    VaR  95%  : ${risk.var_95:>10.2f}  ({last_result.risk.var_95_pct:.2f}% of portfolio)")
    print(f"    VaR  99%  : ${risk.var_99:>10.2f}  ({last_result.risk.var_99_pct:.2f}% of portfolio)")
    print(f"\n  Conditional VaR  (Expected Shortfall)")
    print(f"    CVaR 95%  : ${risk.cvar_95:>10.2f}  ({last_result.risk.cvar_95_pct:.2f}% of portfolio)")
    print(f"    CVaR 99%  : ${risk.cvar_99:>10.2f}  ({last_result.risk.cvar_99_pct:.2f}% of portfolio)")
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
        )

    print(f"{sep2}\n")


if __name__ == "__main__":
    benchmark()