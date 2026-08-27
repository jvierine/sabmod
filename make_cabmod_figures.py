from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import xarray as xr

ROOT = Path(__file__).resolve().parent
try:
    import metablate
except:
    # try local install
    ABLAte_SRC = ROOT.parent / "ablate" / "src"
    if str(ABLAte_SRC) not in sys.path:
        sys.path.insert(0, str(ABLAte_SRC))
    import metablate


FIG_DIR = ROOT / "figures"
DEFAULT_TIME = np.datetime64("2018-06-28T12:45:33", "ns")
DEFAULT_LAT = 69.30
DEFAULT_LON = 16.04
DEFAULT_REFERENCE_ALTITUDE_M = 100e3
DEFAULT_START_ALTITUDE_M = 130e3
DEFAULT_AZIMUTH_DEG = 0.0
DEFAULT_MASS_KG = 1e-8
FIGURE1_VELOCITIES_KM_S = np.array([32.0, 53.0, 72.0], dtype=np.float64)
FIGURE1_ENTRY_ELEVATION_ANGLES_DEG = np.array([70.0, 45.0, 20.0], dtype=np.float64)
DEFAULT_ENTRY_ELEVATION_ANGLE_DEG = float(FIGURE1_ENTRY_ELEVATION_ANGLES_DEG[1])
FIGURE1_SHOW_PEAK_ABLATION_LINES = False
FIGURE2_BASELINE_VELOCITIES_KM_S = np.array([32.0, 52.0, 72.0], dtype=np.float64)
FIGURE2_DENSITY_SCALES = (1.2, 0.8)
ATMOSPHERE_ALTITUDE_GRID_M = np.linspace(50e3, 150e3, 1001)
FIGURE1_ENTRY_ELEVATION_COLORS = tuple(plt.rcParams["axes.prop_cycle"].by_key()["color"])
VELOCITY_SHIFT_TABLE_PATH = FIG_DIR / "velocity_shift_table.tex"
DEFAULT_MATERIAL="cometary"

class CachedScaledAtmosphere(metablate.atmosphere.Atmosphere):
    def __init__(self, base_atmosphere, density_scale):
        super().__init__(supported_species=list(base_atmosphere.species.values()))
        self.base_atmosphere = base_atmosphere
        self.density_scale = float(density_scale)
        self.mean_mass = base_atmosphere.mean_mass
        self._profile = base_atmosphere.density(
            time=DEFAULT_TIME,
            lat=DEFAULT_LAT,
            lon=DEFAULT_LON,
            alt=ATMOSPHERE_ALTITUDE_GRID_M,
            mass_densities=False,
            version=2.1,
        )
        self._altitudes = self._profile.coords["alt"].values.astype(float)
        self._values = {
            key: self._profile[key].values.reshape(-1).astype(float)
            for key in self._profile.data_vars
        }

    def density(self, *args, **kwargs):
        alt = np.atleast_1d(np.asarray(kwargs.get("alt"), dtype=float))
        data_vars = {}
        for key, values in self._values.items():
            interpolated = np.interp(
                alt,
                self._altitudes,
                values,
                left=values[0],
                right=values[-1],
            )
            if key != "Temperature":
                interpolated = interpolated * self.density_scale
            data_vars[key] = (["time", "lon", "lat", "alt"], interpolated.reshape(1, 1, 1, -1))

        return xr.Dataset(
            data_vars,
            coords={
                "time": np.atleast_1d(kwargs.get("time", DEFAULT_TIME)),
                "lon": np.atleast_1d(kwargs.get("lon", DEFAULT_LON)),
                "lat": np.atleast_1d(kwargs.get("lat", DEFAULT_LAT)),
                "alt": alt,
            },
            attrs={"density_scale": self.density_scale},
        )


def build_kero_model(density_scale=1.0):
    base_atmosphere = metablate.atmosphere.AtmPymsis()
    atmosphere = CachedScaledAtmosphere(base_atmosphere, density_scale)
    return metablate.KeroSzasz2008(
        atmosphere=atmosphere,
        config={
            "options": {
                "temperature0": 290,
                "shape_factor": 1.21,
                "emissivity": 0.9,
                "sputtering": False,
                "Gamma": 1.0,
                "Lambda": 1.0,
                "integral_resolution": 40,
            },
            "atmosphere": {
                "version": 2.1,
            },
            "integrate": {
                "minimum_mass_kg": DEFAULT_MASS_KG * 1e-5,
                "max_step_size_sec": 5e-2,
                "max_time_sec": 5.0,
                "method": "RK45",
            },
        },
    )


def simulate_case(
    velocity_km_s,
    density_scale=1.0,
    entry_elevation_angle_deg=DEFAULT_ENTRY_ELEVATION_ANGLE_DEG,
):
    model = build_kero_model(density_scale=density_scale)
    material_data = metablate.material.get(DEFAULT_MATERIAL)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        result = model.run(
            velocity0=velocity_km_s * 1e3,
            mass0=DEFAULT_MASS_KG,
            altitude0=DEFAULT_START_ALTITUDE_M,
            # metablate names this argument zenith_ang, but KeroSzasz2008
            # passes it to azel_to_cart as an elevation angle above horizon.
            zenith_ang=entry_elevation_angle_deg,
            azimuth_ang=DEFAULT_AZIMUTH_DEG,
            material_data=material_data,
            time=DEFAULT_TIME,
            lat=DEFAULT_LAT,
            lon=DEFAULT_LON,
            alt=DEFAULT_REFERENCE_ALTITUDE_M,
        )

    altitude_km = result.altitude.values * 1e-3
    velocity_km_s = result.velocity.values * 1e-3
    mass_loss_rate = np.abs(np.gradient(result.mass.values, result.t, edge_order=1))
    mass_loss_rate[~np.isfinite(mass_loss_rate)] = np.nan

    peak_ind = np.nanargmax(mass_loss_rate)
    sort_inds = np.argsort(altitude_km)

    return {
        "altitude_km": altitude_km[sort_inds],
        "velocity_km_s": velocity_km_s[sort_inds],
        "mass_loss_rate_kg_s": mass_loss_rate[sort_inds],
        "temperature_k": result.temperature.values[sort_inds],
        "mass_kg": result.mass.values[sort_inds],
        "peak_altitude_km": altitude_km[peak_ind],
    }


def draw_peak_ablation_line(ax, result, color, linestyle, show_lines):
    if not show_lines:
        return
    ax.axhline(
        result["peak_altitude_km"],
        color=color,
        linestyle=linestyle,
        lw=1.0,
        alpha=0.55,
    )


def interp1d_linear_extrapolate(x, y, xq):
    """Piecewise-linear interpolation with linear extrapolation at both ends."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    unique_x, unique_indices = np.unique(x, return_index=True)
    x = unique_x
    y = y[unique_indices]

    if x.size < 2:
        return np.full_like(np.asarray(xq, dtype=np.float64), np.nan, dtype=np.float64)

    xq = np.asarray(xq, dtype=np.float64)
    yq = np.interp(xq, x, y)

    low = xq < x[0]
    high = xq > x[-1]
    if np.any(low):
        slope = (y[1] - y[0]) / (x[1] - x[0])
        yq[low] = y[0] + slope * (xq[low] - x[0])
    if np.any(high):
        slope = (y[-1] - y[-2]) / (x[-1] - x[-2])
        yq[high] = y[-1] + slope * (xq[high] - x[-1])
    return yq


def make_figure1(show_peak_ablation_lines=FIGURE1_SHOW_PEAK_ABLATION_LINES):
    results = []
    for entry_elevation_angle_deg in FIGURE1_ENTRY_ELEVATION_ANGLES_DEG:
        for velocity_km_s in FIGURE1_VELOCITIES_KM_S:
            results.append(
                {
                    "entry_elevation_angle_deg": entry_elevation_angle_deg,
                    "velocity_km_s0": velocity_km_s,
                    "data": simulate_case(
                        velocity_km_s,
                        density_scale=1.0,
                        entry_elevation_angle_deg=entry_elevation_angle_deg,
                    ),
                }
            )

    entry_elevation_colors = {
        float(angle): FIGURE1_ENTRY_ELEVATION_COLORS[ind % len(FIGURE1_ENTRY_ELEVATION_COLORS)]
        for ind, angle in enumerate(FIGURE1_ENTRY_ELEVATION_ANGLES_DEG)
    }
    velocity_linestyles = {
        32.0: "-",
        53.0: "--",
        72.0: ":",
    }

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(7.2, 5.4),
        sharey=True,
        constrained_layout=True,
    )
    axes = axes.ravel()

    for item in results:
        result = item["data"]
        entry_elevation_angle_deg = item["entry_elevation_angle_deg"]
        velocity_km_s = item["velocity_km_s0"]
        color = entry_elevation_colors[float(entry_elevation_angle_deg)]
        linestyle = velocity_linestyles[float(velocity_km_s)]
        axes[0].plot(
            result["velocity_km_s"],
            result["altitude_km"],
            color=color,
            linestyle=linestyle,
            lw=2.0,
        )
        draw_peak_ablation_line(axes[0], result, color, linestyle, show_peak_ablation_lines)
    axes[0].set_xlabel("Velocity [km s$^{-1}$]")
    axes[0].set_ylabel("Altitude [km]")
    axes[0].grid(True, alpha=0.25)
    axes[0].set_title("(a) Velocity")
    axes[0].set_ylim(70, 130)

    for item in results:
        result = item["data"]
        entry_elevation_angle_deg = item["entry_elevation_angle_deg"]
        velocity_km_s = item["velocity_km_s0"]
        color = entry_elevation_colors[float(entry_elevation_angle_deg)]
        linestyle = velocity_linestyles[float(velocity_km_s)]
        axes[1].plot(
            result["mass_loss_rate_kg_s"],
            result["altitude_km"],
            color=color,
            linestyle=linestyle,
            lw=2.0,
        )
        draw_peak_ablation_line(axes[1], result, color, linestyle, show_peak_ablation_lines)
    axes[1].set_xscale("log")
    axes[1].set_xlabel(r"Absolute mass-loss rate, $|dm/dt|$ [kg s$^{-1}$]")
    axes[1].grid(True, which="both", alpha=0.25)
    axes[1].set_title("(b) Mass loss rate")
    # Set x-limits to span from one order of magnitude below the global peak
    # mass-loss to the peak itself (on the x-axis, which is log-scaled).
    all_peaks = [np.nanmax(item["data"]["mass_loss_rate_kg_s"]) for item in results]
    peak_mass_loss = np.nanmax(all_peaks) if len(all_peaks) > 0 else np.nan
    if not np.isfinite(peak_mass_loss) or peak_mass_loss <= 0:
        # fallback to previous sensible defaults
        axes[1].set_xlim(1e-7, 1e-6)
    else:
        axes[1].set_xlim(peak_mass_loss / 100.0, peak_mass_loss*10)
    axes[1].set_ylim(70, 130)

    for item in results:
        result = item["data"]
        entry_elevation_angle_deg = item["entry_elevation_angle_deg"]
        velocity_km_s = item["velocity_km_s0"]
        color = entry_elevation_colors[float(entry_elevation_angle_deg)]
        linestyle = velocity_linestyles[float(velocity_km_s)]
        axes[2].plot(
            result["temperature_k"],
            result["altitude_km"],
            color=color,
            linestyle=linestyle,
            lw=2.0,
        )
        draw_peak_ablation_line(axes[2], result, color, linestyle, show_peak_ablation_lines)
    axes[2].set_xlabel("Temperature [K]")
    axes[2].set_ylabel("Altitude [km]")
    axes[2].grid(True, alpha=0.25)
    axes[2].set_title("(c) Temperature")
    axes[2].set_ylim(70, 130)

    for item in results:
        result = item["data"]
        entry_elevation_angle_deg = item["entry_elevation_angle_deg"]
        velocity_km_s = item["velocity_km_s0"]
        color = entry_elevation_colors[float(entry_elevation_angle_deg)]
        linestyle = velocity_linestyles[float(velocity_km_s)]
        axes[3].plot(
            result["mass_kg"],
            result["altitude_km"],
            color=color,
            linestyle=linestyle,
            lw=2.0,
        )
        draw_peak_ablation_line(axes[3], result, color, linestyle, show_peak_ablation_lines)
    axes[3].set_xscale("log")
    axes[3].set_xlabel("Mass [kg]")
    axes[3].grid(True, which="both", alpha=0.25)
    axes[3].set_title("(d) Mass")
    axes[3].set_ylim(70, 130)

    entry_elevation_handles = [
        Line2D([0], [0], color=color, lw=2.0, label=rf"$\alpha={angle:.0f}^\circ$")
        for angle, color in entry_elevation_colors.items()
    ]
    velocity_handles = [
        Line2D(
            [0],
            [0],
            color="0.2",
            linestyle=linestyle,
            lw=2.0,
            label=f"{velocity:.0f} km s$^{{-1}}$",
        )
        for velocity, linestyle in velocity_linestyles.items()
    ]
    axes[0].legend(
        handles=entry_elevation_handles,
        frameon=False,
        loc="upper left",
        title="Entry elevation",
    )
    axes[1].legend(handles=velocity_handles, frameon=False, loc="lower right", title="Velocity")

    fig.savefig(FIG_DIR / "meteor_ablation_single_column.pdf")
    plt.close(fig)


def make_figure2(gamma):
    shifted_velocities = FIGURE2_BASELINE_VELOCITIES_KM_S * gamma ** (-1.0 / 3.0)
    entry_elevation_angles = FIGURE1_ENTRY_ELEVATION_ANGLES_DEG
    table_rows = []
    fig, axes = plt.subplots(
        1,
        len(entry_elevation_angles),
        figsize=(3.2 * len(entry_elevation_angles), 3.7),
        sharey=True,
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes)

    for ax, entry_elevation_angle_deg in zip(axes, entry_elevation_angles):
        baseline = [
            simulate_case(
                v,
                density_scale=1.0,
                entry_elevation_angle_deg=entry_elevation_angle_deg,
            )
            for v in FIGURE2_BASELINE_VELOCITIES_KM_S
        ]
        scaled = [
            simulate_case(
                v,
                density_scale=gamma,
                entry_elevation_angle_deg=entry_elevation_angle_deg,
            )
            for v in shifted_velocities
        ]

        baseline_peak_altitudes = np.array([item["peak_altitude_km"] for item in baseline])
        scaled_peak_altitudes = np.array([item["peak_altitude_km"] for item in scaled])
        model_velocities = interp1d_linear_extrapolate(
            scaled_peak_altitudes,
            shifted_velocities,
            baseline_peak_altitudes,
        )

        for base_v, peak_h, simple_v, model_v in zip(
            FIGURE2_BASELINE_VELOCITIES_KM_S,
            baseline_peak_altitudes,
            shifted_velocities,
            model_velocities,
        ):
            inferred_gamma = (base_v / model_v) ** 3
            gamma_error_percent = 100.0 * (inferred_gamma - gamma) / gamma
            table_rows.append(
                {
                    "gamma": float(gamma),
                    "entry_elevation_angle_deg": float(entry_elevation_angle_deg),
                    "baseline_velocity_km_s": float(base_v),
                    "peak_altitude_km": float(peak_h),
                    "simple_velocity_km_s": float(simple_v),
                    "model_velocity_km_s": float(model_v),
                    "simple_delta_km_s": float(simple_v - base_v),
                    "model_delta_km_s": float(model_v - base_v),
                    "inferred_gamma": float(inferred_gamma),
                    "gamma_error_percent": float(gamma_error_percent),
                }
            )

        ax.plot(
            FIGURE2_BASELINE_VELOCITIES_KM_S,
            baseline_peak_altitudes,
            "-o",
            color="black",
            lw=2.0,
            label=r"MSIS, $\rho_a$",
        )
        ax.plot(
            shifted_velocities,
            scaled_peak_altitudes,
            "--s",
            color="black",
            lw=2.0,
            label=rf"Scaled MSIS, {gamma:.1f}$\rho_a$",
        )

        for base_v, shifted_v, peak_h in zip(
            FIGURE2_BASELINE_VELOCITIES_KM_S,
            shifted_velocities,
            baseline_peak_altitudes,
        ):
            ax.plot([shifted_v, base_v], [peak_h, peak_h], ":", color="tab:red", lw=1.5)

        ax.scatter(
            shifted_velocities,
            baseline_peak_altitudes,
            marker="x",
            s=60,
            color="tab:red",
            label=rf"Predicted, $v_1 = v_0 {gamma:.1f}^{{-1/3}}$",
            zorder=3,
        )
        ax.scatter(
            model_velocities,
            baseline_peak_altitudes,
            marker="o",
            s=54,
            facecolors="none",
            edgecolors="tab:red",
            linewidths=1.6,
            label="Interpolated model shift",
            zorder=4,
        )

        for base_v, peak_h in zip(FIGURE2_BASELINE_VELOCITIES_KM_S, baseline_peak_altitudes):
            ax.annotate(
                f"{base_v:.0f}",
                xy=(base_v, peak_h),
                xytext=(0, 7),
                textcoords="offset points",
                ha="center",
                fontsize=9,
            )

        ax.set_title(rf"$\alpha={entry_elevation_angle_deg:.0f}^\circ$")
        ax.set_xlabel("Geocentric velocity [km s$^{-1}$]")
        ax.grid(True, alpha=0.25)

    axes[0].set_ylabel(r"Altitude of peak $|dm/dt|$ [km]")
    axes[-1].legend(frameon=False, loc="best")

    fig.savefig(FIG_DIR / f"peak_ablation_height-{gamma:.2f}.pdf")
    plt.close(fig)
    return table_rows


def write_velocity_shift_table(rows):
    lines = [
        r"\begin{table}",
        r"\caption{Velocity shifts inferred from the simple density-scaling law and from the full ablation model. The model velocity is obtained by linearly interpolating, with endpoint extrapolation when needed, the velocity in the scaled-density model that gives the same peak-$|dm/dt|$ altitude as the unscaled model. The inferred density scale $\hat{\gamma}_{\mathrm{model}}=(v_0/v_{\mathrm{model}})^3$ shows the neutral-density change that would be inferred if the simplified scaling law were applied to the model velocity shift. The final row for each $\gamma$ gives the mean inferred $\hat{\gamma}_{\mathrm{model}}$ and error across all cases.}",
        r"\label{tab:velocity-shift}",
        r"\begin{tabular}{rrrrrrrrrr}",
        r"\tophline",
        r"$\gamma$ & $\alpha$ & $v_0$ & $h_{\mathrm{peak}}$ & $v_{\mathrm{simple}}$ & $\Delta v_{\mathrm{simple}}$ & $v_{\mathrm{model}}$ & $\Delta v_{\mathrm{model}}$ & $\hat{\gamma}_{\mathrm{model}}$ & error \\",
        r" & deg & km~s$^{-1}$ & km & km~s$^{-1}$ & km~s$^{-1}$ & km~s$^{-1}$ & km~s$^{-1}$ &  & \% \\",
        r"\middlehline",
    ]
    current_gamma = None
    group = []
    for row in rows:
        if row['gamma'] != current_gamma:
            if current_gamma is not None:
                for g_row in group:
                    lines.append(
                        f"{g_row['gamma']:.1f} & "
                        f"{g_row['entry_elevation_angle_deg']:.1f} & "
                        f"{g_row['baseline_velocity_km_s']:.1f} & "
                        f"{g_row['peak_altitude_km']:.1f} & "
                        f"{g_row['simple_velocity_km_s']:.1f} & "
                        f"{g_row['simple_delta_km_s']:.1f} & "
                        f"{g_row['model_velocity_km_s']:.1f} & "
                        f"{g_row['model_delta_km_s']:.1f} & "
                        f"{g_row['inferred_gamma']:.2f} & "
                        f"{g_row['gamma_error_percent']:.0f} \\\\"
                    )
                if current_gamma in [1.2, 0.8]:
                    inferred_gammas = [r['inferred_gamma'] for r in group]
                    errors = [r['gamma_error_percent'] for r in group]
                    mean_gamma = np.mean(inferred_gammas)
                    mean_error = np.mean(errors)
                    lines.append(
                        f"{current_gamma:.1f} & Mean & & & & & & & {mean_gamma:.2f} & {mean_error:.0f} \\\\"
                    )
            current_gamma = row['gamma']
            group = []
        group.append(row)
    # Last group
    for g_row in group:
        lines.append(
            f"{g_row['gamma']:.1f} & "
            f"{g_row['entry_elevation_angle_deg']:.1f} & "
            f"{g_row['baseline_velocity_km_s']:.1f} & "
            f"{g_row['peak_altitude_km']:.1f} & "
            f"{g_row['simple_velocity_km_s']:.1f} & "
            f"{g_row['simple_delta_km_s']:.1f} & "
            f"{g_row['model_velocity_km_s']:.1f} & "
            f"{g_row['model_delta_km_s']:.1f} & "
            f"{g_row['inferred_gamma']:.2f} & "
            f"{g_row['gamma_error_percent']:.0f} \\\\"
        )
    if current_gamma in [1.2, 0.8]:
        inferred_gammas = [r['inferred_gamma'] for r in group]
        errors = [r['gamma_error_percent'] for r in group]
        mean_gamma = np.mean(inferred_gammas)
        mean_error = np.mean(errors)
        lines.append(
            f"{current_gamma:.1f} & Mean & & & & & & & {mean_gamma:.2f} & {mean_error:.0f} \\\\"
        )
    lines.extend(
        [
            r"\bottomhline",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )
    VELOCITY_SHIFT_TABLE_PATH.write_text("\n".join(lines))


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    make_figure1()
    table_rows = []
    table_rows.extend(make_figure2(gamma=1.2))
    make_figure2(gamma=1.0)
    table_rows.extend(make_figure2(gamma=0.8))
    write_velocity_shift_table(table_rows)

    print(f"Saved {FIG_DIR / 'meteor_ablation_single_column.pdf'}")
    for gamma in (1.2, 1.0, 0.8):
        print(f"Saved {FIG_DIR / f'peak_ablation_height-{gamma:.2f}.pdf'}")
    print(f"Saved {VELOCITY_SHIFT_TABLE_PATH}")


if __name__ == "__main__":
    main()
