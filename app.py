import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from scipy.interpolate import interp1d

st.set_page_config(layout="wide")

INCH_TO_M = 0.0254
MPS_TO_FPS = 3.280839895
GRAVITY_MPS2 = 9.80665

# -------------------------
# Raw Tuning data
# -------------------------

distance_pts = np.array([
    1.70,
    2.000,
    2.215,
    2.495,
    2.888,
    3.222,
    3.570,
    3.830,
    4.150,
    4.750
])

shooter_rpm_pts = np.array([
    1855,
    1875,
    1940,
    2015,
    2065,
    2130,
    2200,
    2240,
    2340,
    2365
])

hood_angle_pts = np.array([
    17.5,
    18.5,
    19.0,
    19.5,
    20.5,
    21.5,
    22.5,
    23.0,
    24.5,
    26.5
])

st.title("Shooter Plot")

fit_type = st.sidebar.selectbox(
    "Regression Type",
    [
        "Linear",
        "Quadratic",
        "Cubic",
        "Quartic",
        "Spline"
    ],
    index=4
)

st.sidebar.header("Distance Range")
d_min = st.sidebar.slider(
    "Minimum Distance (m)",
    float(distance_pts.min()),
    float(distance_pts.max()),
    float(distance_pts.min())
)

d_max = st.sidebar.slider(
    "Maximum Distance (m)",
    float(distance_pts.min()),
    float(distance_pts.max()),
    float(distance_pts.max())
)

samples = st.sidebar.slider(
    "Fit Samples",
    50,
    1000,
    300
)

st.sidebar.header("Ballistic Model")
target_height_in = st.sidebar.number_input(
    "Target Height (in)",
    min_value=0.0,
    value=72.0,
    step=1.0
)

launch_height_in = st.sidebar.number_input(
    "Launch Height (in)",
    min_value=0.0,
    value=9.92,
    step=1.0
)

angle_min, angle_max = st.sidebar.slider(
    "Hood Angle Range (deg)",
    1.0,
    89.0,
    (17.5, 55.0),
    step=0.5
)

surface_distance_samples = st.sidebar.slider(
    "Surface Distance Samples",
    20,
    300,
    120
)

surface_angle_samples = st.sidebar.slider(
    "Surface Angle Samples",
    20,
    300,
    120
)

solution_distance_samples = st.sidebar.slider(
    "Solution Table Distances",
    4,
    30,
    10
)

solutions_per_distance = st.sidebar.slider(
    "Solutions per Distance",
    2,
    8,
    4
)

def fit_curve(x, y, fit_type):

    if fit_type == "Spline":
        return interp1d(
            x,
            y,
            kind="cubic",
            fill_value="extrapolate"
        )

    degree_map = {
        "Linear": 1,
        "Quadratic": 2,
        "Cubic": 3,
        "Quartic": 4
    }

    degree = degree_map[fit_type]

    coeffs = np.polyfit(x, y, degree)
    poly = np.poly1d(coeffs)

    return lambda xx: poly(xx)


def compute_fit_metrics(x, y, model):
    y_pred = model(x)
    residuals = y - y_pred
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    rmse = np.sqrt(np.mean(residuals ** 2))
    return {
        "r2": r2,
        "rmse": rmse,
        "num_points": len(y)
    }


def solve_exit_velocity(distance_m, launch_angle_deg, launch_height_m, target_height_m):
    theta = np.deg2rad(launch_angle_deg)
    height_delta_m = target_height_m - launch_height_m

    with np.errstate(divide="ignore", invalid="ignore"):
        denominator = 2.0 * (np.cos(theta) ** 2) * (
            distance_m * np.tan(theta) - height_delta_m
        )
        velocity_mps = np.sqrt((GRAVITY_MPS2 * distance_m ** 2) / denominator)

    return np.where(denominator > 0.0, velocity_mps, np.nan)


def sample_valid_solutions(
    distances_m,
    angle_range_deg,
    solutions_per_distance,
    launch_height_m,
    target_height_m,
    candidate_count=240
):
    candidate_angles = np.linspace(
        angle_range_deg[0],
        angle_range_deg[1],
        candidate_count
    )
    rows = []

    for distance_m in distances_m:
        candidate_velocities = solve_exit_velocity(
            distance_m,
            candidate_angles,
            launch_height_m,
            target_height_m
        )
        valid_indices = np.flatnonzero(np.isfinite(candidate_velocities))

        if len(valid_indices) == 0:
            continue

        sample_count = min(solutions_per_distance, len(valid_indices))
        sampled_positions = np.linspace(0, len(valid_indices) - 1, sample_count)
        sampled_indices = valid_indices[np.unique(np.round(sampled_positions).astype(int))]

        for sampled_index in sampled_indices:
            velocity_mps = candidate_velocities[sampled_index]
            rows.append({
                "Distance (m)": distance_m,
                "Hood Angle (deg)": candidate_angles[sampled_index],
                "Exit Velocity (m/s)": velocity_mps,
                "Exit Velocity (ft/s)": velocity_mps * MPS_TO_FPS,
            })

    return pd.DataFrame(rows)


shooter_model = fit_curve(
    distance_pts,
    shooter_rpm_pts,
    fit_type
)

hood_model = fit_curve(
    distance_pts,
    hood_angle_pts,
    fit_type
)

# -------------------------
# Generate curve
# -------------------------

d = np.linspace(d_min, d_max, samples)

hood = hood_model(d)
shooter = shooter_model(d)

target_height_m = target_height_in * INCH_TO_M
launch_height_m = launch_height_in * INCH_TO_M

surface_distances = np.linspace(d_min, d_max, surface_distance_samples)
surface_angles = np.linspace(angle_min, angle_max, surface_angle_samples)
distance_grid, angle_grid = np.meshgrid(surface_distances, surface_angles)
velocity_grid = solve_exit_velocity(
    distance_grid,
    angle_grid,
    launch_height_m,
    target_height_m
)

hood_surface_line = hood_model(surface_distances)
velocity_surface_line = solve_exit_velocity(
    surface_distances,
    hood_surface_line,
    launch_height_m,
    target_height_m
)

measured_required_velocity = solve_exit_velocity(
    distance_pts,
    hood_angle_pts,
    launch_height_m,
    target_height_m
)

# -------------------------
# 3D Plot
# -------------------------



empirical_tab, ballistic_tab = st.tabs([
    "Empirical Fit",
    "Ballistic Surface"
])

with empirical_tab:
    fig = go.Figure()

    # Main shooter curve
    fig.add_trace(
            go.Scatter3d(
            x=d,
            y=hood,
            z=shooter,
            mode="lines",
            line=dict(
                width=8,
                color='royalblue'
            ),
            name="Shooter Curve"
        )
    )

    # Raw calibration points
    fig.add_trace(
        go.Scatter3d(
            x=distance_pts,
            y=hood_angle_pts,
            z=shooter_rpm_pts,
            mode="markers",
            marker=dict(size=5),
            name="Measured Points"
        )
    )

    fig.update_layout(
        scene=dict(
            xaxis_title="Distance (m)",
            yaxis_title="Hood Angle (deg)",
            zaxis_title="Shooter RPM"
        ),
        height=800
    )

    st.plotly_chart(fig, use_container_width=True)

    # -------------------------
    # Fit quality metrics
    # -------------------------

    shooter_metrics = compute_fit_metrics(distance_pts, shooter_rpm_pts, shooter_model)
    hood_metrics = compute_fit_metrics(distance_pts, hood_angle_pts, hood_model)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Shooter fit R²", f"{shooter_metrics['r2']:.4f}")
        st.metric("Shooter RMSE", f"{shooter_metrics['rmse']:.2f}")
    with col2:
        st.metric("Hood fit R²", f"{hood_metrics['r2']:.4f}")
        st.metric("Hood RMSE", f"{hood_metrics['rmse']:.2f}")

    # -------------------------
    # Data Table
    # -------------------------

    df = pd.DataFrame({
        "Distance": d,
        "Hood Angle": hood,
        "Shooter RPM": shooter,
    })

    st.dataframe(df)

with ballistic_tab:
    valid_velocity_mask = np.isfinite(velocity_grid)
    valid_velocity_count = int(np.count_nonzero(valid_velocity_mask))
    total_velocity_count = int(velocity_grid.size)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Target Height", f"{target_height_in:.1f} in")
    with col2:
        st.metric("Launch Height", f"{launch_height_in:.1f} in")
    with col3:
        st.metric(
            "Valid Surface Points",
            f"{valid_velocity_count:,} / {total_velocity_count:,}"
        )

    ballistic_fig = go.Figure()

    ballistic_fig.add_trace(
        go.Surface(
            x=distance_grid,
            y=angle_grid,
            z=velocity_grid,
            colorscale="Viridis",
            colorbar=dict(title="m/s"),
            connectgaps=False,
            opacity=0.88,
            name="Required Exit Velocity"
        )
    )

    hood_line_mask = np.isfinite(velocity_surface_line)
    ballistic_fig.add_trace(
        go.Scatter3d(
            x=surface_distances[hood_line_mask],
            y=hood_surface_line[hood_line_mask],
            z=velocity_surface_line[hood_line_mask],
            mode="lines",
            line=dict(width=8, color="white"),
            name="Current Hood Curve"
        )
    )

    measured_velocity_mask = np.isfinite(measured_required_velocity)
    ballistic_fig.add_trace(
        go.Scatter3d(
            x=distance_pts[measured_velocity_mask],
            y=hood_angle_pts[measured_velocity_mask],
            z=measured_required_velocity[measured_velocity_mask],
            mode="markers",
            marker=dict(size=5, color="crimson"),
            name="Measured Hood Points"
        )
    )

    ballistic_fig.update_layout(
        scene=dict(
            xaxis_title="Distance (m)",
            yaxis_title="Hood Angle (deg)",
            zaxis_title="Exit Velocity (m/s)"
        ),
        height=800
    )

    if valid_velocity_count == 0:
        st.warning("No valid ballistic solutions in the selected distance and angle range.")
    else:
        st.plotly_chart(ballistic_fig, use_container_width=True)

        valid_velocities = velocity_grid[valid_velocity_mask]
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Minimum Exit Velocity", f"{np.nanmin(valid_velocities):.2f} m/s")
        with col2:
            st.metric("Maximum Exit Velocity", f"{np.nanmax(valid_velocities):.2f} m/s")
        with col3:
            st.metric("Median Exit Velocity", f"{np.nanmedian(valid_velocities):.2f} m/s")

    st.caption(
        "Simple point-mass trajectory: no drag, spin, ball diameter, or target opening tolerance."
    )

    sampled_solution_distances = np.linspace(
        d_min,
        d_max,
        solution_distance_samples
    )
    sampled_solutions = sample_valid_solutions(
        sampled_solution_distances,
        (angle_min, angle_max),
        solutions_per_distance,
        launch_height_m,
        target_height_m
    )

    st.subheader("Sampled Valid Solutions")
    if sampled_solutions.empty:
        st.info("No sampled solutions are valid with the current model settings.")
    else:
        st.dataframe(
            sampled_solutions.style.format({
                "Distance (m)": "{:.3f}",
                "Hood Angle (deg)": "{:.2f}",
                "Exit Velocity (m/s)": "{:.2f}",
                "Exit Velocity (ft/s)": "{:.1f}",
            })
        )
