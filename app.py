import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from scipy.interpolate import interp1d

st.set_page_config(layout="wide", page_title="Shooter Calibration Surface")

IN_TO_M = 0.0254
G = 9.80665

# -------------------------
# Raw tuning data
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

    coeffs = np.polyfit(x, y, degree_map[fit_type])
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


def solve_exit_speed(distance_m, angle_deg, target_height_m, launch_height_m, robot_velocity_mps):
    """Return ball exit speed relative to the robot for a target height."""
    theta = np.deg2rad(angle_deg)
    cos_theta = np.cos(theta)
    tan_theta = np.tan(theta)
    dy = target_height_m - launch_height_m

    a = distance_m * tan_theta - dy
    b = -robot_velocity_mps * distance_m * tan_theta
    c = -0.5 * G * distance_m ** 2
    discriminant = b ** 2 - 4.0 * a * c

    with np.errstate(invalid="ignore", divide="ignore"):
        sqrt_disc = np.sqrt(discriminant)
        q1 = (-b + sqrt_disc) / (2.0 * a)
        q2 = (-b - sqrt_disc) / (2.0 * a)

    q_candidates = np.stack([q1, q2])
    exit_candidates = (q_candidates - robot_velocity_mps) / cos_theta
    exit_candidates = np.where(
        (discriminant >= 0.0)
        & (np.abs(a) > 1e-9)
        & (np.abs(cos_theta) > 1e-9)
        & (q_candidates > 0.0)
        & (exit_candidates > 0.0),
        exit_candidates,
        np.nan
    )

    return np.nanmin(exit_candidates, axis=0)


def add_surface(fig, x, y, z, name, color_scale, opacity):
    fig.add_trace(
        go.Surface(
            x=x,
            y=y,
            z=z,
            name=name,
            colorscale=color_scale,
            showscale=False,
            opacity=opacity,
            hovertemplate=(
                "Distance: %{x:.2f} m<br>"
                "Hood angle: %{y:.1f} deg<br>"
                "Exit speed: %{z:.2f} m/s"
                "<extra>" + name + "</extra>"
            )
        )
    )


st.title("FRC Shooter Calibration")

tab_calibration, tab_theory = st.tabs([
    "Calibration fit",
    "Theoretical scoring surface"
])

with tab_calibration:
    fit_type = st.selectbox(
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

    d_min = st.slider(
        "Minimum Distance (m)",
        float(distance_pts.min()),
        float(distance_pts.max()),
        float(distance_pts.min())
    )

    d_max = st.slider(
        "Maximum Distance (m)",
        float(distance_pts.min()),
        float(distance_pts.max()),
        float(distance_pts.max())
    )

    samples = st.slider(
        "Samples",
        50,
        1000,
        300
    )

    d = np.linspace(d_min, d_max, samples)
    hood = hood_model(d)
    shooter = shooter_model(d)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter3d(
            x=d,
            y=hood,
            z=shooter,
            mode="lines",
            line=dict(
                width=8,
                color="royalblue"
            ),
            name="Shooter Curve"
        )
    )

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

    shooter_metrics = compute_fit_metrics(distance_pts, shooter_rpm_pts, shooter_model)
    hood_metrics = compute_fit_metrics(distance_pts, hood_angle_pts, hood_model)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Shooter fit R²", f"{shooter_metrics['r2']:.4f}")
        st.metric("Shooter RMSE", f"{shooter_metrics['rmse']:.2f}")
    with col2:
        st.metric("Hood fit R²", f"{hood_metrics['r2']:.4f}")
        st.metric("Hood RMSE", f"{hood_metrics['rmse']:.2f}")

    df = pd.DataFrame({
        "Distance": d,
        "Hood Angle": hood,
        "Shooter RPM": shooter,
    })

    st.dataframe(df)

with tab_theory:
    st.subheader("Projectile scoring envelope")

    c1, c2, c3 = st.columns(3)
    with c1:
        theory_d_min = st.number_input(
            "Minimum distance (m)",
            min_value=0.1,
            max_value=20.0,
            value=float(distance_pts.min()),
            step=0.1
        )
        theory_d_max = st.number_input(
            "Maximum distance (m)",
            min_value=0.1,
            max_value=20.0,
            value=float(distance_pts.max()),
            step=0.1
        )
        distance_samples = st.slider("Distance samples", 20, 300, 120)
    with c2:
        angle_min = st.number_input(
            "Minimum hood angle (deg)",
            min_value=1.0,
            max_value=89.0,
            value=10.0,
            step=1.0
        )
        angle_max = st.number_input(
            "Maximum hood angle (deg)",
            min_value=1.0,
            max_value=89.0,
            value=65.0,
            step=1.0
        )
        angle_samples = st.slider("Angle samples", 20, 300, 140)
    with c3:
        launch_height_in = st.number_input(
            "Launch height (in)",
            min_value=0.0,
            max_value=120.0,
            value=28.0,
            step=1.0
        )
        goal_center_height_in = st.number_input(
            "Goal center height (in)",
            min_value=0.0,
            max_value=180.0,
            value=72.0,
            step=1.0
        )
        opening_height_in = st.number_input(
            "Opening height (in)",
            min_value=1.0,
            max_value=120.0,
            value=41.7,
            step=0.1
        )

    c4, c5 = st.columns(2)
    with c4:
        robot_velocity_mps = st.slider(
            "Robot velocity along shot (m/s)",
            -5.0,
            5.0,
            0.0,
            0.1,
            help="Positive means driving toward the goal; negative means driving away."
        )
    with c5:
        max_exit_speed = st.slider(
            "Max displayed exit speed (m/s)",
            1.0,
            40.0,
            20.0,
            0.5
        )

    theory_d_min, theory_d_max = sorted([theory_d_min, theory_d_max])
    angle_min, angle_max = sorted([angle_min, angle_max])

    goal_center_m = goal_center_height_in * IN_TO_M
    half_opening_m = opening_height_in * IN_TO_M / 2.0
    target_lower_m = goal_center_m - half_opening_m
    target_upper_m = goal_center_m + half_opening_m
    launch_height_m = launch_height_in * IN_TO_M

    distances = np.linspace(theory_d_min, theory_d_max, distance_samples)
    angles = np.linspace(angle_min, angle_max, angle_samples)
    distance_grid, angle_grid = np.meshgrid(distances, angles)

    lower_speed = solve_exit_speed(
        distance_grid,
        angle_grid,
        target_lower_m,
        launch_height_m,
        robot_velocity_mps
    )
    upper_speed = solve_exit_speed(
        distance_grid,
        angle_grid,
        target_upper_m,
        launch_height_m,
        robot_velocity_mps
    )

    speed_floor = np.minimum(lower_speed, upper_speed)
    speed_ceiling = np.maximum(lower_speed, upper_speed)
    scoring_mask = (
        np.isfinite(speed_floor)
        & np.isfinite(speed_ceiling)
        & (speed_floor <= max_exit_speed)
    )

    lower_display = np.where(scoring_mask, speed_floor, np.nan)
    upper_display = np.where(scoring_mask, np.minimum(speed_ceiling, max_exit_speed), np.nan)

    fig = go.Figure()
    add_surface(fig, distance_grid, angle_grid, upper_display, "Upper limit", "Greens", 0.78)
    add_surface(fig, distance_grid, angle_grid, lower_display, "Lower limit", "Reds", 0.78)

    fig.update_layout(
        scene=dict(
            xaxis_title="Distance to goal plane (m)",
            yaxis_title="Launch / hood angle (deg)",
            zaxis_title="Exit velocity (m/s)",
            zaxis=dict(range=[0, max_exit_speed])
        ),
        height=820,
        legend=dict(
            x=0.02,
            y=0.98
        ),
        margin=dict(l=0, r=0, t=40, b=0)
    )

    st.plotly_chart(fig, use_container_width=True)

    valid_cells = int(np.count_nonzero(scoring_mask))
    total_cells = int(scoring_mask.size)
    min_speed = np.nanmin(lower_display) if valid_cells else np.nan
    max_speed = np.nanmax(upper_display) if valid_cells else np.nan

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Valid angle-distance cells", f"{valid_cells:,} / {total_cells:,}")
    m2.metric("Lowest exit speed", "n/a" if np.isnan(min_speed) else f"{min_speed:.2f} m/s")
    m3.metric("Highest displayed exit speed", "n/a" if np.isnan(max_speed) else f"{max_speed:.2f} m/s")
    m4.metric("Vertical scoring window", f"{target_lower_m:.2f} m to {target_upper_m:.2f} m")

    scoring_df = pd.DataFrame({
        "Distance (m)": distance_grid[scoring_mask],
        "Hood Angle (deg)": angle_grid[scoring_mask],
        "Lower Exit Speed (m/s)": lower_display[scoring_mask],
        "Upper Exit Speed (m/s)": upper_display[scoring_mask],
    })

    st.dataframe(scoring_df, use_container_width=True)
