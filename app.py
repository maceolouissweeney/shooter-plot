import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from scipy.interpolate import interp1d

st.set_page_config(layout="wide")

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

target_height_m = 1.8288  # 72 inches above ground, FRC fuel target opening height in meters

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


def compute_required_velocity(distance, angle_deg, y0, y_target, g=9.81):
    theta = np.deg2rad(angle_deg)
    dx = distance
    dy = y_target - y0
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    denom = dx * np.tan(theta) - dy
    if denom <= 0 or cos_theta == 0:
        return np.nan
    v2 = g * dx ** 2 / (2 * cos_theta ** 2 * denom)
    if v2 <= 0:
        return np.nan
    return np.sqrt(v2)

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

# -------------------------
# Generate curve
# -------------------------

d = np.linspace(d_min, d_max, samples)

hood = hood_model(d)
shooter = shooter_model(d)
# -------------------------
# 3D Plot
# -------------------------



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
# Ballistic trajectory surface
# -------------------------

st.markdown("### Ballistic solution surface")

y0 = st.slider(
    "Shooter launch height (m)",
    0.5,
    1.5,
    1.0,
    step=0.05
)

target_height = st.number_input(
    "Target height (m)",
    value=target_height_m,
    step=0.01,
    format="%.4f"
)

angle_min = st.slider(
    "Minimum launch angle (deg)",
    10,
    60,
    20
)

angle_max = st.slider(
    "Maximum launch angle (deg)",
    40,
    80,
    60
)

angle_samples = st.slider(
    "Angle samples",
    10,
    80,
    40
)

distance_samples = st.slider(
    "Trajectory distance samples",
    10,
    80,
    40
)

surface_angles = np.linspace(angle_min, angle_max, angle_samples)
surface_distances = np.linspace(d_min, d_max, distance_samples)

velocities = np.full((len(surface_angles), len(surface_distances)), np.nan)
for i, angle in enumerate(surface_angles):
    for j, distance in enumerate(surface_distances):
        velocities[i, j] = compute_required_velocity(distance, angle, y0, target_height)

valid_count = np.count_nonzero(~np.isnan(velocities))

st.write(
    f"Valid ballistic solutions in surface: {valid_count} / {velocities.size}"
)

fig_ballistic = go.Figure(
    data=[
        go.Surface(
            x=surface_distances,
            y=surface_angles,
            z=velocities,
            colorscale="Viridis",
            colorbar=dict(title="Exit velocity (m/s)"),
            showscale=True
        )
    ]
)

fig_ballistic.update_layout(
    scene=dict(
        xaxis_title="Distance (m)",
        yaxis_title="Launch angle (deg)",
        zaxis_title="Required exit velocity (m/s)"
    ),
    height=700
)

st.plotly_chart(fig_ballistic, use_container_width=True)

# -------------------------
# Data Table
# -------------------------

df = pd.DataFrame({
    "Distance": d,
    "Hood Angle": hood,
    "Shooter RPM": shooter,
})

st.dataframe(df)

