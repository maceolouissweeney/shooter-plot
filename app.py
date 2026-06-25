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

tunnel_rpm_pts = np.array([
    1425,
    1450,
    1500,
    1550,
    1650,
    1750,
    1800,
    1950,
    2150,
    2150
])


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

tunnel_model = fit_curve(
    distance_pts,
    tunnel_rpm_pts,
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
tunnel = tunnel_model(d)
# -------------------------
# 3D Plot
# -------------------------

show_points = st.checkbox(
    "Show Calibration Points",
    value=True
)



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
            color=tunnel,
            colorscale="Viridis",
            colorbar=dict(title="Tunnel RPM")
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
# Data Table
# -------------------------

df = pd.DataFrame({
    "Distance": d,
    "Hood Angle": hood,
    "Shooter RPM": shooter,
    "Tunnel RPM": tunnel
})

st.dataframe(df)

