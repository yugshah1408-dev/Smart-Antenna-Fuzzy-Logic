
import streamlit as st
import numpy as np
import skfuzzy as fuzz
import pandas as pd

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Smart Antenna Beam Selection",
    page_icon="📡",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

    .main-title {
        font-size: 36px;
        font-weight: 700;
        text-align: center;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        font-size: 17px;
        color: #666;
        margin-bottom: 30px;
    }

    /* ========================================================
       RESULT BOX
       ======================================================== */

    .result-box {
        padding: 25px;
        border-radius: 15px;
        background-color: #f5f7fa;
        color: #111827 !important;
        text-align: center;
        margin-top: 10px;
    }

    .result-label {
        display: block;
        color: #374151 !important;
        font-size: 18px;
        font-weight: 600;
        margin-bottom: 12px;
    }

    .beam-output {
        display: block;
        color: #111827 !important;
        font-size: 32px;
        font-weight: 700;
    }

    .angle-output {
        display: block;
        color: #111827 !important;
        font-size: 25px;
        font-weight: 600;
    }

    /* ========================================================
       SECTION TITLE
       ======================================================== */

    .section-title {
        font-size: 22px;
        font-weight: 600;
        margin-top: 20px;
        margin-bottom: 10px;
    }

</style>
""", unsafe_allow_html=True)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">📡 Smart Antenna Beam Selection</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Fuzzy Logic Based Beam Selection System'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# FUZZY UNIVERSES
# ============================================================

# RSSI: -100 to -30 dBm
rssi = np.arange(-100, -29.9, 0.1)

# SNR: 0 to 30 dB
snr = np.arange(0, 30.1, 0.1)

# AoA: 0 to 90 degrees
aoa = np.arange(0, 90.1, 0.1)


# ============================================================
# INPUT MEMBERSHIP FUNCTIONS
# ============================================================

# ------------------------------------------------------------
# RSSI membership functions
# ------------------------------------------------------------

rssi_weak = fuzz.trimf(
    rssi,
    [-100, -100, -70]
)

rssi_moderate = fuzz.trimf(
    rssi,
    [-85, -65, -45]
)

rssi_strong = fuzz.trimf(
    rssi,
    [-60, -30, -30]
)


# ------------------------------------------------------------
# SNR membership functions
# ------------------------------------------------------------

snr_low = fuzz.trimf(
    snr,
    [0, 0, 10]
)

snr_moderate = fuzz.trimf(
    snr,
    [5, 15, 25]
)

snr_high = fuzz.trimf(
    snr,
    [20, 30, 30]
)


# ------------------------------------------------------------
# AoA membership functions
# ------------------------------------------------------------

aoa_aligned = fuzz.trimf(
    aoa,
    [0, 0, 30]
)

aoa_oblique = fuzz.trimf(
    aoa,
    [15, 45, 75]
)

aoa_edge = fuzz.trimf(
    aoa,
    [60, 90, 90]
)


# ============================================================
# OUTPUT VALUES
# ============================================================

output_peaks = {
    "Narrow": 0.2,
    "Medium": 1.1,
    "Wide": 2.0
}


# ============================================================
# FUZZY RULES
# ============================================================

rules = [

    # Rule 1
    {
        "rssi": "Weak",
        "snr": "Low",
        "aoa": "Aligned",
        "output": "Narrow"
    },

    # Rule 2
    {
        "rssi": "Weak",
        "snr": "Moderate",
        "aoa": "Aligned",
        "output": "Narrow"
    },

    # Rule 3
    {
        "rssi": "Weak",
        "snr": None,
        "aoa": "Oblique",
        "output": "Medium"
    },

    # Rule 4
    {
        "rssi": "Moderate",
        "snr": None,
        "aoa": "Aligned",
        "output": "Medium"
    },

    # Rule 5
    {
        "rssi": "Moderate",
        "snr": "High",
        "aoa": None,
        "output": "Wide"
    },

    # Rule 6
    {
        "rssi": "Strong",
        "snr": None,
        "aoa": None,
        "output": "Wide"
    },

    # Rule 7
    {
        "rssi": None,
        "snr": "Low",
        "aoa": None,
        "output": "Narrow"
    },

    # Rule 8
    {
        "rssi": None,
        "snr": None,
        "aoa": "Edge",
        "output": "Wide"
    },

    # Rule 9
    {
        "rssi": "Weak",
        "snr": "High",
        "aoa": None,
        "output": "Medium"
    },

    # Rule 10
    {
        "rssi": "Moderate",
        "snr": "Low",
        "aoa": "Oblique",
        "output": "Narrow"
    },

    # Rule 11
    {
        "rssi": "Strong",
        "snr": "High",
        "aoa": "Aligned",
        "output": "Wide"
    },

    # Rule 12
    {
        "rssi": "Moderate",
        "snr": "Moderate",
        "aoa": "Oblique",
        "output": "Wide"
    }
]


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_beam(input_rssi, input_snr, input_aoa):

    # --------------------------------------------------------
    # FUZZIFICATION
    # --------------------------------------------------------

    rssi_mu = {
        "Weak": fuzz.interp_membership(
            rssi,
            rssi_weak,
            input_rssi
        ),

        "Moderate": fuzz.interp_membership(
            rssi,
            rssi_moderate,
            input_rssi
        ),

        "Strong": fuzz.interp_membership(
            rssi,
            rssi_strong,
            input_rssi
        )
    }


    snr_mu = {
        "Low": fuzz.interp_membership(
            snr,
            snr_low,
            input_snr
        ),

        "Moderate": fuzz.interp_membership(
            snr,
            snr_moderate,
            input_snr
        ),

        "High": fuzz.interp_membership(
            snr,
            snr_high,
            input_snr
        )
    }


    # --------------------------------------------------------
    # AoA FUZZIFICATION
    # --------------------------------------------------------

    input_aoa_abs = abs(input_aoa)

    aoa_mu = {
        "Aligned": fuzz.interp_membership(
            aoa,
            aoa_aligned,
            input_aoa_abs
        ),

        "Oblique": fuzz.interp_membership(
            aoa,
            aoa_oblique,
            input_aoa_abs
        ),

        "Edge": fuzz.interp_membership(
            aoa,
            aoa_edge,
            input_aoa_abs
        )
    }


    # --------------------------------------------------------
    # RULE FIRING
    # --------------------------------------------------------

    rule_strengths = []
    fired_rules = []

    for i, rule in enumerate(rules, start=1):

        memberships = []

        if rule["rssi"] is not None:
            memberships.append(
                rssi_mu[rule["rssi"]]
            )

        if rule["snr"] is not None:
            memberships.append(
                snr_mu[rule["snr"]]
            )

        if rule["aoa"] is not None:
            memberships.append(
                aoa_mu[rule["aoa"]]
            )

        # AND operation = minimum
        strength = min(memberships) if memberships else 0.0

        rule_strengths.append(strength)

        if strength > 1e-6:

            fired_rules.append({
                "Rule": f"R{i}",
                "Firing Strength": round(strength, 3),
                "Output": rule["output"]
            })


    # --------------------------------------------------------
    # MAX AGGREGATION
    # --------------------------------------------------------

    aggregated = {
        "Narrow": 0.0,
        "Medium": 0.0,
        "Wide": 0.0
    }

    for rule, strength in zip(
        rules,
        rule_strengths
    ):

        output = rule["output"]

        aggregated[output] = max(
            aggregated[output],
            strength
        )


    # --------------------------------------------------------
    # WEIGHTED-AVERAGE DEFUZZIFICATION
    # --------------------------------------------------------

    numerator = sum(
        aggregated[output] *
        output_peaks[output]
        for output in aggregated
    )

    denominator = sum(
        aggregated.values()
    )

    if denominator > 0:

        beam_solid_angle = (
            numerator / denominator
        )

    else:

        beam_solid_angle = np.nan


    # --------------------------------------------------------
    # DOMINANT OUTPUT
    # --------------------------------------------------------

    dominant_output = max(
        aggregated,
        key=aggregated.get
    )


    return (
        aggregated,
        beam_solid_angle,
        dominant_output,
        fired_rules,
        rssi_mu,
        snr_mu,
        aoa_mu
    )


# ============================================================
# INPUT SECTION
# ============================================================

st.markdown(
    '<div class="section-title">Input Parameters</div>',
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)


# ------------------------------------------------------------
# RSSI
# ------------------------------------------------------------

with col1:

    input_rssi = st.slider(
        "RSSI (dBm)",
        min_value=-100.0,
        max_value=-30.0,
        value=-75.0,
        step=0.1
    )

    st.caption(
        "Range: -100 to -30 dBm"
    )


# ------------------------------------------------------------
# SNR
# ------------------------------------------------------------

with col2:

    input_snr = st.slider(
        "SNR (dB)",
        min_value=0.0,
        max_value=30.0,
        value=12.0,
        step=0.1
    )

    st.caption(
        "Range: 0 to 30 dB"
    )


# ------------------------------------------------------------
# AoA
# ------------------------------------------------------------

with col3:

    input_aoa = st.slider(
        "AoA (°)",
        min_value=0.0,
        max_value=90.0,
        value=20.0,
        step=0.1
    )

    st.caption(
        "Range: 0 to 90°"
    )


# ============================================================
# PREDICT BUTTON
# ============================================================

st.markdown("")

predict_button = st.button(
    "🔍 Predict Beam",
    use_container_width=True
)


# ============================================================
# PREDICTION
# ============================================================

if predict_button:

    (
        aggregated,
        beam_solid_angle,
        dominant_output,
        fired_rules,
        rssi_mu,
        snr_mu,
        aoa_mu
    ) = predict_beam(
        input_rssi,
        input_snr,
        input_aoa
    )


    # ========================================================
    # MAIN RESULT
    # ========================================================

    st.markdown(
        '<div class="section-title">Prediction Result</div>',
        unsafe_allow_html=True
    )

    result_col1, result_col2 = st.columns(2)


    # ========================================================
    # RECOMMENDED BEAM
    # ========================================================

    with result_col1:

        result_html = (
            '<div class="result-box">'
            '<span class="result-label">Recommended Beam</span>'
            f'<span class="beam-output">{dominant_output}</span>'
            '</div>'
        )

        st.markdown(
            result_html,
            unsafe_allow_html=True
        )


    # ========================================================
    # BEAM SOLID ANGLE
    # ========================================================

    with result_col2:

        angle_html = (
            '<div class="result-box">'
            '<span class="result-label">Beam Solid Angle</span>'
            f'<span class="angle-output">{beam_solid_angle:.3f} sr</span>'
            '</div>'
        )

        st.markdown(
            angle_html,
            unsafe_allow_html=True
        )


    # ========================================================
    # INPUT SUMMARY
    # ========================================================

    st.markdown(
        '<div class="section-title">Input Summary</div>',
        unsafe_allow_html=True
    )

    input_data = pd.DataFrame({
        "Parameter": [
            "RSSI",
            "SNR",
            "AoA"
        ],

        "Value": [
            f"{input_rssi:.1f} dBm",
            f"{input_snr:.1f} dB",
            f"{input_aoa:.1f}°"
        ]
    })

    st.table(input_data)


    # ========================================================
    # OUTPUT MEMBERSHIP STRENGTHS
    # ========================================================

    st.markdown(
        '<div class="section-title">Output Membership Strengths</div>',
        unsafe_allow_html=True
    )

    output_df = pd.DataFrame({
        "Beam Type": [
            "Narrow",
            "Medium",
            "Wide"
        ],

        "Membership Strength": [
            aggregated["Narrow"],
            aggregated["Medium"],
            aggregated["Wide"]
        ],

        "Peak Value (sr)": [
            0.2,
            1.1,
            2.0
        ]
    })

    output_df["Membership Strength"] = (
        output_df["Membership Strength"].round(3)
    )

    st.dataframe(
        output_df,
        use_container_width=True,
        hide_index=True
    )


    # ========================================================
    # MEMBERSHIP BAR CHART
    # ========================================================

    chart_df = pd.DataFrame({
        "Beam Type": [
            "Narrow",
            "Medium",
            "Wide"
        ],

        "Strength": [
            aggregated["Narrow"],
            aggregated["Medium"],
            aggregated["Wide"]
        ]
    })

    st.bar_chart(
        chart_df.set_index("Beam Type")
    )


    # ========================================================
    # FUZZIFICATION DETAILS
    # ========================================================

    with st.expander("📊 View Fuzzification Details"):

        fuzz_col1, fuzz_col2, fuzz_col3 = st.columns(3)


        # ----------------------------------------------------
        # RSSI MEMBERSHIP
        # ----------------------------------------------------

        with fuzz_col1:

            st.write("**RSSI Membership**")

            rssi_df = pd.DataFrame({
                "Category": [
                    "Weak",
                    "Moderate",
                    "Strong"
                ],

                "Membership": [
                    rssi_mu["Weak"],
                    rssi_mu["Moderate"],
                    rssi_mu["Strong"]
                ]
            })

            rssi_df["Membership"] = (
                rssi_df["Membership"].round(3)
            )

            st.dataframe(
                rssi_df,
                hide_index=True,
                use_container_width=True
            )


        # ----------------------------------------------------
        # SNR MEMBERSHIP
        # ----------------------------------------------------

        with fuzz_col2:

            st.write("**SNR Membership**")

            snr_df = pd.DataFrame({
                "Category": [
                    "Low",
                    "Moderate",
                    "High"
                ],

                "Membership": [
                    snr_mu["Low"],
                    snr_mu["Moderate"],
                    snr_mu["High"]
                ]
            })

            snr_df["Membership"] = (
                snr_df["Membership"].round(3)
            )

            st.dataframe(
                snr_df,
                hide_index=True,
                use_container_width=True
            )


        # ----------------------------------------------------
        # AoA MEMBERSHIP
        # ----------------------------------------------------

        with fuzz_col3:

            st.write("**AoA Membership**")

            aoa_df = pd.DataFrame({
                "Category": [
                    "Aligned",
                    "Oblique",
                    "Edge"
                ],

                "Membership": [
                    aoa_mu["Aligned"],
                    aoa_mu["Oblique"],
                    aoa_mu["Edge"]
                ]
            })

            aoa_df["Membership"] = (
                aoa_df["Membership"].round(3)
            )

            st.dataframe(
                aoa_df,
                hide_index=True,
                use_container_width=True
            )


    # ========================================================
    # FIRED RULES
    # ========================================================

    st.markdown(
        '<div class="section-title">Fired Rules</div>',
        unsafe_allow_html=True
    )

    if fired_rules:

        fired_df = pd.DataFrame(
            fired_rules
        )

        st.dataframe(
            fired_df,
            use_container_width=True,
            hide_index=True
        )

    else:

        st.warning(
            "No fuzzy rules were fired for these inputs."
        )


    # ========================================================
    # FINAL MESSAGE
    # ========================================================

    st.success(
        f"Recommended beam: {dominant_output} | "
        f"Beam Solid Angle: {beam_solid_angle:.3f} sr"
    )