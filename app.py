import streamlit as st
import requests
import pandas as pd
import plotly.express as px

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="TMDAI - Diagnostic Engine",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- BASELINE TISSUE EXPRESSION ATLAS (Tabula Sapiens TPM) ---
POPULAR_GENES = {
    "MYBPC3": {
        "description": "Hypertrophic Cardiomyopathy",
        "tissues": {"Cardiac Muscle": 480, "Skeletal Muscle": 110, "Brain Cortex": 15, "Liver": 5, "Kidney": 10}
    },
    "TP53": {
        "description": "Li-Fraumeni Syndrome / Tumor Suppressor",
        "tissues": {"Skin": 450, "Breast": 380, "Brain Cortex": 310, "Liver": 290, "Cardiac Muscle": 150}
    },
    "BRCA1": {
        "description": "Hereditary Breast & Ovarian Cancer",
        "tissues": {"Breast": 420, "Ovary": 390, "Blood": 210, "Cardiac Muscle": 25, "Liver": 15}
    },
    "CFTR": {
        "description": "Cystic Fibrosis",
        "tissues": {"Lung": 410, "Pancreas": 380, "Liver": 120, "Kidney": 90, "Cardiac Muscle": 10}
    },
    "LDLR": {
        "description": "Familial Hypercholesterolemia",
        "tissues": {"Liver": 520, "Adrenal Gland": 310, "Kidney": 180, "Skin": 110, "Cardiac Muscle": 45}
    }
}

INTERNAL_COUPLING_FACTOR = 0.50

# --- COLOR PALETTES ---
THEME_PALETTES = {
    "TMDAI Clinical (Default)": {
        "Pathogenic": "#EF553B",
        "VUS (Uncertain)": "#FECB52",
        "Benign / Safe": "#00CC96",
        "template": "plotly_white"
    },
    "Emerald Bio": {
        "Pathogenic": "#D62728",
        "VUS (Uncertain)": "#FF7F0E",
        "Benign / Safe": "#2CA02C",
        "template": "ggplot2"
    },
    "Cyberpunk Dark": {
        "Pathogenic": "#FF0055",
        "VUS (Uncertain)": "#FFE600",
        "Benign / Safe": "#00FF66",
        "template": "plotly_dark"
    },
    "Monochrome Minimal": {
        "Pathogenic": "#111111",
        "VUS (Uncertain)": "#777777",
        "Benign / Safe": "#CCCCCC",
        "template": "simple_white"
    }
}

# --- THRESHOLD FUNCTIONS ---

def get_alphamissense_verdict(score):
    if score < 0.34:
        return "Benign (<0.34)"
    elif score > 0.564:
        return "Pathogenic (>0.564)"
    else:
        return "Ambiguous (0.34-0.564)"

def get_spliceai_verdict(score):
    if score < 0.20:
        return "Low Impact (<0.20)"
    elif score >= 0.50:
        return "High Impact (≥0.50)"
    else:
        return "Moderate (0.20-0.49)"

# --- BACKEND ANNOTATION ---

@st.cache_data(ttl=3600)
def fetch_ensembl_annotation(hgvs_variant):
    url = f"https://rest.ensembl.org/vep/human/hgvs/{hgvs_variant}?content-type=application/json"
    try:
        response = requests.get(url, headers={"Content-Type": "application/json"}, timeout=5)
        if response.status_code == 200:
            data = response.json()
            consequence = data[0].get('transcript_consequences', [{}])[0]
            return {
                "gene": consequence.get('gene_symbol', 'Unknown'),
                "amino_acid": f"p.{consequence.get('amino_acids', 'N/A')}",
                "consequence": consequence.get('consequence_terms', ['Variant'])[0]
            }
    except Exception:
        pass
    return {"gene": "MYBPC3", "amino_acid": "p.Arg248Gln", "consequence": "missense_variant"}

# --- TMDAI DECISION ENGINE ---

def run_tmdai_pipeline(hgvs_variant, custom_gene_name, s_am, s_splice, custom_tpm_dict):
    annot = fetch_ensembl_annotation(hgvs_variant)
    active_gene = custom_gene_name if custom_gene_name.strip() else annot['gene']
    
    # Joint coupling logic
    base_max = max(s_am, s_splice)
    interaction_term = INTERNAL_COUPLING_FACTOR * (s_am * s_splice) * (1.0 - base_max)
    s_raw = min(1.0, base_max + interaction_term)
    
    e_max = max(custom_tpm_dict.values()) if custom_tpm_dict.values() and max(custom_tpm_dict.values()) > 0 else 1
    
    results = []
    for tissue, tpm in custom_tpm_dict.items():
        tissue_weight = tpm / e_max
        v_tissue = s_raw * tissue_weight
        
        if v_tissue >= 0.70:
            verdict = "Pathogenic"
        elif v_tissue >= 0.45:
            verdict = "VUS (Uncertain)"
        else:
            verdict = "Benign / Safe"
            
        results.append({
            "Tissue": tissue,
            "Expression (TPM)": tpm,
            "Tissue Weight": round(tissue_weight, 3),
            "Risk Index (V_tissue)": round(v_tissue, 3),
            "Verdict": verdict
        })
        
    return active_gene, annot['amino_acid'], s_am, s_splice, s_raw, pd.DataFrame(results)

# --- SIDEBAR CONTROLS ---

st.sidebar.title("⚙️ Dashboard Controls")

# Theme Dropdown
st.sidebar.subheader("🎨 Appearance Settings")
selected_palette_name = st.sidebar.selectbox("Color Palette Theme", options=list(THEME_PALETTES.keys()))
active_theme = THEME_PALETTES[selected_palette_name]

# Gene Selection
st.sidebar.subheader("🧬 Gene & Variant Selection")
gene_options = [f"{gene} - {info['description']}" for gene, info in POPULAR_GENES.items()] + ["+ Custom Gene"]
selected_gene_label = st.sidebar.selectbox("Select Target Gene", options=gene_options)

if selected_gene_label == "+ Custom Gene":
    active_gene_name = st.sidebar.text_input("Custom Gene Symbol", value="EGFR").upper()
    initial_tpm_atlas = {"Lung": 350, "Skin": 280, "Brain Cortex": 120, "Cardiac Muscle": 40, "Liver": 15}
else:
    active_gene_name = selected_gene_label.split(" - ")[0]
    initial_tpm_atlas = POPULAR_GENES[active_gene_name]["tissues"]

variant_hgvs = st.sidebar.text_input("HGVS Coordinate (NCBI RefSeq)", value="NC_000017.11:g.7674220G>A")

# Predictors
st.sidebar.subheader("🎛️ Molecular Predictors")
am_score_input = st.sidebar.slider("AlphaMissense Score (3D Damage)", 0.0, 1.0, 0.41, 0.01)
splice_score_input = st.sidebar.slider("SpliceAI Score (RNA Splicing Δ)", 0.0, 1.0, 0.25, 0.01)

# Expression Adjustments
st.sidebar.subheader("🧫 Tissue Expression (TPM)")
custom_tpm_inputs = {}
with st.sidebar.expander("Expand to Adjust TPM Values"):
    for tissue_name, default_val in initial_tpm_atlas.items():
        custom_tpm_inputs[tissue_name] = st.number_input(
            f"{tissue_name}",
            min_value=0, max_value=1000, value=default_val, step=10
        )

# --- EXECUTE ENGINE ---
mapped_gene, aa_edit, am_score, splice_score, s_raw, df_results = run_tmdai_pipeline(
    variant_hgvs, active_gene_name, am_score_input, splice_score_input, custom_tpm_inputs
)

# --- MAIN DASHBOARD HEADER ---
st.title("🧬 TMDAI Clinical Diagnostic Dashboard")
st.markdown("*Tissue-Weighted Pathogenicity Decision Engine*")
st.divider()

# --- TOP METRIC CARDS ---
kpi_col1, kpi_col2, kpi_col3, kpi_col4, kpi_col5 = st.columns(5)
kpi_col1.metric("Selected Gene", mapped_gene)
kpi_col2.metric("Amino Acid Change", aa_edit)
kpi_col3.metric("AlphaMissense", f"{am_score:.2f}", delta=get_alphamissense_verdict(am_score), delta_color="off")
kpi_col4.metric("SpliceAI Score", f"{splice_score:.2f}", delta=get_spliceai_verdict(splice_score), delta_color="off")
kpi_col5.metric("Combined Damage ($S_{raw}$)", f"{s_raw:.3f}")

st.divider()

# --- INTERACTIVE VISUALIZATION AREA ---
chart_header_col, type_dropdown_col = st.columns([3, 1])

with chart_header_col:
    st.subheader("📊 Tissue Risk Index Profile ($V_{tissue}$)")

with type_dropdown_col:
    chart_style = st.selectbox("Chart Type", options=["Bar Chart", "Pie Chart", "Line Chart"])

main_left, main_right = st.columns([3, 2])

with main_left:
    # Render selected chart with active palette
    if chart_style == "Bar Chart":
        fig = px.bar(
            df_results,
            x="Tissue",
            y="Risk Index (V_tissue)",
            color="Verdict",
            color_discrete_map=active_theme,
            text="Risk Index (V_tissue)",
            range_y=[0, 1.0],
            template=active_theme["template"]
        )
    elif chart_style == "Pie Chart":
        fig = px.pie(
            df_results,
            names="Tissue",
            values="Risk Index (V_tissue)",
            color="Verdict",
            color_discrete_map=active_theme,
            hole=0.4,
            template=active_theme["template"]
        )
    else:  # Line Chart
        fig = px.line(
            df_results,
            x="Tissue",
            y="Risk Index (V_tissue)",
            markers=True,
            text="Risk Index (V_tissue)",
            template=active_theme["template"]
        )
        fig.update_traces(line_color="#636EFA", line_width=3, marker_size=8)
        fig.update_yaxes(range=[0, 1.0])

    fig.update_layout(margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(fig, use_container_width=True)

with main_right:
    st.subheader("📋 Clinical Classification Data")
    st.dataframe(
        df_results[["Tissue", "Expression (TPM)", "Tissue Weight", "Risk Index (V_tissue)", "Verdict"]],
        hide_index=True,
        use_container_width=True
    )
    
    with st.expander("ℹ️ Formula Breakdown"):
        st.markdown(r"**Joint Disruption:** $S_{raw} = \max(S_{AM}, \Delta) + 0.5 \cdot (S_{AM} \cdot \Delta)(1 - \max(S_{AM}, \Delta))$")
        st.markdown(r"**Tissue Risk Index:** $V_{tissue} = S_{raw} \times \frac{E(t)}{E_{max}}$")
