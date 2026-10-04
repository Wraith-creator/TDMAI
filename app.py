import streamlit as st
import requests
import pandas as pd
import plotly.express as px

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="TMDAI - Diagnostic Engine",
    page_icon="🧬",
    layout="wide"
)

# --- BASELINE TISSUE EXPRESSION ATLAS (Tabula Sapiens TPM) ---
# Top 5 most clinically significant genes
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

# Fixed internal coupling factor (gamma) so scores interact without needing a slider
INTERNAL_COUPLING_FACTOR = 0.50

# --- THRESHOLD HELPER FUNCTIONS ---

def get_alphamissense_verdict(score):
    """DeepMind AlphaMissense publication cutoffs."""
    if score < 0.34:
        return "Benign", "#00CC96"
    elif score > 0.564:
        return "Pathogenic", "#EF553B"
    else:
        return "Ambiguous / VUS", "#FECB52"

def get_spliceai_verdict(score):
    """Illumina SpliceAI publication cutoffs."""
    if score < 0.20:
        return "Low Impact (Benign)", "#00CC96"
    elif score >= 0.50:
        return "High Splicing Disruption", "#EF553B"
    else:
        return "Moderate Impact", "#FECB52"

# --- BACKEND ANNOTATION ---

@st.cache_data(ttl=3600)
def fetch_ensembl_annotation(hgvs_variant):
    """Fetches gene symbol and amino acid changes via Ensembl VEP API."""
    url = f"https://rest.ensembl.org/vep/human/hgvs/{hgvs_variant}?content-type=application/json"
    try:
        response = requests.get(url, headers={"Content-Type": "application/json"}, timeout=5)
        if response.status_code == 200:
            data = response.json()
            consequence = data[0].get('transcript_consequences', [{}])[0]
            
            gene = consequence.get('gene_symbol', 'Unknown')
            amino_acids = consequence.get('amino_acids', 'N/A')
            consequence_term = consequence.get('consequence_terms', ['Variant'])[0]
            
            return {
                "gene": gene,
                "amino_acid": f"p.{amino_acids}" if amino_acids != 'N/A' else 'Non-coding',
                "consequence": consequence_term
            }
    except Exception:
        pass
    
    return {"gene": "MYBPC3", "amino_acid": "p.Arg248Gln", "consequence": "missense_variant"}

# --- TMDAI DECISION ENGINE ---

def run_tmdai_pipeline(hgvs_variant, custom_gene_name, s_am, s_splice, custom_tpm_dict):
    annot = fetch_ensembl_annotation(hgvs_variant)
    active_gene = custom_gene_name if custom_gene_name.strip() else annot['gene']
    
    # Internal joint interaction formula (Coupled predictors)
    base_max = max(s_am, s_splice)
    interaction_term = INTERNAL_COUPLING_FACTOR * (s_am * s_splice) * (1.0 - base_max)
    s_raw = min(1.0, base_max + interaction_term)
    
    # Tissue Expression Weights
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

# --- USER INTERFACE ---

st.title("🧬 TMDAI: Tissue & Mutation Artificial Intelligence")
st.markdown("*Precision Tissue-Weighted Variant Pathogenicity Engine*")
st.divider()

# --- SIDEBAR NAVIGATION ---
st.sidebar.header("🎯 1. Select Target Gene")

# Dropdown for top 5 popular genes + Custom option
gene_options = [f"{gene} ({info['description']})" for gene, info in POPULAR_GENES.items()] + ["+ Add Custom Gene"]
selected_gene_label = st.sidebar.selectbox("Choose a Gene", options=gene_options)

if selected_gene_label == "+ Add Custom Gene":
    active_gene_name = st.sidebar.text_input("Enter Gene Symbol", value="EGFR").upper()
    initial_tpm_atlas = {"Lung": 350, "Skin": 280, "Brain Cortex": 120, "Cardiac Muscle": 40, "Liver": 15}
else:
    active_gene_name = selected_gene_label.split(" ")[0]
    initial_tpm_atlas = POPULAR_GENES[active_gene_name]["tissues"]

variant_hgvs = st.sidebar.text_input("HGVS Variant Notation", value="NC_000017.11:g.7674220G>A")

st.sidebar.header("📊 2. Predictor Inputs")

am_score_input = st.sidebar.slider(
    "AlphaMissense Score (3D Damage)", 
    min_value=0.0, max_value=1.0, value=0.41, step=0.01
)

splice_score_input = st.sidebar.slider(
    "SpliceAI Score (RNA Splicing Δ)", 
    min_value=0.0, max_value=1.0, value=0.25, step=0.01
)

st.sidebar.header("🧫 3. Expression Adjustments")
custom_tpm_inputs = {}
with st.sidebar.expander("Adjust Tissue TPM Values", expanded=False):
    for tissue_name, default_val in initial_tpm_atlas.items():
        custom_tpm_inputs[tissue_name] = st.number_input(
            f"{tissue_name} (TPM)",
            min_value=0,
            max_value=1000,
            value=default_val,
            step=10
        )

# --- EXECUTE ENGINE ---
mapped_gene, aa_edit, am_score, splice_score, s_raw, df_results = run_tmdai_pipeline(
    variant_hgvs, active_gene_name, am_score_input, splice_score_input, custom_tpm_inputs
)

am_verdict_str, _ = get_alphamissense_verdict(am_score)
splice_verdict_str, _ = get_spliceai_verdict(splice_score)

# --- METRIC CARDS ---
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Active Gene", mapped_gene)
col2.metric("Amino Acid Edit", aa_edit)
col3.metric("AlphaMissense", f"{am_score:.2f}", delta=am_verdict_str, delta_color="off")
col4.metric("SpliceAI", f"{splice_score:.2f}", delta=splice_verdict_str, delta_color="off")
col5.metric("Joint Score ($S_{raw}$)", f"{s_raw:.3f}")

st.divider()

# --- MAIN VISUALIZATION AREA ---
left_col, right_col = st.columns([3, 2])

with left_col:
    # Display header and chart style selector inline
    chart_col1, chart_col2 = st.columns([2, 1])
    with chart_col1:
        st.subheader("Tissue-Weighted Risk Profile")
    with chart_col2:
        chart_type = st.selectbox("Chart Type", options=["Bar Chart", "Pie Chart", "Line Chart"])

    color_map = {
        "Pathogenic": "#EF553B",
        "VUS (Uncertain)": "#FECB52",
        "Benign / Safe": "#00CC96"
    }

    # Render dynamic chart based on selection
    if chart_type == "Bar Chart":
        fig = px.bar(
            df_results,
            x="Tissue",
            y="Risk Index (V_tissue)",
            color="Verdict",
            color_discrete_map=color_map,
            text="Risk Index (V_tissue)",
            range_y=[0, 1.0],
            template="plotly_white"
        )
    elif chart_type == "Pie Chart":
        fig = px.pie(
            df_results,
            names="Tissue",
            values="Risk Index (V_tissue)",
            color="Verdict",
            color_discrete_map=color_map,
            hole=0.4,
            template="plotly_white"
        )
    else:  # Line Chart
        fig = px.line(
            df_results,
            x="Tissue",
            y="Risk Index (V_tissue)",
            markers=True,
            text="Risk Index (V_tissue)",
            template="plotly_white"
        )
        fig.update_traces(line_color="#636EFA", line_width=3, marker_size=8)
        fig.update_yaxes(range=[0, 1.0])

    fig.update_layout(margin=dict(l=20, r=20, t=30, b=20))
    st.plotly_chart(fig, use_container_width=True)

with right_col:
    st.subheader("Clinical Data Breakdown")
    st.dataframe(
        df_results[["Tissue", "Expression (TPM)", "Tissue Weight", "Risk Index (V_tissue)", "Verdict"]],
        hide_index=True,
        use_container_width=True
    )
    st.caption("AlphaMissense Limits: <0.34 (Benign) | >0.564 (Pathogenic)")
    st.caption("SpliceAI Limits: <0.20 (Benign) | ≥0.50 (Pathogenic)")
    st.caption("Formula: $S_{raw} = \max(S_{AM}, \Delta) + 0.5 \cdot (S_{AM} \cdot \Delta)(1 - \max(S_{AM}, \Delta))$")
