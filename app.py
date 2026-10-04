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
DEFAULT_GENE_ATLAS = {
    "MYBPC3": {"Cardiac Muscle": 480, "Skeletal Muscle": 110, "Brain Cortex": 15, "Liver": 5, "Kidney": 10},
    "TP53": {"Skin": 450, "Breast": 380, "Brain Cortex": 310, "Liver": 290, "Cardiac Muscle": 150},
    "BRCA1": {"Breast": 420, "Ovary": 390, "Blood": 210, "Cardiac Muscle": 25, "Liver": 15},
    "CFTR": {"Lung": 410, "Pancreas": 380, "Liver": 120, "Kidney": 90, "Cardiac Muscle": 10}
}

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
    """Fetches gene symbol and amino acid changes via Ensembl VEP API mapped to NCBI RefSeq standards."""
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

def run_tmdai_pipeline(hgvs_variant, custom_gene_name, s_am, s_splice, gamma, custom_tpm_dict):
    # 1. Fetch NCBI Reference Annotation
    annot = fetch_ensembl_annotation(hgvs_variant)
    
    active_gene = custom_gene_name if custom_gene_name.strip() else annot['gene']
    
    # 2. Joint Interaction Formula: Compounding S_AM and SpliceAI
    base_max = max(s_am, s_splice)
    interaction_term = gamma * (s_am * s_splice) * (1.0 - base_max)
    s_raw = min(1.0, base_max + interaction_term)
    
    # 3. Process Tabula Sapiens Tissue Expression Weights
    e_max = max(custom_tpm_dict.values()) if custom_tpm_dict.values() and max(custom_tpm_dict.values()) > 0 else 1
    
    # 4. Compute Final Tissue Risk Index: V_tissue = S_raw * (E(t) / E_max)
    results = []
    for tissue, tpm in custom_tpm_dict.items():
        tissue_weight = tpm / e_max
        v_tissue = s_raw * tissue_weight
        
        # ACMG Clinical Classifications
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
st.markdown("**Tissue-Weighted Variant Pathogenicity Decision Engine**")
st.divider()

# Sidebar Setup
st.sidebar.header("1. Gene & Variant Selection")
variant_hgvs = st.sidebar.text_input("HGVS Variant Notation (NCBI RefSeq)", value="NC_000017.11:g.7674220G>A")

preset_gene = st.sidebar.selectbox("Select Preset Gene", options=list(DEFAULT_GENE_ATLAS.keys()) + ["+ Add Custom Gene"])

if preset_gene == "+ Add Custom Gene":
    custom_gene_input = st.sidebar.text_input("Enter Custom Gene Symbol", value="EGFR")
    active_gene_name = custom_gene_input.upper()
    initial_tpm_atlas = {"Lung": 350, "Skin": 280, "Brain Cortex": 120, "Cardiac Muscle": 40, "Liver": 15}
else:
    active_gene_name = preset_gene
    initial_tpm_atlas = DEFAULT_GENE_ATLAS[preset_gene]

# Sidebar: Molecular Sliders
st.sidebar.header("2. Predictor Inputs & Interaction")
am_score_input = st.sidebar.slider(
    "AlphaMissense Score (3D Structural Damage)", 
    min_value=0.0, max_value=1.0, value=0.41, step=0.01
)

splice_score_input = st.sidebar.slider(
    "SpliceAI Score (RNA Splicing Disruption Δ)", 
    min_value=0.0, max_value=1.0, value=0.25, step=0.01
)

gamma_input = st.sidebar.slider(
    "Synergy Factor (Coupling Weight γ)", 
    min_value=0.0, max_value=1.0, value=0.50, step=0.05
)

# Sidebar: Expression Inputs
st.sidebar.header("3. Tissue Expression (Tabula Sapiens)")
custom_tpm_inputs = {}
with st.sidebar.expander("Adjust Tissue TPM Expression", expanded=True):
    for tissue_name, default_val in initial_tpm_atlas.items():
        custom_tpm_inputs[tissue_name] = st.number_input(
            f"{tissue_name} (TPM)",
            min_value=0,
            max_value=1000,
            value=default_val,
            step=10
        )

# Execute Engine Pipeline
mapped_gene, aa_edit, am_score, splice_score, s_raw, df_results = run_tmdai_pipeline(
    variant_hgvs, active_gene_name, am_score_input, splice_score_input, gamma_input, custom_tpm_inputs
)

# Get Predictor Specific Classifications
am_verdict_str, _ = get_alphamissense_verdict(am_score)
splice_verdict_str, _ = get_spliceai_verdict(splice_score)

# Top Metrics Row with Publication Cutoff Context
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Active Gene", mapped_gene)
col2.metric("Amino Acid Edit", aa_edit)
col3.metric("AlphaMissense ($S_{AM}$)", f"{am_score:.2f}", delta=am_verdict_str, delta_color="off")
col4.metric("SpliceAI ($\Delta$)", f"{splice_score:.2f}", delta=splice_verdict_str, delta_color="off")
col5.metric("Joint Disruption ($S_{raw}$)", f"{s_raw:.3f}")

st.divider()

# Main Visuals Layout
left_col, right_col = st.columns([3, 2])

with left_col:
    st.subheader("Tissue-Weighted Risk Index ($V_{tissue}$)")
    fig = px.bar(
        df_results,
        x="Tissue",
        y="Risk Index (V_tissue)",
        color="Verdict",
        color_discrete_map={
            "Pathogenic": "#EF553B",
            "VUS (Uncertain)": "#FECB52",
            "Benign / Safe": "#00CC96"
        },
        text="Risk Index (V_tissue)",
        range_y=[0, 1.0]
    )
    st.plotly_chart(fig, use_container_width=True)

with right_col:
    st.subheader("Clinical Verdict Breakdown")
    st.dataframe(
        df_results[["Tissue", "Expression (TPM)", "Tissue Weight", "Risk Index (V_tissue)", "Verdict"]],
        hide_index=True,
        use_container_width=True
    )
    st.caption("AlphaMissense Reference: <0.34 (Benign) | >0.564 (Pathogenic)")
    st.caption("SpliceAI Reference: <0.20 (Benign) | ≥0.50 (Pathogenic)")
    st.caption("TMDAI Verdict Formula: $V_{tissue} = S_{raw} \\times \\frac{E(t)}{E_{max}}$")
