# app.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy import stats
from statsmodels.stats.multitest import multipletests
from sklearn.utils import resample
import itertools
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.colors import to_rgba
import matplotlib.patches as mpatches
import textwrap
from matplotlib.gridspec import GridSpec
from matplotlib.offsetbox import DrawingArea, TextArea, HPacker, VPacker, AnnotationBbox
import seaborn as sns
import matplotlib

# Prevent matplotlib from trying to use any Xwindows backend.
matplotlib.use('Agg')

# -------------------------------
# Custom CSS for Outlines
# -------------------------------
def add_css():
    st.markdown(
        """
        <style>
        .section {
            border: 1px solid #e0e0e0;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
            background-color: #f9f9f9;
        }
        .section-title {
            font-size: 24px;
            font-weight: bold;
            margin-bottom: 10px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# -------------------------------
# Data Processing Functions
# -------------------------------
def bootstrap_correlations(df, n_iterations=500, method='pearson', progress_bar=None, status_text=None, start_progress=0.0, end_progress=1.0):
    correlations = []
    for i in range(n_iterations):
        df_resampled = resample(df)
        corr_matrix = df_resampled.corr(method=method)
        correlations.append(corr_matrix)
        if progress_bar and status_text:
            # Calculate incremental progress
            progress = start_progress + (i + 1) / n_iterations * (end_progress - start_progress)
            progress_bar.progress(int(progress * 100))
            status_text.text(f"Bootstrapping {method.capitalize()} Correlations... ({i+1}/{n_iterations})")
    median_corr = pd.concat(correlations).groupby(level=0).median()
    return median_corr

def calculate_p_values(df, method='pearson'):
    p_values = pd.DataFrame(np.ones((df.shape[1], df.shape[1])), columns=df.columns, index=df.columns)
    for col1, col2 in itertools.combinations(df.columns, 2):
        try:
            _, p_val = stats.pearsonr(df[col1], df[col2])
            p_values.at[col1, col2] = p_val
            p_values.at[col2, col1] = p_val
        except Exception:
            p_values.at[col1, col2] = 1
            p_values.at[col2, col1] = 1
    return p_values

def correct_p_values(p_values):
    _, corrected, _, _ = multipletests(p_values.values.flatten(), alpha=0.05, method='fdr_bh')
    corrected_p = pd.DataFrame(corrected.reshape(p_values.shape), index=p_values.index, columns=p_values.columns)
    return corrected_p

def find_common_parameters(dataframes):
    """
    Identify parameters (columns) that are common across multiple DataFrames.
    """
    if not dataframes:
        return []
    common_columns = set(dataframes[0].columns)
    for df in dataframes[1:]:
        common_columns &= set(df.columns)
    common_columns.discard('date')
    return list(common_columns)

def remove_outliers_zscore(df, threshold=3):
    """
    Remove outliers from a DataFrame using the Z-score method.
    """
    st.write("Applying Z-Score Method to filter outliers...")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    z_scores = np.abs(stats.zscore(df[numeric_cols], nan_policy="omit"))
    mask = (z_scores < threshold).all(axis=1)
    filtered_df = df[mask]
    st.write(f"Outliers removed: {len(df) - len(filtered_df)}")
    return filtered_df

def validate_correlation_matrix(df, n_iterations=500, alpha=0.05, progress_bar=None, status_text=None, start_progress=0.0, end_progress=1.0):
    st.write(f"DataFrame shape: {df.shape}")
    st.write("Bootstrapping correlation matrices...")

    # Ensure all columns are numeric
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.dropna(axis=1, how='all')  # Drop columns that are entirely non-numeric or NaN

    # Bootstrap Pearson correlations
    pearson_corr = bootstrap_correlations(
        df, n_iterations=n_iterations, method='pearson',
        progress_bar=progress_bar, status_text=status_text,
        start_progress=start_progress, end_progress=start_progress + (end_progress - start_progress) / 3
    )

    # Bootstrap Spearman correlations
    spearman_corr = bootstrap_correlations(
        df, n_iterations=n_iterations, method='spearman',
        progress_bar=progress_bar, status_text=status_text,
        start_progress=start_progress + (end_progress - start_progress) / 3,
        end_progress=start_progress + 2 * (end_progress - start_progress) / 3
    )

    # Bootstrap Kendall correlations
    kendall_corr = bootstrap_correlations(
        df, n_iterations=n_iterations, method='kendall',
        progress_bar=progress_bar, status_text=status_text,
        start_progress=start_progress + 2 * (end_progress - start_progress) / 3,
        end_progress=end_progress
    )

    # Average the correlation matrices
    avg_corr_matrix = (pearson_corr + spearman_corr + kendall_corr) / 3

    st.write("Calculating and correcting p-values...")
    p_values = calculate_p_values(df, method='pearson')
    corrected_p_values = correct_p_values(p_values)

    sig_mask = (corrected_p_values < alpha).astype(int)
    filtered_corr_matrix = avg_corr_matrix.where(sig_mask > 0).fillna(0)

    st.write("Correlation matrix validated and filtered based on significance.")
    return filtered_corr_matrix

# -------------------------------
# Visualization Functions
# -------------------------------
def generate_heatmap(df, title, labels, progress_bar, status_text, start_progress, end_progress):
    filtered_corr_matrix = validate_correlation_matrix(
        df,
        n_iterations=500,
        alpha=0.05,
        progress_bar=progress_bar,
        status_text=status_text,
        start_progress=start_progress,
        end_progress=end_progress
    )
    parameter_order = sorted(filtered_corr_matrix.index)
    filtered_corr_matrix = filtered_corr_matrix.loc[parameter_order, parameter_order]

    np.fill_diagonal(filtered_corr_matrix.values, 1)

    st.subheader(title)
    fig = px.imshow(
        filtered_corr_matrix,
        text_auto=".2f",
        color_continuous_scale="RdBu",
        zmin=-1,
        zmax=1,
        labels={"x": labels[0], "y": labels[1], "color": "Correlation Coefficient"},
        title=title,
    )

    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=20),
            x=0.5,               # Center horizontally
            xanchor='center',
            yanchor='top'
        ),
        xaxis=dict(tickangle=45, title=None, tickfont=dict(size=12)),
        yaxis=dict(title=None, tickfont=dict(size=12)),
        autosize=False,
        width=800,
        height=600,
        margin=dict(l=100, r=100, t=100, b=100),
    )

    st.plotly_chart(fig)
    return filtered_corr_matrix

def generate_network_diagram_streamlit(labels, correlation_matrices, parameters, globally_shared=True, progress_bar=None, status_text=None, start_progress=0.0, end_progress=1.0):
    """
    Generate a parameter-based network diagram.
    If globally_shared is True, use globally shared parameters.
    Otherwise, use locally shared parameters for each edge.
    """
    G = nx.MultiGraph()
    diagram_type = "Globally Shared" if globally_shared else "Locally Shared"

    st.subheader(f"{diagram_type} Network Diagram")

    # Collect data for edge summary boxes
    edge_summaries = []

    total_connections = len(labels) - 1
    for i in range(len(labels) - 1):
        st.write(f"Processing connection: {labels[i]} → {labels[i + 1]}")

        # Retrieve the filtered correlation matrix for this pair
        filtered_corr_matrix = correlation_matrices[i]

        # Track added edges to avoid duplicates
        added_edges = set()

        if globally_shared:
            parameters_to_use = parameters
        else:
            parameters_to_use = parameters[i]

        node1 = labels[i]
        node2 = labels[i + 1]

        edge_summary = {
            'nodes': (node1, node2),
            'parameters': []
        }

        for param in parameters_to_use:
            edge_key = (node1, node2, param)

            param1 = f"{param}_{node1}"
            param2 = f"{param}_{node2}"

            if param1 in filtered_corr_matrix.index and param2 in filtered_corr_matrix.columns:
                corr_value = filtered_corr_matrix.loc[param1, param2]

                if corr_value == 0 or edge_key in added_edges:
                    continue

                G.add_node(node1, label=node1)
                G.add_node(node2, label=node2)
                G.add_edge(
                    node1,
                    node2,
                    parameter=param,
                    correlation=corr_value,
                    weight=abs(corr_value),
                    key=param
                )
                added_edges.add(edge_key)
                edge_summary['parameters'].append((param, corr_value))

        if edge_summary['parameters']:
            edge_summaries.append(edge_summary)

        if progress_bar and status_text:
            progress = start_progress + (i + 1) / total_connections * (end_progress - start_progress)
            progress_bar.progress(int(progress * 100))
            status_text.text(f"Processing connection: {node1} → {node2}")

    if G.number_of_nodes() == 0:
        st.warning("No nodes to display in the network diagram.")
        return

    # Create a figure with GridSpec: 2 rows
    fig = plt.figure(figsize=(18, 18))
    gs = GridSpec(2, 1, height_ratios=[3, 1], hspace=0.3)

    ax_network = fig.add_subplot(gs[0, 0])

    if globally_shared:
        pos = nx.kamada_kawai_layout(G)
    else:
        pos = nx.spring_layout(G, k=0.15, iterations=200, seed=42)

    node_colors = ["lightblue"] * len(G.nodes())
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=8000, ax=ax_network)

    max_label_width = 10
    formatted_labels = {}
    for node in G.nodes():
        label_text = G.nodes[node]['label'].replace("_", " ")
        wrapped_label = "\n".join(textwrap.wrap(label_text, width=max_label_width))
        formatted_labels[node] = wrapped_label

    nx.draw_networkx_labels(G, pos, labels=formatted_labels, font_size=10, ax=ax_network)

    unique_parameters = list(set(d['parameter'] for _, _, k, d in G.edges(keys=True, data=True)))
    num_params = len(unique_parameters)
    base_colors = plt.cm.tab10.colors
    if num_params > len(base_colors):
        base_colors = plt.cm.tab20.colors
    parameter_colors = dict(zip(unique_parameters, base_colors[:num_params]))

    def adjust_color_intensity(base_color, corr_value):
        rgba = to_rgba(base_color)
        intensity = 1.0
        return (rgba[0], rgba[1], rgba[2], intensity)

    num_edges = len(G.edges(keys=True))
    curvature_values = np.linspace(-0.5, 0.5, num_edges)

    for idx, (u, v, key, d) in enumerate(G.edges(data=True, keys=True)):
        curvature = curvature_values[idx] if num_edges > 1 else 0.2
        corr_value = d['correlation']
        parameter = d['parameter']
        base_color = parameter_colors[parameter]
        edge_color = adjust_color_intensity(base_color, corr_value)
        style = 'solid' if corr_value >= 0 else 'dashed'

        nx.draw_networkx_edges(
            G, pos,
            edgelist=[(u, v, key)],
            connectionstyle=f"arc3,rad={curvature}",
            edge_color=[edge_color],
            width=d["weight"] * 5,
            style=style,
            ax=ax_network
        )

    ax_network.set_title(f"{diagram_type} Parameter-Based Network Diagram", fontsize=16, pad=20, weight="bold")

    section_boxes = []
    for summary in edge_summaries:
        node1, node2 = summary['nodes']
        process_pair_title = f"{node1} → {node2}"
        title_area = TextArea(process_pair_title, textprops=dict(color='black', size=12, weight='bold'))

        content_boxes = []
        for param, corr in summary['parameters']:
            color = parameter_colors[param]
            da = DrawingArea(20, 10, 0, 0)
            da.add_artist(mpatches.Rectangle((0, 0), 20, 10, fc=color, ec='black'))
            line_text = f"{param}: {corr:.2f}"
            ta = TextArea(line_text, textprops=dict(color='black', size=10))
            hbox = HPacker(children=[da, ta], align="center", pad=0, sep=5)
            content_boxes.append(hbox)
        section_box = VPacker(children=[title_area] + content_boxes, align="left", pad=0, sep=2)
        section_boxes.append(section_box)

    all_sections_box = VPacker(children=section_boxes, align="left", pad=0, sep=10)

    interpretation_text = (
        "Diagram Interpretation:\n"
        "• Nodes represent processes.\n"
        "• Edges represent significant correlations between parameters.\n"
        "• Edge colors correspond to parameters (see edge summaries below).\n"
        "• Solid lines: Positive correlations.\n"
        "• Dashed lines: Negative correlations.\n"
        "• Edge thickness reflects correlation strength."
    )
    interpretation_area = TextArea(interpretation_text, textprops=dict(fontsize=12))

    combined_box = VPacker(children=[all_sections_box, interpretation_area], align="left", pad=20, sep=20)

    ax_text = fig.add_subplot(gs[1, 0])
    ax_text.axis("off")

    ab = AnnotationBbox(
        combined_box, (0.5, 0.5), xycoords='axes fraction',
        box_alignment=(0.5, 0.5),
        bboxprops=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5', alpha=0.9)
    )
    ax_text.add_artist(ab)

    plt.tight_layout()
    st.pyplot(fig)

def plot_gspd_bar_chart(process_labels, globally_shared_parameters, correlation_matrices, progress_bar, status_text, progress_increment):
    # ... unchanged ...
    pass

def plot_gspd_line_graph(process_labels, globally_shared_parameters, correlation_matrices, progress_bar, status_text, progress_increment):
    # ... unchanged ...
    pass

def generate_targeted_network_diagram_streamlit(process_labels, dataframes, progress_bar, status_text, progress_increment, n_iterations=500, alpha=0.05):
    """
    Generate a targeted network diagram centered around a selected parameter 
    from a selected process. Allows independent adjustment of alpha.
    """
    st.write("### Targeted Network Diagram")

    selected_process_label = st.selectbox(
        "Select a Process:",
        options=process_labels,
        help="Choose the process you want to center the network diagram around."
    )
    process_choice = process_labels.index(selected_process_label)
    selected_dataframe = dataframes[process_choice]

    available_parameters = selected_dataframe.columns.drop('date', errors='ignore')
    selected_parameter = st.selectbox(
        "Select a Parameter:",
        options=available_parameters,
        help="Choose the parameter to center the network diagram around."
    )

    alpha = st.number_input(
        "Set Significance Level (alpha):",
        min_value=0.0001,
        max_value=1.0,
        value=0.05,
        step=0.005,
        help="Adjust the significance level for correlation filtering."
    )

    if st.button("Generate Targeted Network Diagram"):
        st.write(f"Generating network diagram for **{selected_parameter}** in **{selected_process_label}** with alpha={alpha}...")

        status_text.text("Preparing data for targeted network diagram...")
        progress_bar.progress(int((0.05 * progress_increment) * 100))

        # 1) Merge data
        combined_df = selected_dataframe[['date', selected_parameter]].copy()
        combined_df.columns = ['date', f"{selected_parameter}_{selected_process_label}"]

        df_same_process = selected_dataframe.drop(columns=[selected_parameter], errors='ignore')
        df_same_process.columns = [f"{col}_{selected_process_label}" if col != 'date' else 'date' for col in df_same_process.columns]
        combined_df = pd.merge(combined_df, df_same_process, on='date', how='inner')

        for idx, df in enumerate(dataframes):
            if idx != process_choice:
                process_label = process_labels[idx]
                df_temp = df.copy()
                df_temp.columns = [f"{col}_{process_label}" if col != 'date' else 'date' for col in df_temp.columns]
                combined_df = pd.merge(combined_df, df_temp, on='date', how='inner')

        # ---------------------------
        # DEBUG: Print final merged DF
        # ---------------------------
        st.write("**[DEBUG] Final Merged DataFrame for Targeted Diagram**")
        st.write("Shape:", combined_df.shape)               # <-- ADDED THIS
        st.dataframe(combined_df.head(10))                  # <-- ADDED THIS

        combined_df = combined_df.replace([np.inf, -np.inf], np.nan)
        combined_df = combined_df.dropna()
        numeric_columns = combined_df.select_dtypes(include=[np.number]).columns
        combined_df = combined_df[numeric_columns]

        # Outlier removal
        combined_df = remove_outliers_zscore(combined_df, threshold=3)
        progress_bar.progress(int((0.10 * progress_increment) * 100))

        # 2) Bootstrapping Correlations
        status_text.text("Bootstrapping correlations...")
        pearson_corr = bootstrap_correlations(
            combined_df,
            n_iterations=n_iterations,
            method='pearson',
            progress_bar=progress_bar,
            status_text=status_text,
            start_progress=0.0,
            end_progress=0.3 * progress_increment
        )
        spearman_corr = bootstrap_correlations(
            combined_df,
            n_iterations=n_iterations,
            method='spearman',
            progress_bar=progress_bar,
            status_text=status_text,
            start_progress=0.3 * progress_increment,
            end_progress=0.6 * progress_increment
        )
        kendall_corr = bootstrap_correlations(
            combined_df,
            n_iterations=n_iterations,
            method='kendall',
            progress_bar=progress_bar,
            status_text=status_text,
            start_progress=0.6 * progress_increment,
            end_progress=0.9 * progress_increment
        )

        avg_corr_matrix = (pearson_corr + spearman_corr + kendall_corr) / 3

        target_param_full = f"{selected_parameter}_{selected_process_label}"
        if target_param_full not in avg_corr_matrix.columns:
            st.error(f"The selected parameter '{selected_parameter}' is not available in the data.")
            return
        target_correlations = avg_corr_matrix[target_param_full].drop(target_param_full)

        # 3) Calculate p-values (Pearson) only for target vs. others
        status_text.text("Calculating and correcting p-values...")
        p_values = pd.Series(dtype=float)
        for col in target_correlations.index:
            if np.all(combined_df[target_param_full] == combined_df[col]):
                p_values[col] = 0.0
                continue
            try:
                _, p_val = stats.pearsonr(combined_df[target_param_full], combined_df[col])
                p_values[col] = p_val
            except Exception as e:
                st.error(f"Error calculating p-value between {target_param_full} and {col}: {e}")
                p_values[col] = 1.0
                continue

        # 4) Multiple testing correction (FDR)
        _, corrected_p_values, _, _ = multipletests(p_values.values, alpha=alpha, method='fdr_bh')
        significance_mask = corrected_p_values < alpha
        significant_correlations = target_correlations[significance_mask]
        significant_p_values = corrected_p_values[significance_mask]

        if significant_correlations.empty:
            st.warning("No significant correlations found with the selected alpha level.")
            progress_bar.progress(int((0.95 * progress_increment) * 100))
            status_text.text("No significant correlations found.")
            return

        # 5) Build final DF of correlations
        corr_data = pd.DataFrame({
            'Parameter': significant_correlations.index,
            'Correlation': significant_correlations.values,
            'P-value': significant_p_values
        })
        corr_data['Process'] = corr_data['Parameter'].apply(lambda x: x.rsplit('_', 1)[1])
        corr_data['Parameter Name'] = corr_data['Parameter'].apply(lambda x: x.rsplit('_', 1)[0])
        corr_data = corr_data.sort_values('Correlation', key=abs, ascending=False)

        # ---------------------------
        # Show table of significant correlations
        # ---------------------------
        st.write("### Table of Significant Correlations (Targeted)")  # <-- ADDED THIS
        st.dataframe(corr_data)                                       # <-- ADDED THIS

        # 6) Build the network diagram from the significant correlations
        # ... existing code below ...
        G = nx.Graph()
        G.add_node(target_param_full, label=selected_parameter, process=selected_process_label)

        internal_corr = corr_data[corr_data['Process'] == selected_process_label]
        external_corr = corr_data[corr_data['Process'] != selected_process_label]

        for idx, row in internal_corr.iterrows():
            G.add_node(row['Parameter'], label=row['Parameter Name'], process=row['Process'])
            G.add_edge(target_param_full, row['Parameter'],
                       correlation=row['Correlation'],
                       weight=abs(row['Correlation']))

        for idx, row in external_corr.iterrows():
            G.add_node(row['Parameter'], label=row['Parameter Name'], process=row['Process'])
            G.add_edge(target_param_full, row['Parameter'],
                       correlation=row['Correlation'],
                       weight=abs(row['Correlation']))

        pos = nx.spring_layout(G, seed=42)
        internal_nodes = [node for node in G.nodes if G.nodes[node]['process'] == selected_process_label and node != target_param_full]
        external_nodes = [node for node in G.nodes if G.nodes[node]['process'] != selected_process_label]
        target_pos = pos[target_param_full]
        for node in internal_nodes:
            pos[node][0] -= 0.5
        for node in external_nodes:
            pos[node][0] += 0.5

        fig, ax = plt.subplots(figsize=(14, 10))
        processes = list(set(nx.get_node_attributes(G, 'process').values()))
        color_map = {process: idx for idx, process in enumerate(processes)}
        cmap = plt.get_cmap('tab20')
        num_colors = len(processes)
        colors = [cmap(i / num_colors) for i in range(num_colors)]
        process_color_mapping = {process: colors[idx] for idx, process in enumerate(processes)}
        node_colors = [process_color_mapping[G.nodes[node]['process']] for node in G.nodes]

        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=3000, ax=ax)
        labels = {node: f"{G.nodes[node]['label']}\n({G.nodes[node]['process']})" for node in G.nodes}
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=10, ax=ax)

        edge_colors = ['green' if G.edges[edge]['correlation'] > 0 else 'red' for edge in G.edges]
        edge_weights = [G.edges[edge]['weight'] * 5 for edge in G.edges]
        edge_labels = {(u, v): f"{G.edges[(u, v)]['correlation']:.2f}" for u, v in G.edges}
        nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=edge_weights, ax=ax)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='blue', font_size=8, ax=ax)

        process_legend = [
            plt.Line2D([0], [0], marker='o', color='w', label=proc,
                       markerfacecolor=process_color_mapping[proc],
                       markersize=10) for proc in processes
        ]
        ax.legend(handles=process_legend, title='Processes', loc='upper left', bbox_to_anchor=(1, 1))

        green_line = plt.Line2D([], [], color='green', marker='_', linestyle='-', label='Positive Correlation')
        red_line = plt.Line2D([], [], color='red', marker='_', linestyle='-', label='Negative Correlation')
        ax.legend(handles=[green_line, red_line], title='Correlation Sign', loc='upper left', bbox_to_anchor=(1, 0.9))

        ax.set_title(f"Targeted Network Diagram for {selected_parameter} in {selected_process_label} (alpha={alpha})", fontsize=16, weight="bold")
        ax.axis('off')
        plt.tight_layout()
        st.pyplot(fig)

        # Generate bar chart
        st.write("### Correlation Coefficients with Selected Parameter")
        fig_bar, ax_bar = plt.subplots(figsize=(10, 6))
        sns.barplot(data=corr_data, x='Correlation', y='Parameter Name',
                    hue='Process', dodge=False, palette='tab20', ax=ax_bar)
        ax_bar.axvline(0, color='grey', linewidth=1)
        ax_bar.set_title(f"Correlation Coefficients with {selected_parameter} in {selected_process_label}",
                         fontsize=14, weight="bold")
        ax_bar.set_xlabel('Correlation Coefficient')
        ax_bar.set_ylabel('Parameters')
        ax_bar.legend(title='Process', bbox_to_anchor=(1, 1))
        plt.tight_layout()
        st.pyplot(fig_bar)

        try:
            end_progress = min(max(progress_increment, 0), 1)
            progress_bar.progress(int(end_progress * 100))
            status_text.text("Targeted Network Diagram generated.")
        except Exception as e:
            st.error(f"Error updating progress bar: {e}")

# -------------------------------
# Main Streamlit App
# -------------------------------
def main():
    st.set_page_config(page_title="WWTP Unit Processes Network Visualization", layout="wide")
    add_css()
    st.markdown("<h1 style='text-align: center; color:rgb(0, 0, 0);'>WWTP Unit Processes Network Visualization</h1>", unsafe_allow_html=True)

    # 1. Instructions
    st.markdown("<div class='section'>", unsafe_allow_html=True)
    st.markdown("""
    <div class='section-title'>Instructions</div>
    1. **Upload Files:** ...
    2. **Label Processes:** ...
    3. **Select Date Range:** ...
    4. **Reorder Processes:** ...
    5. **Generate Visualizations:** ...
    6. **Targeted Network Diagram:** ...
    """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # 2. File Upload and Labeling
    st.markdown("<div class='section'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Upload and Label Files</div>", unsafe_allow_html=True)

    uploaded_files = st.file_uploader("Choose CSV or Excel files", accept_multiple_files=True, type=['csv','xlsx','xls'])
    process_labels = []
    dataframes = []

    if uploaded_files:
        for idx, uploaded_file in enumerate(uploaded_files):
            st.subheader(f"Process File {idx + 1}: {uploaded_file.name}")
            try:
                if uploaded_file.name.endswith(('.xlsx','.xls')):
                    df = pd.read_excel(uploaded_file)
                else:
                    df = pd.read_csv(uploaded_file)
                df.columns = df.columns.str.lower().str.strip()
                if 'date' not in df.columns:
                    st.error(f"The file **{uploaded_file.name}** does not contain a 'date' column.")
                    st.stop()

                df['date'] = pd.to_datetime(df['date'], errors='coerce')
                df.dropna(subset=['date'], inplace=True)
                df = remove_outliers_zscore(df)
                dataframes.append(df)

                label = st.text_input(
                    f"Enter a label for **{uploaded_file.name}**:",
                    value=uploaded_file.name.split('.')[0],
                    key=f"label_{idx}"
                )
                process_labels.append(label)
            except Exception as e:
                st.error(f"Error processing file **{uploaded_file.name}**: {e}")
                st.stop()

        if len(dataframes) < 2:
            st.warning("Please upload at least two files.")
            st.stop()

        common_params = find_common_parameters(dataframes)
        if not common_params:
            st.error("No common parameters found.")
            st.stop()

        st.success(f"Common parameters identified: {', '.join(common_params)}")
    st.markdown("</div>", unsafe_allow_html=True)

    if uploaded_files and len(dataframes) >=2 and common_params:
        st.markdown("<div class='section'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Reorder Uploaded Files</div>", unsafe_allow_html=True)
        st.write("Please assign an order...")

        with st.sidebar:
            st.markdown("### Reorder Uploaded Files")
            order_numbers = []
            for idx, file in enumerate(uploaded_files):
                order = st.number_input(
                    f"Order for {file.name}",
                    min_value=1, max_value=len(uploaded_files),
                    value=idx+1, step=1, key=f"order_sidebar_{idx}"
                )
                order_numbers.append(order)
            if len(set(order_numbers))!= len(order_numbers):
                st.error("Each file must have a unique order.")
                st.stop()

            file_orders = list(zip(uploaded_files, process_labels, order_numbers))
            sorted_files = sorted(file_orders, key=lambda x:x[2])
            uploaded_files_sorted, process_labels_sorted, _ = zip(*sorted_files)
            dataframes_sorted = [
                df for _,_,df in sorted(
                    zip(uploaded_files, process_labels, dataframes),
                    key=lambda x: order_numbers[uploaded_files.index(x[0])]
                )
            ]
        st.markdown("</div>", unsafe_allow_html=True)

        # 4. Select Date Range
        st.markdown("<div class='section'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Select Date Range</div>", unsafe_allow_html=True)
        all_dates = pd.concat([df['date'] for df in dataframes_sorted])
        min_date = all_dates.min()
        max_date = all_dates.max()

        selected_dates = st.date_input(
            "Select Date Range for Analysis",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
            help="Choose start/end dates."
        )
        if isinstance(selected_dates, tuple) and len(selected_dates)==2:
            start_date, end_date = selected_dates
        else:
            st.error("Please select a valid start/end date.")
            st.stop()

        dataframes_filtered = []
        for idx, df in enumerate(dataframes_sorted):
            filtered_df = df[(df['date']>=pd.to_datetime(start_date)) & (df['date']<=pd.to_datetime(end_date))]
            dataframes_filtered.append(filtered_df)
            st.write(f"**{process_labels_sorted[idx]}**: {len(filtered_df)} records after filtering.")

        dataframes_sorted = dataframes_filtered
        common_params = find_common_parameters(dataframes_sorted)
        if not common_params:
            st.error("No common parameters found after date filtering.")
            st.stop()

        st.success(f"Common parameters after date filtering: {', '.join(common_params)}")
        st.markdown("</div>", unsafe_allow_html=True)

        # 5. Generate Heatmaps
        st.markdown("<div class='section'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Generate Heatmaps</div>", unsafe_allow_html=True)

        correlation_matrices = []
        parameters_per_edge = []
        for i in range(len(uploaded_files_sorted)-1):
            st.markdown(f"### Heatmap: **{process_labels_sorted[i]}** vs **{process_labels_sorted[i+1]}**")

            heatmap_progress = st.progress(0)
            heatmap_status = st.empty()

            df1 = dataframes_sorted[i][['date']+common_params]
            df2 = dataframes_sorted[i+1][['date']+common_params]
            merged_df = pd.merge(
                df1, df2, on="date",
                suffixes=(f"_{process_labels_sorted[i]}", f"_{process_labels_sorted[i+1]}")
            )
            merged_df = merged_df.drop(columns=["date"],errors="ignore")
            merged_df = merged_df.replace([np.inf,-np.inf], np.nan)
            merged_df = merged_df.dropna()
            numeric_columns = merged_df.select_dtypes(include=[np.number]).columns
            merged_df = merged_df[numeric_columns]

            filtered_corr_matrix = generate_heatmap(
                merged_df,
                f"Correlation Coefficient Heatmap: {process_labels_sorted[i]} vs {process_labels_sorted[i+1]}",
                ("X-Axis","Y-Axis"),
                progress_bar=heatmap_progress,
                status_text=heatmap_status,
                start_progress=0.0,
                end_progress=1.0
            )
            correlation_matrices.append(filtered_corr_matrix)

            shared_params = []
            for param in common_params:
                infl_param = f"{param}_{process_labels_sorted[i]}"
                ode_param  = f"{param}_{process_labels_sorted[i+1]}"
                if infl_param in filtered_corr_matrix.index and ode_param in filtered_corr_matrix.columns:
                    if filtered_corr_matrix.loc[infl_param, ode_param]!= 0:
                        shared_params.append(param)
            parameters_per_edge.append(shared_params)

        st.markdown("</div>", unsafe_allow_html=True)

        # 6. Identify Globally Shared
        st.markdown("<div class='section'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Globally Shared Parameters</div>", unsafe_allow_html=True)
        globally_shared_parameters = set(parameters_per_edge[0])
        for params in parameters_per_edge[1:]:
            globally_shared_parameters &= set(params)
        st.markdown(f"**Globally shared parameters:** {', '.join(globally_shared_parameters) if globally_shared_parameters else 'None'}")
        if not globally_shared_parameters:
            st.error("No globally shared parameters found.")
            st.stop()
        st.markdown("</div>", unsafe_allow_html=True)

        # 7. Generate Visualizations
        st.markdown("<div class='section'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Generate Visualizations</div>", unsafe_allow_html=True)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            if st.button("Generate Globally Shared Network Diagram"):
                global_net_progress = st.progress(0)
                global_net_status   = st.empty()
                generate_network_diagram_streamlit(
                    process_labels_sorted,
                    correlation_matrices,
                    globally_shared_parameters,
                    globally_shared=True,
                    progress_bar=global_net_progress,
                    status_text=global_net_status,
                    start_progress=0.0,
                    end_progress=1.0
                )

        with col2:
            if st.button("Generate Locally Shared Network Diagram"):
                local_net_progress = st.progress(0)
                local_net_status   = st.empty()
                generate_network_diagram_streamlit(
                    process_labels_sorted,
                    correlation_matrices,
                    parameters_per_edge,
                    globally_shared=False,
                    progress_bar=local_net_progress,
                    status_text=local_net_status,
                    start_progress=0.0,
                    end_progress=1.0
                )

        with col3:
            if st.button("Generate Bar Chart for Globally Shared Parameters"):
                bar_chart_progress = st.progress(0)
                bar_chart_status   = st.empty()
                plot_gspd_bar_chart(
                    process_labels_sorted,
                    globally_shared_parameters,
                    correlation_matrices,
                    progress_bar=bar_chart_progress,
                    status_text=bar_chart_status,
                    progress_increment=1.0
                )

        with col4:
            if st.button("Generate Line Graph for Globally Shared Parameters"):
                line_graph_progress = st.progress(0)
                line_graph_status   = st.empty()
                plot_gspd_line_graph(
                    process_labels_sorted,
                    globally_shared_parameters,
                    correlation_matrices,
                    progress_bar=line_graph_progress,
                    status_text=line_graph_status,
                    progress_increment=1.0
                )

        st.markdown("</div>", unsafe_allow_html=True)

        # 8. Targeted Diagram
        st.markdown("<div class='section'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Targeted Network Diagram</div>", unsafe_allow_html=True)
        st.write("Generate a network diagram centered around a specific parameter from a selected process.")

        targeted_net_progress = st.progress(0)
        targeted_net_status   = st.empty()
        generate_targeted_network_diagram_streamlit(
            process_labels_sorted,
            dataframes_sorted,
            progress_bar=targeted_net_progress,
            status_text=targeted_net_status,
            progress_increment=1.0
        )
        st.markdown("</div>", unsafe_allow_html=True)

def main():
    st.set_page_config(page_title="WWTP Unit Processes Network Visualization", layout="wide")
    add_css()
    st.markdown("<h1 style='text-align: center; color:rgb(0, 0, 0);'>WWTP Unit Processes Network Visualization</h1>", unsafe_allow_html=True)
    # Then the rest of your code for instructions, etc...
    # NOTE: Actually, from your snippet it looks like the 'main' is already fully defined above. 
    # So if you'd like to keep it consistent, just use your existing 'main' exactly as shown.

if __name__ == "__main__":
    main()