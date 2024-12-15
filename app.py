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
from io import StringIO

# Prevent matplotlib from trying to use any Xwindows backend.
matplotlib.use('Agg')

# -------------------------------
# Data Processing Functions
# -------------------------------

def bootstrap_correlations(df, n_iterations=500, method='pearson', progress_bar=None, status_text=None, start_progress=0.0, end_progress=0.4):
    correlations = []
    for i in range(n_iterations):
        df_resampled = resample(df)
        corr_matrix = df_resampled.corr(method=method)
        correlations.append(corr_matrix)

        # Update progress bar and status
        if progress_bar and status_text:
            progress = start_progress + (i + 1) / n_iterations * (end_progress - start_progress)
            progress_bar.progress(int(progress * 100))
            status_text.text(f"Bootstrapping {method.capitalize()} Correlations... ({i+1}/{n_iterations})")

    median_corr = pd.concat(correlations).groupby(level=0).median()
    return median_corr

def calculate_p_values(df, method='pearson'):
    p_values = pd.DataFrame(np.ones((df.shape[1], df.shape[1])), columns=df.columns, index=df.columns)
    for col1, col2 in itertools.combinations(df.columns, 2):
        try:
            if method == 'pearson':
                _, p_val = stats.pearsonr(df[col1], df[col2])
            elif method == 'spearman':
                _, p_val = stats.spearmanr(df[col1], df[col2])
            elif method == 'kendall':
                _, p_val = stats.kendalltau(df[col1], df[col2])
            else:
                p_val = 1.0
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

    # Start with all columns from the first DataFrame
    common_columns = set(dataframes[0].columns)

    # Intersect with columns from the remaining DataFrames
    for df in dataframes[1:]:
        common_columns &= set(df.columns)

    # Exclude the 'date' column
    common_columns.discard('date')

    return list(common_columns)

def remove_outliers_zscore(df, threshold=3):
    """
    Remove outliers from a DataFrame using the Z-score method.
    """
    st.write("Applying Z-Score Method to filter outliers...")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        st.warning("No numeric columns found to apply Z-Score filtering.")
        return df
    z_scores = (df[numeric_cols] - df[numeric_cols].mean()) / df[numeric_cols].std()
    mask = (z_scores.abs() < threshold).all(axis=1)
    filtered_df = df[mask]
    st.write(f"Outliers removed: {len(df) - len(filtered_df)}")
    return filtered_df

def validate_correlation_matrix(df, n_iterations=500, alpha=0.05, progress_bar=None, status_text=None, start_progress=0.0, end_progress=0.4):
    """
    Validate correlations using bootstrapping and p-value correction.
    Returns a filtered correlation matrix with only significant values.
    """
    st.write(f"DataFrame shape: {df.shape}")
    st.write("Bootstrapping correlation matrices...")

    # Ensure all columns are numeric
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.dropna(axis=1, how='all')  # Drop columns that are entirely non-numeric or NaN

    if df.empty:
        st.error("No numeric data available after removing non-numeric columns and NaN values.")
        return pd.DataFrame()

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

def generate_heatmap(df, title, labels, container):
    """
    Generate a heatmap and return the filtered correlation matrix.
    Each heatmap has its own progress bar and status text within its container.
    """
    with container:
        st.subheader(title)
        
        # Initialize individual progress bar and status text
        heatmap_progress = st.empty()
        heatmap_bar = st.progress(0)
        
        # Validate correlation matrix with progress updates
        filtered_corr_matrix = validate_correlation_matrix(
            df, progress_bar=heatmap_bar, status_text=heatmap_progress, 
            start_progress=0.0, end_progress=0.4
        )
        
        if filtered_corr_matrix.empty:
            st.warning("Cannot generate heatmap due to empty correlation matrix.")
            return filtered_corr_matrix
        
        parameter_order = sorted(filtered_corr_matrix.index)
        filtered_corr_matrix = filtered_corr_matrix.loc[parameter_order, parameter_order]
    
        np.fill_diagonal(filtered_corr_matrix.values, 1)
    
        # Check if the correlation matrix has valid data
        if filtered_corr_matrix.empty or filtered_corr_matrix.isnull().all().all():
            st.warning("Correlation matrix is empty or contains only NaN values.")
            heatmap_bar.progress(100)
            heatmap_progress.text("Heatmap generation incomplete due to insufficient data.")
            return filtered_corr_matrix
    
        fig = px.imshow(
            filtered_corr_matrix,
            text_auto=".2f",
            color_continuous_scale="RdBu",
            zmin=-1,
            zmax=1,
            # Use generic axis labels to fix hover text issue
            labels={"x": "X-Axis", "y": "Y-Axis", "color": "Correlation Coefficient"},
            title=title,
        )
    
        fig.update_layout(
            title=dict(
                text=title,
                font=dict(size=20),
                x=0.5,               # Center horizontally
                xanchor='center',    # Anchor the title at the center
                yanchor='top'        # Anchor the title at the top
            ),
            xaxis=dict(tickangle=45, title=None, tickfont=dict(size=12)),
            yaxis=dict(title=None, tickfont=dict(size=12)),
            autosize=True,
            margin=dict(l=100, r=100, t=100, b=100),
        )
    
        st.plotly_chart(fig, use_container_width=True)
        
        # Complete progress
        heatmap_bar.progress(100)
        heatmap_progress.text("Heatmap generation complete.")
        
        return filtered_corr_matrix

def generate_network_diagram_streamlit(labels, correlation_matrices, parameters, container, globally_shared=True):
    """
    Generate a parameter-based network diagram.
    If globally_shared is True, use globally shared parameters.
    Otherwise, use locally shared parameters for each edge.
    Each network diagram has its own progress bar and status text within its container.
    """
    with container:
        diagram_type = "Globally Shared" if globally_shared else "Locally Shared"
        st.subheader(f"{diagram_type} Network Diagram")
    
        # Initialize individual progress bar and status text
        network_progress = st.empty()
        network_bar = st.progress(0)
    
        # Initialize graph
        G = nx.MultiGraph()
    
        # Collect data for edge summary boxes
        edge_summaries = []
    
        total_connections = len(labels) -1
        heatmap_progress_increment = 0.4  # From heatmaps
        network_progress_increment = 0.3  # 30% of total
        bar_chart_progress_increment = 0.1
        line_graph_progress_increment = 0.1
        targeted_network_progress_increment = 0.1
    
        # Collect and add edges based on significant correlations
        num_edges = len(correlation_matrices)
        for i in range(num_edges):
            st.write(f"Processing connection: {labels[i]} → {labels[i + 1]}")
    
            # Retrieve the filtered correlation matrix for this pair
            filtered_corr_matrix = correlation_matrices[i]
    
            # Track added edges to avoid duplicates
            added_edges = set()
    
            if globally_shared:
                parameters_to_use = parameters  # Use the set of globally shared parameters
            else:
                parameters_to_use = parameters[i]  # Use the list of parameters for this edge
    
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
    
                    # Add nodes and edge
                    G.add_node(node1, label=node1)
                    G.add_node(node2, label=node2)
    
                    G.add_edge(
                        node1,
                        node2,
                        parameter=param,
                        correlation=corr_value,
                        weight=abs(corr_value),
                        key=param  # Use the parameter as the key for multi-edges
                    )
                    added_edges.add(edge_key)
    
                    # Add to edge summary
                    edge_summary['parameters'].append((param, corr_value))
    
            if edge_summary['parameters']:
                edge_summaries.append(edge_summary)
    
            # Update progress
            if network_bar and network_progress:
                progress = (i +1)/total_connections * network_progress_increment * 100
                network_bar.progress(int(progress))
                network_progress.text(f"Processing connection: {node1} → {node2}")
    
        if G.number_of_nodes() == 0:
            st.warning("No nodes to display in the network diagram.")
            network_bar.progress(int(network_progress_increment * 100))
            network_progress.text("Network Diagram generation incomplete.")
            return
    
        # Create a figure with GridSpec: 2 rows (network diagram and text boxes)
        fig = plt.figure(figsize=(18, 18))  # (width, height)
        gs = GridSpec(2, 1, height_ratios=[3, 1], hspace=0.3)
    
        # Upper subplot for the network diagram
        ax_network = fig.add_subplot(gs[0, 0])
    
        # Adjust layout
        if globally_shared:
            pos = nx.kamada_kawai_layout(G)
        else:
            pos = nx.spring_layout(G, k=0.15, iterations=200, seed=42)  # Adjusted 'k' for closer nodes
    
        # Draw nodes
        node_colors = ["lightblue"] * len(G.nodes())
        nx.draw_networkx_nodes(G, pos, node_size=8000, node_color=node_colors, ax=ax_network)
    
        # Wrap node labels
        max_label_width = 10  # Adjust as needed
        formatted_labels = {}
        for node in G.nodes():
            label_text = G.nodes[node]['label'].replace("_", " ")
            wrapped_label = "\n".join(textwrap.wrap(label_text, width=max_label_width))
            formatted_labels[node] = wrapped_label
    
        # Draw labels with formatted labels
        nx.draw_networkx_labels(G, pos, labels=formatted_labels, font_size=12, font_weight="bold", ax=ax_network)
    
        # Assign unique colors to parameters
        unique_parameters = list(set(d['parameter'] for u, v, k, d in G.edges(keys=True, data=True)))
        num_params = len(unique_parameters)
        base_colors = plt.cm.tab10.colors  # You can choose other colormaps if you have more than 10 parameters
        if num_params > len(base_colors):
            base_colors = plt.cm.tab20.colors  # Use a colormap with more colors
        parameter_colors = dict(zip(unique_parameters, base_colors[:num_params]))
    
        # Function to adjust color intensity based on correlation strength
        def adjust_color_intensity(base_color, corr_value):
            rgba = to_rgba(base_color)
            intensity = 1.0  # Keep alpha at 1 for consistency
            adjusted_color = (rgba[0], rgba[1], rgba[2], intensity)
            return adjusted_color
    
        # Draw edges with curvature to avoid overlaps
        num_edges_total = len(G.edges(keys=True))
        curvature_values = np.linspace(-0.5, 0.5, num_edges_total)  # Adjusted for better curvature
    
        for idx, (u, v, key, d) in enumerate(G.edges(data=True, keys=True)):
            curvature = curvature_values[idx] if num_edges_total > 1 else 0.2
            corr_value = d['correlation']
            parameter = d['parameter']
            base_color = parameter_colors[parameter]
            edge_color = adjust_color_intensity(base_color, corr_value)
    
            # Choose line style based on correlation sign
            style = 'solid' if corr_value >= 0 else 'dashed'
    
            # Draw the edge
            nx.draw_networkx_edges(
                G,
                pos,
                edgelist=[(u, v, key)],
                connectionstyle=f"arc3,rad={curvature}",
                edge_color=[edge_color],
                width=d["weight"] * 5,
                style=style,
                ax=ax_network
            )
    
        # Set title for the network diagram
        ax_network.set_title(f"{diagram_type} Parameter-Based Network Diagram", fontsize=16, pad=20, weight="bold")
    
        # Create consolidated edge summary text box
        section_boxes = []
    
        for summary in edge_summaries:
            node1, node2 = summary['nodes']
            process_pair_title = f"{node1} → {node2}"
            title_area = TextArea(process_pair_title, textprops=dict(color='black', size=12, weight='bold'))
    
            # Create content boxes for parameters
            content_boxes = []
            for param, corr in summary['parameters']:
                color = parameter_colors[param]
                da = DrawingArea(20, 10, 0, 0)
                da.add_artist(mpatches.Rectangle((0, 0), 20, 10, fc=color, ec='black'))
                line_text = f"{param}: {corr:.2f}"
                ta = TextArea(line_text, textprops=dict(color='black', size=10))
                hbox = HPacker(children=[da, ta], align="center", pad=0, sep=5)
                content_boxes.append(hbox)
            # Pack the title and parameters vertically
            section_box = VPacker(children=[title_area] + content_boxes, align="left", pad=0, sep=2)
            section_boxes.append(section_box)
    
        # Pack all sections into one box with spacing
        all_sections_box = VPacker(children=section_boxes, align="left", pad=0, sep=10)
    
        # Diagram Interpretation
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
    
        # Combine edge summaries and interpretation
        combined_box = VPacker(children=[all_sections_box, interpretation_area], align="left", pad=20, sep=20)
    
        # Create the lower subplot for text boxes
        ax_text = fig.add_subplot(gs[1, 0])
        ax_text.axis("off")  # Hide the axes
    
        ab = AnnotationBbox(
            combined_box,
            (0.5, 0.5),  # Center of the subplot
            xycoords='axes fraction',
            box_alignment=(0.5, 0.5),
            bboxprops=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5', alpha=0.9)
        )
        ax_text.add_artist(ab)
    
        plt.tight_layout()
        st.pyplot(fig)
        
        # Complete progress
        network_bar.progress(int(network_progress_increment * 100))
        network_progress.text(f"{diagram_type} Network Diagram generation complete.")

def plot_gspd_bar_chart(process_labels, globally_shared_parameters, correlation_matrices, container, progress_increment):
    """
    Generate a bar chart summarizing correlations for globally shared parameters across process pairs.
    Each bar chart has its own progress bar and status text within its container.
    """
    with container:
        st.write("### Bar Chart: Globally Shared Parameter Correlations")
        
        # Initialize individual progress bar and status text
        bar_chart_progress = st.empty()
        bar_chart_bar = st.progress(0)
        
        # Initialize data structure for correlations
        data = {param: [] for param in globally_shared_parameters}
        process_pairs = []
    
        num_process_pairs = len(correlation_matrices)
        num_parameters = len(globally_shared_parameters)
        total_steps = num_process_pairs * num_parameters
        step = 0
    
        # Collect correlation data for each process pair
        for i, matrix in enumerate(correlation_matrices):
            pair_label = f"{process_labels[i]} → {process_labels[i + 1]}"
            process_pairs.append(pair_label)
    
            for param in globally_shared_parameters:
                infl_param = f"{param}_{process_labels[i]}"
                ode_param = f"{param}_{process_labels[i + 1]}"
    
                if infl_param in matrix.index and ode_param in matrix.columns:
                    corr_value = matrix.loc[infl_param, ode_param]
                    data[param].append(corr_value)
                else:
                    data[param].append(0)  # Fill missing correlations with 0
                step +=1
                progress = (step / total_steps) * progress_increment * 100
                bar_chart_bar.progress(int(progress))
                bar_chart_progress.text(f"Generating Bar Chart... ({step}/{total_steps})")
    
        # Compute y-limits for consistent axes
        all_correlations = [corr for correlations in data.values() for corr in correlations]
        ymin = min(all_correlations + [0])
        ymax = max(all_correlations + [0])
        y_range = ymax - ymin
        margin = y_range * 0.1  # Add 10% margin
        ymin -= margin
        ymax += margin
    
        # Plot bar chart
        num_process_pairs = len(process_pairs)
        num_parameters = len(globally_shared_parameters)
        total_bar_width = 0.8  # Total width for all bars at one x position
        bar_width = total_bar_width / num_parameters
    
        x = np.arange(num_process_pairs)  # Positions of the process pairs
    
        fig, ax = plt.subplots(figsize=(14, 8))
    
        # Plot bars
        for i, (param, correlations) in enumerate(sorted(data.items())):
            offset = (i - (num_parameters - 1) / 2) * bar_width
            x_positions = x + offset
            ax.bar(x_positions, correlations, bar_width, label=param, alpha=0.9)
    
        ax.set_xlabel("Process Pairs", fontsize=12)
        ax.set_ylabel("Correlation Coefficient (r)", fontsize=12)
    
        # Adjust the title position
        ax.set_title("Globally Shared Parameter Correlations (Bar Chart)", fontsize=16, weight="bold", pad=30)
    
        # Adjust x-ticks
        ax.set_xticks(x)
        ax.set_xticklabels(process_pairs, rotation=45, ha="right")
    
        # Set y-limits
        ax.set_ylim(ymin, ymax)
    
        # Add horizontal line at r=0
        ax.axhline(y=0, color='grey', linewidth=1)
    
        # Adjust the layout to make room for the legend and title
        fig.subplots_adjust(top=0.85, bottom=0.2)
    
        # Position the legend just below the title and add a box around it
        legend = fig.legend(
            title="Parameters",
            loc='upper center',
            bbox_to_anchor=(0.5, 0.90),
            ncol=len(globally_shared_parameters),
            frameon=True  # Add a box around the legend
        )
    
        # Adjust the legend's transparency and outline color
        legend.get_frame().set_facecolor('white')  # Set legend background color to white
        legend.get_frame().set_alpha(1.0)  # Make legend opaque
        legend.get_frame().set_edgecolor('black')  # Set legend outline color to black
    
        st.pyplot(fig)
        
        # Complete progress
        bar_chart_bar.progress(int((1.0) * progress_increment * 100))
        bar_chart_progress.text("Bar Chart generation complete.")

def plot_gspd_line_graph(process_labels, globally_shared_parameters, correlation_matrices, container, progress_increment):
    """
    Generate a line graph summarizing correlations for globally shared parameters across process pairs.
    Each line graph has its own progress bar and status text within its container.
    """
    with container:
        st.write("### Line Graph: Globally Shared Parameter Correlations")
        
        # Initialize individual progress bar and status text
        line_graph_progress = st.empty()
        line_graph_bar = st.progress(0)
        
        # Initialize data structure for correlations
        data = {param: [] for param in globally_shared_parameters}
        process_pairs = []
    
        num_process_pairs = len(correlation_matrices)
        num_parameters = len(globally_shared_parameters)
        total_steps = num_process_pairs * num_parameters
        step = 0
    
        # Collect correlation data for each process pair
        for i, matrix in enumerate(correlation_matrices):
            pair_label = f"{process_labels[i]} → {process_labels[i + 1]}"
            process_pairs.append(pair_label)
    
            for param in globally_shared_parameters:
                infl_param = f"{param}_{process_labels[i]}"
                ode_param = f"{param}_{process_labels[i + 1]}"
    
                if infl_param in matrix.index and ode_param in matrix.columns:
                    corr_value = matrix.loc[infl_param, ode_param]
                    data[param].append(corr_value)
                else:
                    data[param].append(0)  # Fill missing correlations with 0
                step +=1
                progress = (step / total_steps) * progress_increment * 100
                line_graph_bar.progress(int(progress))
                line_graph_progress.text(f"Generating Line Graph... ({step}/{total_steps})")
    
        # Compute y-limits for consistent axes
        all_correlations = [corr for correlations in data.values() for corr in correlations]
        ymin = min(all_correlations + [0])
        ymax = max(all_correlations + [0])
        y_range = ymax - ymin
        margin = y_range * 0.1  # Add 10% margin
        ymin -= margin
        ymax += margin
    
        x = np.arange(len(process_pairs))  # Positions of the process pairs
    
        fig, ax = plt.subplots(figsize=(14, 8))
    
        # Plot lines
        for i, (param, correlations) in enumerate(sorted(data.items())):
            ax.plot(x, correlations, marker='o', linewidth=2, label=param)
    
        ax.set_xlabel("Process Pairs", fontsize=12)
        ax.set_ylabel("Correlation Coefficient (r)", fontsize=12)
    
        # Adjust the title position
        ax.set_title("Globally Shared Parameter Correlations (Line Graph)", fontsize=16, weight="bold", pad=30)
    
        # Adjust x-ticks
        ax.set_xticks(x)
        ax.set_xticklabels(process_pairs, rotation=45, ha="right")
    
        # Set y-limits
        ax.set_ylim(ymin, ymax)
    
        # Add horizontal line at r=0
        ax.axhline(y=0, color='grey', linewidth=1)
    
        # Adjust the layout to make room for the legend and title
        fig.subplots_adjust(top=0.85, bottom=0.2)
    
        # Position the legend just below the title and add a box around it
        legend = fig.legend(
            title="Parameters",
            loc='upper center',
            bbox_to_anchor=(0.5, 0.90),
            ncol=len(globally_shared_parameters),
            frameon=True  # Add a box around the legend
        )
    
        # Adjust the legend's transparency and outline color
        legend.get_frame().set_facecolor('white')  # Set legend background color to white
        legend.get_frame().set_alpha(1.0)  # Make legend opaque
        legend.get_frame().set_edgecolor('black')  # Set legend outline color to black
    
        st.pyplot(fig)
        
        # Complete progress
        line_graph_bar.progress(int((1.0) * progress_increment * 100))
        line_graph_progress.text("Line Graph generation complete.")

def generate_targeted_network_diagram_streamlit(process_labels, dataframes, container, progress_increment, n_iterations=500, alpha=0.05):
    """
    Generate a targeted network diagram centered around a selected parameter from a selected process.
    Each targeted network diagram has its own progress bar and status text within its container.
    """
    with container:
        st.write("### Targeted Network Diagram")
    
        # Initialize individual progress bar and status text
        targeted_progress = st.empty()
        targeted_bar = st.progress(0)
    
        # User selects a process
        selected_process_label = st.selectbox(
            "Select a Process:",
            options=process_labels,
            help="Choose the process you want to center the network diagram around."
        )
        process_choice = process_labels.index(selected_process_label)
        selected_dataframe = dataframes[process_choice]
    
        # Display available parameters in the selected process
        available_parameters = selected_dataframe.columns.drop('date', errors='ignore')
        selected_parameter = st.selectbox(
            "Select a Parameter:",
            options=available_parameters,
            help="Choose the parameter to center the network diagram around."
        )
    
        # User sets the significance level (alpha)
        alpha = st.number_input(
            "Set Significance Level (alpha):",
            min_value=0.0001,
            max_value=0.1,
            value=0.05,
            step=0.005,
            help="Adjust the significance level for correlation filtering."
        )
    
        if st.button("Generate Targeted Network Diagram"):
            st.write(f"Generating network diagram for **{selected_parameter}** in **{selected_process_label}** with alpha={alpha}...")
    
            # Update status
            targeted_progress.text("Preparing data for targeted network diagram...")
            targeted_bar.progress(int((0.05 * progress_increment) * 100))  # Data preparation as 5% of progress_increment
    
            # Prepare data for correlations
            combined_df = selected_dataframe[['date', selected_parameter]].copy()
            combined_df.columns = ['date', f"{selected_parameter}_{selected_process_label}"]
    
            # Include parameters from the same process
            df_same_process = selected_dataframe.drop(columns=[selected_parameter], errors="ignore")
            df_same_process.columns = [f"{col}_{selected_process_label}" if col != 'date' else 'date' for col in df_same_process.columns]
    
            # Merge on 'date'
            combined_df = pd.merge(combined_df, df_same_process, on='date', how='inner')
    
            # Include parameters from other processes
            for idx, df in enumerate(dataframes):
                if idx != process_choice:
                    process_label = process_labels[idx]
                    df_temp = df.copy()
                    df_temp.columns = [f"{col}_{process_label}" if col != 'date' else 'date' for col in df_temp.columns]
                    combined_df = pd.merge(combined_df, df_temp, on='date', how='inner')
    
            # Handle invalid values
            combined_df = combined_df.replace([np.inf, -np.inf], np.nan)
            combined_df = combined_df.dropna()
            numeric_columns = combined_df.select_dtypes(include=[np.number]).columns
            combined_df = combined_df[numeric_columns]
    
            # Apply Z-score outlier removal
            combined_df = remove_outliers_zscore(combined_df, threshold=3)
            targeted_bar.progress(int((0.10 * progress_increment) * 100))  # Data cleaning as additional 5%
    
            # Bootstrapping Correlations
            targeted_progress.text("Bootstrapping correlations...")
            pearson_corr = bootstrap_correlations(
                combined_df, n_iterations=n_iterations, method='pearson',
                progress_bar=targeted_bar, status_text=targeted_progress,
                start_progress=0.10, end_progress=0.40 * progress_increment
            )
            spearman_corr = bootstrap_correlations(
                combined_df, n_iterations=n_iterations, method='spearman',
                progress_bar=targeted_bar, status_text=targeted_progress,
                start_progress=0.40 * progress_increment, end_progress=0.70 * progress_increment
            )
            kendall_corr = bootstrap_correlations(
                combined_df, n_iterations=n_iterations, method='kendall',
                progress_bar=targeted_bar, status_text=targeted_progress,
                start_progress=0.70 * progress_increment, end_progress=0.95 * progress_increment
            )
    
            # Average the correlation matrices
            avg_corr_matrix = (pearson_corr + spearman_corr + kendall_corr) / 3
    
            target_param_full = f"{selected_parameter}_{selected_process_label}"
            if target_param_full not in avg_corr_matrix.columns:
                st.error(f"The selected parameter '{selected_parameter}' is not available in the data.")
                targeted_bar.progress(int((0.95 * progress_increment) * 100))
                targeted_progress.text("Targeted Network Diagram generation failed.")
                return
            target_correlations = avg_corr_matrix[target_param_full].drop(target_param_full)
    
            # Calculate p-values
            targeted_progress.text("Calculating and correcting p-values...")
            p_values = pd.Series(dtype=float)
            for col in target_correlations.index:
                if np.all(combined_df[target_param_full] == combined_df[col]):
                    # Perfect correlation, p-value is zero
                    p_values[col] = 0.0
                    continue
                try:
                    _, p_val = stats.pearsonr(combined_df[target_param_full], combined_df[col])
                    p_values[col] = p_val
                except Exception as e:
                    st.error(f"Error calculating p-value between {target_param_full} and {col}: {e}")
                    p_values[col] = 1.0  # Assign non-significant p-value
                    continue
    
            # Apply multiple testing correction
            _, corrected_p_values, _, _ = multipletests(p_values.values, alpha=alpha, method='fdr_bh')
            significance_mask = corrected_p_values < alpha
            significant_correlations = target_correlations[significance_mask]
            significant_p_values = corrected_p_values[significance_mask]
    
            # Check if any significant correlations are found
            if significant_correlations.empty:
                st.warning("No significant correlations found with the selected alpha level.")
                targeted_bar.progress(int((0.95 * progress_increment) * 100))
                targeted_progress.text("No significant correlations found.")
                return
    
            # Prepare data for bar chart
            corr_data = pd.DataFrame({
                'Parameter': significant_correlations.index,
                'Correlation': significant_correlations.values,
                'P-value': significant_p_values
            })
            corr_data['Process'] = corr_data['Parameter'].apply(lambda x: x.rsplit('_', 1)[1])
            corr_data['Parameter Name'] = corr_data['Parameter'].apply(lambda x: x.rsplit('_', 1)[0])
            corr_data = corr_data.sort_values('Correlation', key=abs, ascending=False)
    
            # Separate internal and external correlations
            internal_corr = corr_data[corr_data['Process'] == selected_process_label]
            external_corr = corr_data[corr_data['Process'] != selected_process_label]
    
            # Generate the network diagram
            G = nx.Graph()
            G.add_node(target_param_full, label=selected_parameter, process=selected_process_label)
    
            # Add internal correlations
            for idx, row in internal_corr.iterrows():
                G.add_node(row['Parameter'], label=row['Parameter Name'], process=row['Process'])
                G.add_edge(
                    target_param_full,
                    row['Parameter'],
                    correlation=row['Correlation'],
                    weight=abs(row['Correlation'])
                )
    
            # Add external correlations
            for idx, row in external_corr.iterrows():
                G.add_node(row['Parameter'], label=row['Parameter Name'], process=row['Process'])
                G.add_edge(
                    target_param_full,
                    row['Parameter'],
                    correlation=row['Correlation'],
                    weight=abs(row['Correlation'])
                )
    
            # Draw the network diagram
            pos = nx.spring_layout(G, seed=42)
    
            # Node colors based on process
            processes = list(set(nx.get_node_attributes(G, 'process').values()))
            color_map = {process: idx for idx, process in enumerate(processes)}
            cmap = plt.get_cmap('tab20')
            num_colors = len(processes)
            colors = [cmap(i / num_colors) for i in range(num_colors)]
            process_color_mapping = {process: colors[idx] for idx, process in enumerate(processes)}
            node_colors = [process_color_mapping[G.nodes[node]['process']] for node in G.nodes]
    
            fig, ax = plt.subplots(figsize=(14, 10))
    
            # Draw nodes
            nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=3000, ax=ax)
    
            # Draw labels
            labels = {node: f"{G.nodes[node]['label']}\n({G.nodes[node]['process']})" for node in G.nodes}
            nx.draw_networkx_labels(G, pos, labels=labels, font_size=10, ax=ax)
    
            # Edge colors and labels
            edge_colors = ['green' if G.edges[edge]['correlation'] > 0 else 'red' for edge in G.edges]
            edge_weights = [G.edges[edge]['weight'] * 5 for edge in G.edges]
            edge_labels = {(u, v): f"{G.edges[(u, v)]['correlation']:.2f}" for u, v in G.edges}
    
            # Draw edges
            nx.draw_networkx_edges(
                G, pos,
                edge_color=edge_colors,
                width=edge_weights,
                ax=ax
            )
    
            # Add edge labels for correlation coefficients
            nx.draw_networkx_edge_labels(
                G, pos,
                edge_labels=edge_labels,
                font_color='blue',
                font_size=8,
                ax=ax
            )
    
            # Add legend for processes
            process_legend = [plt.Line2D([0], [0], marker='o', color='w', label=process,
                                         markerfacecolor=process_color_mapping[process], markersize=10) for process in processes]
            ax.legend(handles=process_legend, title='Processes', loc='upper left', bbox_to_anchor=(1, 1))
    
            # Add edge legend
            green_line = plt.Line2D([], [], color='green', marker='_', linestyle='-', label='Positive Correlation')
            red_line = plt.Line2D([], [], color='red', marker='_', linestyle='-', label='Negative Correlation')
            ax.legend(handles=[green_line, red_line], title='Correlation Sign', loc='upper left', bbox_to_anchor=(1, 0.9))
    
            ax.set_title(f"Targeted Network Diagram for {selected_parameter} in {selected_process_label} (alpha={alpha})", fontsize=16, weight="bold")
            ax.axis('off')
            plt.tight_layout()
            st.pyplot(fig)
    
            # Generate bar chart of correlation coefficients
            st.write("### Correlation Coefficients with Selected Parameter")
            fig_bar, ax_bar = plt.subplots(figsize=(10, 6))
            sns.barplot(
                data=corr_data,
                x='Correlation',
                y='Parameter Name',
                hue='Process',
                dodge=False,
                palette='tab20',
                ax=ax_bar
            )
            ax_bar.axvline(0, color='grey', linewidth=1)
            ax_bar.set_title(f"Correlation Coefficients with {selected_parameter} in {selected_process_label}", fontsize=14, weight="bold")
            ax_bar.set_xlabel('Correlation Coefficient')
            ax_bar.set_ylabel('Parameters')
            ax_bar.legend(title='Process', bbox_to_anchor=(1, 1))
            plt.tight_layout()
            st.pyplot(fig_bar)
    
            # Update progress
            targeted_bar.progress(int((1.0) * progress_increment * 100))
            targeted_progress.text("Targeted Network Diagram generated.")

# -------------------------------
# Main Streamlit App
# -------------------------------

def main():
    st.set_page_config(page_title="WWTP Network Visualization", layout="wide")
    
    st.title("WWTP Unit Processes Network Visualization")
    st.write("""
    Welcome to the WWTP (Waste Water Treatment Plant) Unit Processes Network Visualization tool. 
    This application allows you to upload multiple data files, generate correlation heatmaps, and visualize the relationships between different processes and parameters within your WWTP system.
    """)
    
    # -------------------------------
    # Information Bar
    # -------------------------------
    st.info("""
    **Instructions:**
    
    - **File Upload:**
      - Upload multiple CSV or Excel files containing your WWTP process data.
      - Each file must include a 'date' column and one or more parameter columns.
      - Ensure that the parameter names are consistent across all files.
    
    - **File Ordering:**
      - After uploading, assign an order to each file to define the sequence of processes.
      - The order determines how correlation heatmaps and network diagrams are generated between processes.
    
    - **Visualization Interpretation:**
      - **Heatmaps:** Display the correlation coefficients between parameters of consecutive processes.
      - **Network Diagrams:** Illustrate significant correlations between parameters, highlighting positive and negative relationships.
      - **Bar and Line Charts:** Summarize the strength of correlations across different process pairs.
      - **Targeted Network Diagrams:** Focus on specific parameters within a chosen process to explore their connections.
    
    **CSV/Excel Formatting Requirements:**
    
    - **Columns:**
      - Must include a 'date' column.
      - Parameter columns should have consistent names across all files.
    - **Date Format:**
      - Dates should be in a recognizable date format (e.g., YYYY-MM-DD).
    - **Data Quality:**
      - Ensure there are no missing or non-numeric values in parameter columns.
    """)
    
    # -------------------------------
    # File Uploader with Drag-and-Drop and Label Assignment
    # -------------------------------
    st.subheader("Upload and Arrange Your Data Files")
    st.write("Drag and drop your CSV or Excel files below, then arrange them in the desired order and assign labels.")

    # Streamlit doesn't have native drag-and-drop reordering. We can simulate it using checkbox and reordering via Streamlit's interactive widgets.

    uploaded_files = st.file_uploader(
        "Upload CSV or Excel Files",
        accept_multiple_files=True,
        type=['csv', 'xlsx', 'xls']
    )

    if uploaded_files:
        st.success(f"{len(uploaded_files)} files uploaded successfully.")
        
        # Display uploaded files
        st.write("### Uploaded Files:")
        file_order = pd.DataFrame({
            'Filename': [file.name for file in uploaded_files]
        })
        
        # Allow user to reorder files
        st.write("### Arrange Files in Desired Order")
        reordered_filenames = st.experimental_data_editor(file_order, num_rows="dynamic", use_container_width=True, key="file_order")
        
        # Assign labels to each file
        st.write("### Assign Labels to Each Process")
        process_labels = []
        for idx, file in enumerate(uploaded_files):
            label = st.text_input(
                f"Label for '{file.name}':",
                value=f"Process {idx + 1}",
                key=f"label_{idx}"
            )
            process_labels.append(label)
        
        # Confirm button
        if st.button("Confirm File Upload and Order"):
            st.success("Files have been uploaded and ordered successfully.")
            
            # Process and sort files based on the reordered filenames
            sorted_filenames = reordered_filenames['Filename'].tolist()
            sorted_files = []
            sorted_labels = []
            for filename in sorted_filenames:
                for file, label in zip(uploaded_files, process_labels):
                    if file.name == filename:
                        sorted_files.append(file)
                        sorted_labels.append(label)
                        break
            
            # -------------------------------
            # Data Processing and Visualization
            # -------------------------------
            dataframes = []
            for idx, uploaded_file in enumerate(sorted_files):
                st.subheader(f"Process File {idx + 1}: {uploaded_file.name}")
                try:
                    # Read the uploaded file
                    if uploaded_file.name.endswith(('.xlsx', '.xls')):
                        df = pd.read_excel(uploaded_file)
                    else:
                        df = pd.read_csv(uploaded_file)
    
                    df.columns = df.columns.str.lower().str.strip()
                    if 'date' not in df.columns:
                        st.error(f"The file **{uploaded_file.name}** does not contain a 'date' column.")
                        st.stop()
    
                    df['date'] = pd.to_datetime(df['date'], errors='coerce')
                    df = df.dropna(subset=["date"])
                    df = remove_outliers_zscore(df)
                    dataframes.append(df)
                except Exception as e:
                    st.error(f"Error processing file **{uploaded_file.name}**: {e}")
                    st.stop()
    
            if len(dataframes) < 2:
                st.warning("Please upload at least two files to generate diagrams.")
                st.stop()
    
            # Identify common parameters
            common_params = find_common_parameters(dataframes)
            if not common_params:
                st.error("No common parameters found across all uploaded files.")
                st.stop()
    
            st.success(f"Common parameters identified: {', '.join(common_params)}")
    
            # Assign progress fractions
            num_heatmaps = len(dataframes) -1
            heatmap_progress_fraction = 0.4  # 40%
            heatmap_step = heatmap_progress_fraction / num_heatmaps
    
            network_diagram_progress_fraction = 0.3  # 30%
            network_diagram_step = network_diagram_progress_fraction / 2  # Two diagrams
    
            bar_chart_progress_fraction = 0.1  # 10%
            line_graph_progress_fraction = 0.1  # 10%
            targeted_network_progress_fraction = 0.1  # 10%
    
            # Generate heatmaps and store correlation matrices
            correlation_matrices = []
            parameters_per_edge = []
            for i in range(len(dataframes) - 1):
                heatmap_container = st.container()
                merged_df = pd.merge(
                    dataframes[i][['date'] + common_params],
                    dataframes[i + 1][['date'] + common_params],
                    on="date",
                    suffixes=(f"_{sorted_labels[i]}", f"_{sorted_labels[i + 1]}")
                ).drop(columns=["date"], errors="ignore") \
                 .replace([np.inf, -np.inf], np.nan) \
                 .dropna() \
                 .select_dtypes(include=[np.number])
    
                filtered_corr_matrix = generate_heatmap(
                    df=merged_df,
                    title=f"Correlation Coefficient Heatmap: {sorted_labels[i]} vs {sorted_labels[i + 1]}",
                    labels=(sorted_labels[i], sorted_labels[i + 1]),
                    container=heatmap_container
                )
                correlation_matrices.append(filtered_corr_matrix)
    
                # Identify parameters contributing to the correlation
                shared_params = []
                for param in common_params:
                    infl_param = f"{param}_{sorted_labels[i]}"
                    ode_param = f"{param}_{sorted_labels[i + 1]}"
                    if infl_param in filtered_corr_matrix.index and ode_param in filtered_corr_matrix.columns:
                        if filtered_corr_matrix.loc[infl_param, ode_param] != 0:
                            shared_params.append(param)
                parameters_per_edge.append(shared_params)
    
            # Identify globally shared parameters
            globally_shared_parameters = set(parameters_per_edge[0])
            for params in parameters_per_edge[1:]:
                globally_shared_parameters &= set(params)
    
            st.markdown(f"**Globally shared parameters across all node pairs:** {', '.join(globally_shared_parameters) if globally_shared_parameters else 'None'}")
            if not globally_shared_parameters:
                st.error("No globally shared parameters found.")
                st.stop()
    
            # Buttons to generate network diagrams and additional visualizations
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                if st.button("Generate Globally Shared Network Diagram"):
                    network_container = st.container()
                    generate_network_diagram_streamlit(
                        labels=sorted_labels,
                        correlation_matrices=correlation_matrices,
                        parameters=globally_shared_parameters,
                        container=network_container,
                        globally_shared=True
                    )
    
            with col2:
                if st.button("Generate Locally Shared Network Diagram"):
                    network_container = st.container()
                    generate_network_diagram_streamlit(
                        labels=sorted_labels,
                        correlation_matrices=correlation_matrices,
                        parameters=parameters_per_edge,
                        container=network_container,
                        globally_shared=False
                    )
    
            with col3:
                if st.button("Generate Bar Chart for Globally Shared Parameters"):
                    bar_chart_container = st.container()
                    plot_gspd_bar_chart(
                        process_labels=sorted_labels,
                        globally_shared_parameters=globally_shared_parameters,
                        correlation_matrices=correlation_matrices,
                        container=bar_chart_container,
                        progress_increment=bar_chart_progress_fraction
                    )
    
            with col4:
                if st.button("Generate Line Graph for Globally Shared Parameters"):
                    line_graph_container = st.container()
                    plot_gspd_line_graph(
                        process_labels=sorted_labels,
                        globally_shared_parameters=globally_shared_parameters,
                        correlation_matrices=correlation_matrices,
                        container=line_graph_container,
                        progress_increment=line_graph_progress_fraction
                    )
    
            st.markdown("---")
            st.markdown("### Additional Visualizations:")
            st.write("Use the buttons above to generate network diagrams and correlation summary charts.")
    
            st.markdown("---")
            st.markdown("### Targeted Network Diagram:")
            st.write("Generate a network diagram centered around a specific parameter from a selected process.")
    
            # Integrate the targeted network diagram
            targeted_network_container = st.container()
            generate_targeted_network_diagram_streamlit(
                process_labels=sorted_labels, 
                dataframes=dataframes, 
                container=targeted_network_container, 
                progress_increment=targeted_network_progress_fraction
            )
    

# -------------------------------
# Run the Streamlit App
# -------------------------------

if __name__ == "__main__":
    main()