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
    import pandas as pd
    import numpy as np
    from sklearn.utils import resample
    import itertools
    from scipy import stats

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
    import itertools
    import pandas as pd
    import numpy as np
    from scipy import stats

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
    import pandas as pd
    import numpy as np
    from statsmodels.stats.multitest import multipletests

    _, corrected, _, _ = multipletests(p_values.values.flatten(), alpha=0.05, method='fdr_bh')
    corrected_p = pd.DataFrame(corrected.reshape(p_values.shape), index=p_values.index, columns=p_values.columns)
    return corrected_p


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
    import streamlit as st
    import numpy as np
    from scipy import stats

    st.write("Applying Z-Score Method to filter outliers...")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        st.write("[DEBUG] No numeric columns found, skipping outlier removal.")
        return df

    before_count = len(df)
    z_scores = np.abs(stats.zscore(df[numeric_cols], nan_policy="omit"))
    mask = (z_scores < threshold).all(axis=1)
    filtered_df = df[mask]
    after_count = len(filtered_df)

    st.write(f"[DEBUG] Outlier removal: removed {before_count - after_count} rows (threshold={threshold}). Remaining: {after_count}")
    return filtered_df

def validate_correlation_matrix(df, n_iterations=500, alpha=0.05, progress_bar=None, status_text=None, start_progress=0.0, end_progress=1.0):
    import streamlit as st

    st.write(f"[DEBUG] DataFrame shape entering validate_correlation_matrix: {df.shape}")
    st.write("[DEBUG] Bootstrapping correlation matrices...")

    # Ensure numeric
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.dropna(axis=1, how='all')  # Drop all-NaN columns

    # 1) Pearson
    pearson_corr = bootstrap_correlations(
        df, n_iterations=n_iterations, method='pearson',
        progress_bar=progress_bar, status_text=status_text,
        start_progress=start_progress, end_progress=start_progress + (end_progress - start_progress) / 3
    )

    # 2) Spearman
    spearman_corr = bootstrap_correlations(
        df, n_iterations=n_iterations, method='spearman',
        progress_bar=progress_bar, status_text=status_text,
        start_progress=start_progress + (end_progress - start_progress) / 3,
        end_progress=start_progress + 2 * (end_progress - start_progress) / 3
    )

    # 3) Kendall
    kendall_corr = bootstrap_correlations(
        df, n_iterations=n_iterations, method='kendall',
        progress_bar=progress_bar, status_text=status_text,
        start_progress=start_progress + 2 * (end_progress - start_progress) / 3,
        end_progress=end_progress
    )

    avg_corr_matrix = (pearson_corr + spearman_corr + kendall_corr) / 3

    st.write("[DEBUG] Calculating and correcting p-values now...")
    p_values = calculate_p_values(df, method='pearson')
    corrected_p_values = correct_p_values(p_values)

    sig_mask = (corrected_p_values < alpha).astype(int)
    filtered_corr_matrix = avg_corr_matrix.where(sig_mask > 0).fillna(0)

    st.write("[DEBUG] Correlation matrix validated and filtered based on significance.")
    st.write(f"[DEBUG] filtered_corr_matrix shape: {filtered_corr_matrix.shape}")
    return filtered_corr_matrix

# -------------------------------
# Visualization Functions
# -------------------------------

def generate_heatmap(df, title, labels, progress_bar, status_text, start_progress, end_progress):
    import streamlit as st
    import plotly.express as px
    import numpy as np

    filtered_corr_matrix = validate_correlation_matrix(
        df, 
        n_iterations=500,  # or your default
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
            x=0.5, 
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
    import streamlit as st
    import networkx as nx
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgba
    import numpy as np
    import textwrap
    from matplotlib.gridspec import GridSpec
    from matplotlib.offsetbox import DrawingArea, TextArea, HPacker, VPacker, AnnotationBbox
    import matplotlib.patches as mpatches

    G = nx.MultiGraph()
    diagram_type = "Globally Shared" if globally_shared else "Locally Shared"

    st.subheader(f"{diagram_type} Network Diagram")

    # Collect data for edge summary boxes
    edge_summaries = []

    total_connections = len(labels) - 1
    for i in range(len(labels) - 1):
        st.write(f"[DEBUG] Processing connection: {labels[i]} → {labels[i + 1]}")

        filtered_corr_matrix = correlation_matrices[i]
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
            progress = start_progress + (i +1)/total_connections * (end_progress - start_progress)
            progress_bar.progress(int(progress * 100))
            status_text.text(f"Processing connection: {node1} → {node2}")

    if G.number_of_nodes() == 0:
        st.warning("No nodes to display in the network diagram.")
        return

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
        "• Edge colors correspond to parameters.\n"
        "• Solid lines: Positive correlations.\n"
        "• Dashed lines: Negative correlations.\n"
        "• Edge thickness reflects correlation strength."
    )
    interpretation_area = TextArea(interpretation_text, textprops=dict(fontsize=12))
    combined_box = VPacker(children=[all_sections_box, interpretation_area], align="left", pad=20, sep=20)

    ax_text = fig.add_subplot(gs[1, 0])
    ax_text.axis("off")

    ab = AnnotationBbox(
        combined_box,
        (0.5, 0.5),
        xycoords='axes fraction',
        box_alignment=(0.5, 0.5),
        bboxprops=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5', alpha=0.9)
    )
    ax_text.add_artist(ab)

    plt.tight_layout()
    st.pyplot(fig)

def plot_gspd_bar_chart(process_labels, globally_shared_parameters, correlation_matrices, progress_bar, status_text, progress_increment):
    """
    Generate a bar chart summarizing correlations for globally shared parameters across process pairs.
    """
    st.write("### Bar Chart: Globally Shared Parameter Correlations")
    
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
            progress = (step / total_steps) * progress_increment
            progress_bar.progress(int(progress * 100))
            status_text.text(f"Generating Bar Chart... ({step}/{total_steps})")

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

        # Update progress
        step +=1
        progress = (step / total_steps) * progress_increment
        progress_bar.progress(int(progress * 100))
        status_text.text(f"Generating Bar Chart... ({step}/{total_steps})")

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
    ax.axhline(y=0, color='black', linewidth=1)

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

def plot_gspd_line_graph(process_labels, globally_shared_parameters, correlation_matrices, progress_bar, status_text, progress_increment):
    import streamlit as st
    import matplotlib.pyplot as plt
    import numpy as np

    st.write("### Line Graph: Globally Shared Parameter Correlations")

    data = {param: [] for param in globally_shared_parameters}
    process_pairs = []

    num_process_pairs = len(correlation_matrices)
    num_parameters = len(globally_shared_parameters)
    total_steps = num_process_pairs * num_parameters
    step = 0

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
                data[param].append(0)

            step +=1
            progress = (step / total_steps) * progress_increment
            progress_bar.progress(int(progress * 100))
            status_text.text(f"[DEBUG] Generating Line Graph... ({step}/{total_steps})")

    all_correlations = [corr for correlations in data.values() for corr in correlations]
    ymin = min(all_correlations + [0])
    ymax = max(all_correlations + [0])
    y_range = ymax - ymin
    margin = y_range * 0.1
    ymin -= margin
    ymax += margin

    x = np.arange(len(process_pairs))
    fig, ax = plt.subplots(figsize=(14, 8))

    sorted_items = sorted(data.items())
    for i, (param, correlations) in enumerate(sorted_items):
        ax.plot(x, correlations, marker='o', linewidth=2, label=param)

        step +=1
        progress = (step / total_steps) * progress_increment
        progress_bar.progress(int(progress * 100))
        status_text.text(f"[DEBUG] Generating Line Graph... ({step}/{total_steps})")

    ax.set_xlabel("Process Pairs", fontsize=12)
    ax.set_ylabel("Correlation Coefficient (r)", fontsize=12)
    ax.set_title("Globally Shared Parameter Correlations (Line Graph)", fontsize=16, weight="bold", pad=30)
    ax.set_xticks(x)
    ax.set_xticklabels(process_pairs, rotation=45, ha="right")
    ax.set_ylim(ymin, ymax)
    ax.axhline(y=0, color='black', linewidth=1)
    fig.subplots_adjust(top=0.85, bottom=0.2)

    legend = fig.legend(
        title="Parameters",
        loc='upper center',
        bbox_to_anchor=(0.5, 0.90),
        ncol=len(globally_shared_parameters),
        frameon=True
    )
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(1.0)
    legend.get_frame().set_edgecolor('black')

    st.pyplot(fig)


def generate_targeted_network_diagram_streamlit(
    process_labels,
    dataframes,
    progress_bar,
    status_text,
    progress_increment,
    n_iterations=500,
    alpha=0.05
):
    import streamlit as st
    import pandas as pd
    import numpy as np
    import networkx as nx
    import matplotlib.pyplot as plt
    import seaborn as sns
    from statsmodels.stats.multitest import multipletests
    from scipy import stats

    st.write("### Targeted Network Diagram")

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
        max_value=1.0,
        value=0.05,
        step=0.005,
        help="Adjust the significance level for correlation filtering."
    )

    if st.button("Generate Targeted Network Diagram"):
        st.write(f"Generating network diagram for **{selected_parameter}** in **{selected_process_label}** with alpha={alpha}...")

        # Update status
        status_text.text("Preparing data for targeted network diagram...")
        progress_bar.progress(int(0.05 * progress_increment * 100))

        # 1) Prepare data for correlations
        combined_df = selected_dataframe[['date', selected_parameter]].copy()
        combined_df.columns = ['date', f"{selected_parameter}_{selected_process_label}"]

        # Same-process columns
        df_same_process = selected_dataframe.drop(columns=[selected_parameter], errors='ignore')
        df_same_process.columns = [
            f"{col}_{selected_process_label}" if col != 'date' else 'date'
            for col in df_same_process.columns
        ]
        combined_df = pd.merge(combined_df, df_same_process, on='date', how='inner')

        # Include parameters from other processes
        for idx, df in enumerate(dataframes):
            if idx != process_choice:
                process_label = process_labels[idx]
                df_temp = df.copy()
                df_temp.columns = [
                    f"{col}_{process_label}" if col != 'date' else 'date'
                    for col in df_temp.columns
                ]
                combined_df = pd.merge(combined_df, df_temp, on='date', how='inner')

        # Handle invalid values
        combined_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        combined_df.dropna(inplace=True)
        numeric_cols = combined_df.select_dtypes(include=[np.number]).columns
        combined_df = combined_df[numeric_cols]

        # [DEBUG] Show shape & sample
        st.write("[DEBUG] Combined DF shape (pre-outlier):", combined_df.shape)
        st.dataframe(combined_df.head(5))

        # 2) Outlier removal
        from scipy.stats import zscore
        before_outliers = len(combined_df)
        zvals = np.abs(zscore(combined_df, nan_policy='omit'))
        mask = (zvals < 3).all(axis=1)
        combined_df = combined_df[mask]
        after_outliers = len(combined_df)
        st.write(f"[DEBUG] Outlier removal: removed {before_outliers - after_outliers}. Remaining: {len(combined_df)}")

        progress_bar.progress(int(0.10 * progress_increment * 100))

        # 3) Bootstrapping
        from sklearn.utils import resample

        def local_bootstrap(df_local, iters, method):
            corr_list = []
            for _ in range(iters):
                dfr = resample(df_local)
                corr_matrix = dfr.corr(method=method)
                corr_list.append(corr_matrix)
            return pd.concat(corr_list).groupby(level=0).median()

        st.write("[DEBUG] Bootstrapping Pearson, Spearman, Kendall.")
        pearson_corr = local_bootstrap(combined_df, n_iterations, 'pearson')
        spearman_corr = local_bootstrap(combined_df, n_iterations, 'spearman')
        kendall_corr = local_bootstrap(combined_df, n_iterations, 'kendall')
        avg_corr_matrix = (pearson_corr + spearman_corr + kendall_corr) / 3

        # 4) Target param row
        target_param_full = f"{selected_parameter}_{selected_process_label}"
        if target_param_full not in avg_corr_matrix.columns:
            st.error(f"[DEBUG] The selected parameter '{selected_parameter}' is not available.")
            return

        target_correlations = avg_corr_matrix[target_param_full].drop(target_param_full, errors='ignore')

        # 5) p-values
        pvals = {}
        x = combined_df[target_param_full]
        for col in target_correlations.index:
            y = combined_df[col]
            if x.equals(y):
                pvals[col] = 1.0
                continue
            try:
                _, p_val = stats.pearsonr(x, y)
                pvals[col] = p_val
            except:
                pvals[col] = 1.0

        if len(pvals) == 0:
            st.warning("[DEBUG] No pairs to test.")
            return

        pvals_s = pd.Series(pvals)
        _, pvals_corr, _, _ = multipletests(pvals_s.values, alpha=alpha, method='fdr_bh')
        pvals_corr_s = pd.Series(pvals_corr, index=pvals_s.index)
        mask_sig = (pvals_corr_s < alpha)
        if not mask_sig.any():
            st.warning("No significant correlations found.")
            progress_bar.progress(int(0.95 * progress_increment * 100))
            return

        sig_corr_vals = target_correlations[mask_sig]
        df_sig = pd.DataFrame({
            'Parameter': sig_corr_vals.index,
            'Correlation': sig_corr_vals.values,
            'P-value': pvals_s[mask_sig],
            'P-value (corrected)': pvals_corr_s[mask_sig],
        }).sort_values('Correlation', key=abs, ascending=False)

        st.write("[DEBUG] Significant Correlations Table:")
        st.dataframe(df_sig)

        # 6) Build network
        G = nx.Graph()
        G.add_node(target_param_full, label=selected_parameter, process=selected_process_label)

        df_sig['Process'] = df_sig['Parameter'].apply(lambda x: x.rsplit('_', 1)[1])
        df_sig['Parameter Name'] = df_sig['Parameter'].apply(lambda x: x.rsplit('_', 1)[0])

        # Separate internal vs external
        internal_corr = df_sig[df_sig['Process'] == selected_process_label]
        external_corr = df_sig[df_sig['Process'] != selected_process_label]

        # Add internal edges
        for _, row in internal_corr.iterrows():
            G.add_node(row['Parameter'], label=row['Parameter Name'], process=row['Process'])
            G.add_edge(
                target_param_full, row['Parameter'],
                correlation=row['Correlation'],
                weight=abs(row['Correlation'])
            )

        # Add external edges
        for _, row in external_corr.iterrows():
            G.add_node(row['Parameter'], label=row['Parameter Name'], process=row['Process'])
            G.add_edge(
                target_param_full, row['Parameter'],
                correlation=row['Correlation'],
                weight=abs(row['Correlation'])
            )

        # 7) Draw network
        pos = nx.spring_layout(G, seed=42)
        internal_nodes = [n for n in G.nodes if G.nodes[n]['process'] == selected_process_label and n != target_param_full]
        external_nodes = [n for n in G.nodes if G.nodes[n]['process'] != selected_process_label]
        for nd in internal_nodes:
            pos[nd][0] -= 0.5
        for nd in external_nodes:
            pos[nd][0] += 0.5

        fig, ax = plt.subplots(figsize=(14, 10))
        processes = list(set(nx.get_node_attributes(G, 'process').values()))
        color_map = {proc: i for i, proc in enumerate(processes)}
        cmap = plt.get_cmap('tab20')
        colors = [cmap(i / len(processes)) for i in range(len(processes))]
        process_color_mapping = {proc: colors[i] for i, proc in enumerate(processes)}
        node_colors = [process_color_mapping[G.nodes[n]['process']] for n in G.nodes]

        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=3000, ax=ax)
        labels_dict = {n: f"{G.nodes[n]['label']}\n({G.nodes[n]['process']})" for n in G.nodes}
        nx.draw_networkx_labels(G, pos, labels=labels_dict, font_size=10, ax=ax)

        # Edge style
        edge_colors = [
            'green' if G.edges[e]['correlation'] > 0 else 'red'
            for e in G.edges
        ]
        edge_weights = [abs(G.edges[e]['correlation']) * 5 for e in G.edges]
        edge_labels = {(u, v): f"{G.edges[(u, v)]['correlation']:.2f}" for u, v in G.edges}

        nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=edge_weights, ax=ax)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='blue', font_size=8, ax=ax)

        # Legend
        proc_legend = [
            plt.Line2D([0], [0], marker='o', color='w', label=pr,
                       markerfacecolor=process_color_mapping[pr], markersize=10)
            for pr in processes
        ]
        ax.legend(handles=proc_legend, title='Processes', loc='upper left', bbox_to_anchor=(1, 1))

        green_line = plt.Line2D([], [], color='green', marker='_', linestyle='-', label='Positive Correlation')
        red_line = plt.Line2D([], [], color='red', marker='_', linestyle='-', label='Negative Correlation')
        ax.legend(handles=[green_line, red_line], title='Correlation Sign', loc='upper left', bbox_to_anchor=(1, 0.9))

        ax.set_title(
            f"Targeted Network Diagram for {selected_parameter} in {selected_process_label} (alpha={alpha})",
            fontsize=16, weight="bold"
        )
        ax.axis('off')
        plt.tight_layout()
        st.pyplot(fig)

        # 8) Bar chart
        st.write("### Correlation Coefficients with Selected Parameter")
        fig_bar, ax_bar = plt.subplots(figsize=(10, 6))
        sns.barplot(
            data=df_sig,
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
    # Set page config as the very first Streamlit command
    st.set_page_config(page_title="WWTP Unit Processes Network Visualization", layout="wide")
    
    # Add custom CSS for outlines
    add_css()

    # Add the main title
    st.markdown("<h1 style='text-align: center; color:rgb(0, 0, 0);'>WWTP Unit Processes Network Visualization</h1>", unsafe_allow_html=True)
    
    # -------------------------------
    # 1. Instructions Section
    # -------------------------------
    st.markdown("<div class='section'>", unsafe_allow_html=True)
    st.markdown("""
    <div class='section-title'>Instructions</div>
    1. **Upload Files:** Upload your CSV or Excel files containing process data. Ensure each file has a 'date' column.
    
    2. **Label Processes:** Assign descriptive labels to each uploaded process file.
    
    3. **Select Date Range:** Choose the specific date range you want to analyze.
    
    4. **Reorder Processes:** After uploading, assign an order to the processes based on their real-life sequence (upstream to downstream).
    
    5. **Generate Visualizations:** Click the buttons to generate correlation heatmaps, network diagrams, bar charts, and line graphs.
    
    6. **Targeted Network Diagram:** Use the section below to generate a network diagram centered around a specific parameter from a selected process.
    
    
    """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # -------------------------------
    # 2. File Upload and Labeling
    # -------------------------------
    st.markdown("<div class='section'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Upload and Label Files</div>", unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "Choose CSV or Excel files",
        accept_multiple_files=True,
        type=['csv', 'xlsx', 'xls']
    )
    process_labels = []
    dataframes = []

    if uploaded_files:
        for idx, uploaded_file in enumerate(uploaded_files):
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

                # Prompt user for a label for this process
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
            st.warning("Please upload at least two files to generate diagrams.")
            st.stop()

        # Identify common parameters
        common_params = find_common_parameters(dataframes)
        if not common_params:
            st.error("No common parameters found across all uploaded files.")
            st.stop()

        st.success(f"Common parameters identified: {', '.join(common_params)}")
    st.markdown("</div>", unsafe_allow_html=True)

    if uploaded_files and len(dataframes) >=2 and common_params:
        # -------------------------------
        # 3. Reordering Uploaded Files via Sidebar
        # -------------------------------
        st.markdown("<div class='section'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Reorder Uploaded Files</div>", unsafe_allow_html=True)
        st.write("Please assign an order to the uploaded files based on their sequence in real life (upstream to downstream).")

        with st.sidebar:
            st.markdown("### Reorder Uploaded Files")
            st.write("Please assign an order to the uploaded files based on their sequence in real life (upstream to downstream).")

            # Initialize list to store order
            order_numbers = []
            for idx, file in enumerate(uploaded_files):
                order = st.number_input(
                    f"Order for {file.name}", 
                    min_value=1, 
                    max_value=len(uploaded_files), 
                    value=idx+1, 
                    step=1, 
                    key=f"order_sidebar_{idx}"
                )
                order_numbers.append(order)

            # Validate unique order numbers
            if len(set(order_numbers)) != len(order_numbers):
                st.error("Each file must have a unique order number. Please adjust the order numbers accordingly.")
                st.stop()

            # Combine files, labels, and order
            file_orders = list(zip(uploaded_files, process_labels, order_numbers))

            # Sort files based on order
            sorted_files = sorted(file_orders, key=lambda x: x[2])

            # Unzip sorted files and labels
            uploaded_files_sorted, process_labels_sorted, _ = zip(*sorted_files)
            dataframes_sorted = [df for _, _, df in sorted(zip(uploaded_files, process_labels, dataframes), key=lambda x: order_numbers[uploaded_files.index(x[0])])]

        st.markdown("</div>", unsafe_allow_html=True)

        # -------------------------------
        # 4. Select Date Range
        # -------------------------------
        st.markdown("<div class='section'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Select Date Range</div>", unsafe_allow_html=True)

        # Combine all dates from the sorted dataframes
        all_dates = pd.concat([df['date'] for df in dataframes_sorted])

        # Determine the overall minimum and maximum dates
        min_date = all_dates.min()
        max_date = all_dates.max()

        # Display the date range picker
        selected_dates = st.date_input(
            "Select Date Range for Analysis",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
            help="Choose the start and end dates for the analysis."
        )

        # Ensure that the user has selected a start and end date
        if isinstance(selected_dates, tuple) and len(selected_dates) == 2:
            start_date, end_date = selected_dates
        else:
            st.error("Please select a valid start and end date.")
            st.stop()

        # Apply the date filter to each dataframe
        dataframes_filtered = []
        for idx, df in enumerate(dataframes_sorted):
            filtered_df = df[(df['date'] >= pd.to_datetime(start_date)) & (df['date'] <= pd.to_datetime(end_date))]
            dataframes_filtered.append(filtered_df)
            st.write(f"**{process_labels_sorted[idx]}**: {len(filtered_df)} records after filtering.")

        # Update the dataframes_sorted to the filtered dataframes
        dataframes_sorted = dataframes_filtered

        # Recompute common parameters after filtering
        common_params = find_common_parameters(dataframes_sorted)
        if not common_params:
            st.error("No common parameters found after applying the date filter.")
            st.stop()

        st.success(f"Common parameters after date filtering: {', '.join(common_params)}")
        st.markdown("</div>", unsafe_allow_html=True)

        # -------------------------------
        # 5. Generate Heatmaps and Store Correlation Matrices
        # -------------------------------
        st.markdown("<div class='section'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Generate Heatmaps</div>", unsafe_allow_html=True)

        correlation_matrices = []
        parameters_per_edge = []
        for i in range(len(uploaded_files_sorted) - 1):
            st.markdown(f"### Heatmap: **{process_labels_sorted[i]}** vs **{process_labels_sorted[i + 1]}**")

            # Create separate progress bar and status
            heatmap_progress = st.progress(0)
            heatmap_status = st.empty()

            # Merge data
            df1 = dataframes_sorted[i][['date'] + common_params]
            df2 = dataframes_sorted[i + 1][['date'] + common_params]
            merged_df = pd.merge(
                df1, df2, on="date",
                suffixes=(f"_{process_labels_sorted[i]}", f"_{process_labels_sorted[i + 1]}")
            )
            merged_df = merged_df.drop(columns=["date"], errors="ignore")
            merged_df = merged_df.replace([np.inf, -np.inf], np.nan)
            merged_df = merged_df.dropna()
            numeric_columns = merged_df.select_dtypes(include=[np.number]).columns
            merged_df = merged_df[numeric_columns]

            # Generate heatmap
            filtered_corr_matrix = generate_heatmap(
                merged_df,
                f"Correlation Coefficient Heatmap: {process_labels_sorted[i]} vs {process_labels_sorted[i + 1]}",
                ("X-Axis", "Y-Axis"),
                progress_bar=heatmap_progress,
                status_text=heatmap_status,
                start_progress=0.0,
                end_progress=1.0
            )
            correlation_matrices.append(filtered_corr_matrix)

            # Identify parameters contributing to the correlation
            shared_params = []
            for param in common_params:
                infl_param = f"{param}_{process_labels_sorted[i]}"
                ode_param = f"{param}_{process_labels_sorted[i + 1]}"
                if infl_param in filtered_corr_matrix.index and ode_param in filtered_corr_matrix.columns:
                    if filtered_corr_matrix.loc[infl_param, ode_param] != 0:
                        shared_params.append(param)
            parameters_per_edge.append(shared_params)

        st.markdown("</div>", unsafe_allow_html=True)

        # -------------------------------
        # 6. Identify Globally Shared Parameters
        # -------------------------------
        st.markdown("<div class='section'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Globally Shared Parameters</div>", unsafe_allow_html=True)
        globally_shared_parameters = set(parameters_per_edge[0])
        for params in parameters_per_edge[1:]:
            globally_shared_parameters &= set(params)

        st.markdown(f"**Globally shared parameters across all node pairs:** {', '.join(globally_shared_parameters) if globally_shared_parameters else 'None'}")
        if not globally_shared_parameters:
            st.error("No globally shared parameters found.")
            st.stop()
        st.markdown("</div>", unsafe_allow_html=True)

        # -------------------------------
        # 7. Generate Network Diagrams and Charts with Separate Progress Bars
        # -------------------------------
        st.markdown("<div class='section'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Generate Visualizations</div>", unsafe_allow_html=True)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            if st.button("Generate Globally Shared Network Diagram"):
                # Create separate progress bar and status
                global_net_progress = st.progress(0)
                global_net_status = st.empty()

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
                # Create separate progress bar and status
                local_net_progress = st.progress(0)
                local_net_status = st.empty()

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
                # Create separate progress bar and status
                bar_chart_progress = st.progress(0)
                bar_chart_status = st.empty()

                plot_gspd_bar_chart(
                    process_labels_sorted,
                    globally_shared_parameters,
                    correlation_matrices,
                    progress_bar=bar_chart_progress,
                    status_text=bar_chart_status,
                    progress_increment=1.0  # Full progress for bar chart
                )

        with col4:
            if st.button("Generate Line Graph for Globally Shared Parameters"):
                # Create separate progress bar and status
                line_graph_progress = st.progress(0)
                line_graph_status = st.empty()

                plot_gspd_line_graph(
                    process_labels_sorted,
                    globally_shared_parameters,
                    correlation_matrices,
                    progress_bar=line_graph_progress,
                    status_text=line_graph_status,
                    progress_increment=1.0  # Full progress for line graph
                )

        st.markdown("</div>", unsafe_allow_html=True)

        # -------------------------------
        # 8. Targeted Network Diagram Section with Separate Progress Bar
        # -------------------------------
        st.markdown("<div class='section'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Targeted Network Diagram</div>", unsafe_allow_html=True)
        st.write("Generate a network diagram centered around a specific parameter from a selected process.")

        # Create separate progress bar and status
        targeted_net_progress = st.progress(0)
        targeted_net_status = st.empty()

        generate_targeted_network_diagram_streamlit(
            process_labels_sorted,
            dataframes_sorted,
            progress_bar=targeted_net_progress,
            status_text=targeted_net_status,
            progress_increment=1.0  # Full progress for targeted network diagram
        )
        st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------
# Run the Streamlit App
# -------------------------------

if __name__ == "__main__":
    main()