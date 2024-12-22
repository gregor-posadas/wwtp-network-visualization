# WWTP Statistical Framework and Visualization Tool

## Overview
This Python-based tool provides advanced statistical analysis and visualization for wastewater treatment plant (WWTP) data. It is designed to support operators and researchers by transforming complex datasets into actionable insights through intuitive heatmaps and network diagrams.

---

## Features
- **Statistical Framework**:
  - Bootstrapping for robust correlation analysis.
  - Multiple correlation methods: Pearson, Spearman, Kendall.
  - Multiple testing correction using the Benjamini-Hochberg False Discovery Rate (FDR).
- **Visualization Tools**:
  - Heatmaps for correlation matrices.
  - Network diagrams for parameter relationships.
  - Bar and line charts for trend analysis.
- **Accessibility**:
  - Streamlit-based web interface.
  - Intuitive workflows for non-expert users.

---

## Installation

### Prerequisites
1. Install Python 3.8 or later.
2. Install required libraries:
   ```bash
   pip install pandas numpy scipy statsmodels plotly matplotlib seaborn networkx streamlit
   ```

---

## How to Use

### 1. Run the Tool
1. Clone the repository or copy the code into your local environment.
2. Launch the tool using:
   ```bash
   streamlit run app.py
   ```
3. Open the provided URL in your browser.

### 2. Upload Data
- Upload CSV or Excel files containing wastewater treatment data.
- Ensure files include a `date` column for alignment across datasets.

### 3. Configure Settings
- Select correlation methods: Pearson, Spearman, Kendall.
- Filter data by date range.
- Assign labels to uploaded datasets.

### 4. Generate Visualizations
- Use the interface to generate:
  - **Heatmaps**: Visualize parameter correlations.
  - **Globally Shared Network Diagrams**: Explore shared parameter relationships across processes.
  - **Targeted Network Diagrams**: Focus on a specific parameter and its correlations.
  - **Bar and Line Charts**: Summarize trends and correlations.

---

## Outputs
- **Heatmaps**: Color-coded grids displaying the strength and direction of parameter correlations.
- **Network Diagrams**:
  - Nodes represent parameters or processes.
  - Edges represent significant correlations, with color and thickness indicating strength.
- **Charts**: Summarize correlations and trends across datasets.

---

## Statistical Methods

### Bootstrapping
- Resamples data to compute robust correlation coefficients across iterations.

### Multiple Correlation Methods
- **Pearson**: Measures linear relationships.
- **Spearman**: Captures monotonic relationships.
- **Kendall**: Evaluates ordinal associations.

### P-Value Adjustment
- Applies Benjamini-Hochberg FDR to control for false positives.

---

## Example Workflow
1. Upload multiple datasets (e.g., influent, effluent).
2. Filter datasets by date and common parameters.
3. Generate visualizations for actionable insights:
   - Heatmaps to identify key correlations.
   - Network diagrams to visualize relationships between processes.
   - Bar and line charts for summarizing trends.

---

## Contribution
Contributions are welcome! Open an issue or submit a pull request in the repository.

---

## License
This tool is licensed under the [MIT License](LICENSE).

---

## Contact
For questions or support, contact Gregor Posadas at gregorposadas@u.boisestate.edu
