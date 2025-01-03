# WWTP Statistical Framework and Visualization Tool

## Overview

The **WWTP Statistical Framework and Visualization Tool** is a comprehensive Python-based application designed to facilitate advanced statistical analysis and visualization of data from Wastewater Treatment Plants (WWTPs). By transforming complex datasets into actionable insights, this tool empowers WWTP operators, engineers, and researchers to optimize treatment processes, monitor performance, and make informed decisions. Utilizing robust statistical methods and intuitive visualizations, the tool bridges the gap between raw data and meaningful interpretation.

---

## Features

### **1. Advanced Statistical Framework**
- **Bootstrapping for Robust Correlation Analysis:**
  - **Purpose:** Enhances the reliability of correlation estimates by mitigating the effects of outliers and sampling variability.
  - **Method:** Repeatedly resamples the dataset with replacement and computes correlation coefficients across iterations to derive a median correlation matrix.
  
- **Multiple Correlation Methods:**
  - **Pearson Correlation:**
    - **Description:** Measures the linear relationship between two continuous variables.
    - **Use Case:** Ideal for detecting direct, proportional relationships (e.g., flow rate vs. COD).
  
  - **Spearman Correlation:**
    - **Description:** Assesses the monotonic relationship between two variables based on rank.
    - **Use Case:** Suitable for non-linear but consistently increasing or decreasing relationships (e.g., temperature vs. BOD).
  
  - **Kendall Tau Correlation:**
    - **Description:** Evaluates the ordinal association between two variables.
    - **Use Case:** Effective for small sample sizes or data with tied ranks (e.g., NH₄⁺-N vs. COD).

- **Multiple Testing Correction:**
  - **Benjamini-Hochberg False Discovery Rate (FDR):**
    - **Purpose:** Controls the expected proportion of false positives (Type I errors) when conducting multiple hypothesis tests.
    - **Method:** Adjusts p-values to account for the number of comparisons, reducing the likelihood of identifying spurious correlations.

### **2. Comprehensive Visualization Tools**
- **Heatmaps for Correlation Matrices:**
  - **Functionality:** Visualize the strength and direction of correlations between parameters using color-coded grids.
  - **Benefits:** Quickly identify significant relationships and patterns across multiple parameters and processes.

- **Network Diagrams for Parameter Relationships:**
  - **Globally Shared Network Diagrams:**
    - **Description:** Illustrate relationships between parameters that are consistently correlated across all process pairs.
  
  - **Locally Shared Network Diagrams:**
    - **Description:** Highlight parameter correlations specific to individual process pairs, allowing for targeted analysis.

- **Bar and Line Charts for Trend Analysis:**
  - **Bar Charts:**
    - **Use Case:** Summarize correlation coefficients of globally shared parameters across different process pairs.
  
  - **Line Graphs:**
    - **Use Case:** Display trends in correlation coefficients over time or across process sequences, facilitating the detection of temporal patterns.

### **3. User-Friendly Accessibility**
- **Streamlit-Based Web Interface:**
  - **Advantages:** Offers an interactive and responsive platform accessible via web browsers without requiring advanced technical skills.
  
- **Intuitive Workflows for Non-Expert Users:**
  - **Design:** Simplifies complex statistical processes into easy-to-navigate steps, ensuring accessibility for users with varying levels of expertise.

---

## Installation

### Prerequisites

1. **Python Installation:**
   - **Requirement:** Python 3.8 or later.
   - **Download:** [Python Official Website](https://www.python.org/downloads/)

2. **Required Python Libraries:**
   - Install the necessary libraries using `pip`:
     ```bash
     pip install pandas numpy scipy statsmodels plotly matplotlib seaborn networkx streamlit
     ```
   - **Library Descriptions:**
     - `pandas`: Data manipulation and analysis.
     - `numpy`: Numerical computations.
     - `scipy`: Scientific and technical computing, including statistical functions.
     - `statsmodels`: Statistical modeling and hypothesis testing.
     - `plotly`: Interactive visualizations.
     - `matplotlib`: Static plotting.
     - `seaborn`: Statistical data visualization.
     - `networkx`: Network and graph analysis.
     - `streamlit`: Web application framework for data science.

### Setup Steps

1. **Clone the Repository or Copy the Code:**
   - **Clone via Git:**
     ```bash
     git clone https://github.com/yourusername/wwtp-statistical-tool.git
     ```
   - **Or, copy the `app.py` file into your local environment.**

2. **Navigate to the Project Directory:**
   ```bash
   cd wwtp-statistical-tool
   ```

3. **Launch the Application:**
   ```bash
   streamlit run app.py
   ```

4. **Access the Application:**
   - After running the above command, Streamlit will provide a local URL (e.g., `http://localhost:8501`).
   - Open this URL in your web browser to interact with the tool.

5. **Direct Access to Streamlit App**
   - Alternatively, if you just want to access the tool directly on your browser, go to https://wwtp-network-visualization.streamlit.app/.
   - Streamlit automatically "hibernates" apps that have not been used for a while, so you may be prompted to reactivate the app.
---

## How to Use

### **Step 1: Launch the Tool**

1. **Start the Application:**
   - Execute the following command in your terminal:
     ```bash
     streamlit run app.py
     ```
   
2. **Open in Browser:**
   - Navigate to the URL provided by Streamlit (typically `http://localhost:8501`).

### **Step 2: Upload Data**

1. **Prepare Your Data Files:**
   - Ensure each dataset represents a specific process within the WWTP (e.g., Influent, Activated Sludge, Effluent).
   - Each file should be in CSV or Excel format and include a `date` column alongside common parameters (e.g., `flow_rate`, `COD`, `BOD`, `NH4_N`, `temperature`).
   - Three (3) sample files are available in the Git repository for testing purposes, as well as to use for reference in formatting your own input files.

2. **Upload Files:**
   - In the **Upload and Label Files** section, use the file uploader to select and upload multiple CSV or Excel files.
   - Supported formats: `.csv`, `.xlsx`, `.xls`.

3. **Assign Descriptive Labels:**
   - For each uploaded file, provide a meaningful label (e.g., "Influent", "Activated Sludge", "Effluent") to identify the process.

4. **Validation:**
   - The tool will automatically identify common parameters across all uploaded datasets.
   - Ensure that all files contain the `date` column and the same set of parameters for accurate analysis.

### **Step 3: Reorder Processes**

1. **Assign Sequence Order:**
   - In the sidebar, assign an order number to each uploaded process based on their real-life sequence within the WWTP (e.g., Influent → Activated Sludge → Effluent).

2. **Ensure Unique Ordering:**
   - Each process must have a unique order number to maintain the correct sequence for analysis and visualization.

3. **Confirm Sorting:**
   - The tool will sort the datasets based on the assigned order to reflect the actual flow within the treatment plant.

### **Step 4: Select Date Range**

1. **Choose Analysis Period:**
   - Utilize the date range picker to select the start and end dates for your analysis.
   - Ensure that the selected range is covered across all datasets for comprehensive correlation analysis.

2. **Filter Data:**
   - The tool will filter each dataset to include only records within the specified date range.
   - It will also report the number of records retained post-filtering for transparency.

### **Step 5: Generate Visualizations**

1. **Heatmaps:**
   - **Function:** Visualize the strength and direction of correlations between parameters across different processes.
   - **Usage:** Navigate to the **Generate Heatmaps** section and select process pairs to generate.

2. **Network Diagrams:**
   - **Globally Shared Network Diagrams:** Explore consistent correlations across all process pairs.
   - **Locally Shared Network Diagrams:** Highlight process-specific parameter relationships.

3. **Bar and Line Charts:**
   - Summarize correlation trends and highlight globally shared parameters.

4. **Targeted Network Diagrams:**
   - Focus on a specific parameter and process to investigate direct correlations.

### **Step 6: Interpret Outputs**

- **Heatmaps:** Identify strong relationships through color-coded grids.
- **Network Diagrams:** Visualize parameter dependencies using nodes and edges.
- **Bar and Line Charts:** Quantify trends in globally shared parameter relationships.
- **Targeted Network Diagrams:** Investigate specific parameter behaviors for troubleshooting and optimization.

---

## Troubleshooting

### **1. Missing 'date' Column**
- **Issue:** The uploaded file lacks a `date` column.
- **Solution:** Ensure each dataset includes a `date` column with properly formatted dates (`YYYY-MM-DD`).

### **2. Inconsistent Parameter Names**
- **Issue:** Parameters are named differently across datasets (e.g., `NH4_N` vs. `NH4-N`).
- **Solution:** Standardize parameter names before uploading.

### **3. Insufficient Data After Filtering**
- **Issue:** Selecting a narrow date range results in too few records for meaningful analysis.
- **Solution:** Broaden the date range to include more data points.

### **4. No Common Parameters Identified**
- **Issue:** The tool cannot find common parameters across all uploaded datasets.
- **Solution:** Ensure that all datasets share at least one parameter besides the `date` column.

---

## License

This tool is licensed under the [MIT License](LICENSE). You are free to use, modify, and distribute it as long as you include the original license and attribution.

---

## Contact

For questions, support, or suggestions, please reach out to:

**Gregor Posadas**  
Email: [gregorposadas@u.boisestate.edu](mailto:gregorposadas@u.boisestate.edu)  
Feel free to connect via email for any inquiries related to the **WWTP Statistical Framework and Visualization Tool**.
