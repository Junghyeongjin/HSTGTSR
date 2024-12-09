import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import timedelta
from PIL import Image
import base64
import numpy as np
import io

# Function to convert the GIF file into Base64 format
def get_image_file_as_base64(file_path):
    """Convert an image file to a Base64 string."""
    with open(file_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# Set the page layout to wide
st.set_page_config(layout="wide")

# Correct absolute path to your GIF image
gif_path = "data\Sales report.gif"

# Display GIF on the first row
try:
    gif_base64 = get_image_file_as_base64(gif_path)
    gif_html = f"""
    <div style="text-align: center; margin-bottom: 20px;">
        <img src="data:image/gif;base64,{gif_base64}" 
             style="width:800px; height:600px;"  <!-- Adjust size -->
             
    </div>
    """
    st.markdown(gif_html, unsafe_allow_html=True)
except FileNotFoundError:
    st.error("GIF not found. Please check the path.")

# Display text on the second row
st.markdown(
    """
    <div style="text-align: center; margin-top: 10px;">
        <h1 style="font-size:36px; color:black;">Interactive Hansae Target Sales Report</h1>
        <p style="font-size:18px; color:gray;">Contact: Jung Hyeong jin (hjjung@hansae.com)</p>
    </div>
    """, 
    unsafe_allow_html=True
)
    
# Password Protection
password = st.sidebar.text_input("Enter Access Password:", type="password")
if password != "HansaeCJS!":
    st.warning("Please input correct Password.")
    st.stop()

# Function to convert custom date format to datetime with error handling and year adjustment for January
def convert_to_date_fixed(date_str):
    if pd.isna(date_str):  # Check for NaN or None values
        return None
    try:
        parts = str(date_str).split()
        month = parts[0]
        week_number = int(parts[2])  # Extract the numeric part after "Wk"
        year = int(parts[3])
        
        # Map months to numerical values
        month_mapping = {
            "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4,
            "May": 5, "Jun": 6, "Jul": 7, "Aug": 8,
            "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12
        }
        
        # Adjust year for January
        if month == "Jan":
            year += 1

        # Calculate the starting date of the given week in the given month and year
        first_day_of_month = pd.Timestamp(year=year, month=month_mapping[month], day=1)
        start_of_week = first_day_of_month + pd.DateOffset(weeks=week_number - 1)
        
        return start_of_week
    except (IndexError, ValueError, KeyError):
        return None

# Upload CSV Files and enforce UTF-8, UTF-8-SIG, or GB2312 encoding with error handling
uploaded_files = st.sidebar.file_uploader("Upload Sales Report CSV Files - from GreenField", accept_multiple_files=True, type=["csv"])

# Define a consistent column name mapping for normalization
column_name_mapping = {
    "DATE": "DATE",
    "DEPARTMENT NUMBER": "DEPARTMENT NUMBER",
    "DEPARTMENT": "DEPARTMENT",
    "DPCI": "DPCI",
    "SIZE": "SIZE",
    "MFG STYLE": "PID NUMBER",
    "COLOR": "COLOR",
    "ITEM DESCRIPTION": "ITEM DESCRIPTION",
    "REG SALES U": "REG SALES U",
    "PROMO SALES U": "PROMO SALES U",
    "CLEAR SALES U": "CLEAR SALES U",
    "EOH+OT U": "EOH+OT U",
    "REG SALES $": "REG SALES $",
    "PROMO SALES $": "PROMO SALES $",
    "CLEAR SALES $": "CLEAR SALES $",
    "EOH+OT $": "EOH+OT $",
    "GROSS MARGIN $": "GROSS MARGIN $"  # Added Gross Margin $
}

# List of columns that should be numeric
numeric_columns = [
    "COUNT_OF_ROWS", "DEPARTMENT NUMBER", "REG SALES U", "PROMO SALES U", "CLEAR SALES U", "EOH+OT U",
    "REG SALES $", "PROMO SALES $", "CLEAR SALES $", "EOH+OT $", "GROSS MARGIN $"  # Added Gross Margin $
]

# Load data from uploaded files
if uploaded_files:
    data_frames = []
    
    # Progress Bar
    progress_text = st.empty()
    progress_bar = st.progress(0)
    
    for i, file in enumerate(uploaded_files):
        # Attempt to read the file using different encodings
        try:
            df = pd.read_csv(file, encoding='utf-8')
        except UnicodeDecodeError:
            try:
                df = pd.read_csv(file, encoding='utf-8-sig')
            except UnicodeDecodeError:
                try:
                    df = pd.read_csv(file, encoding='gb2312')
                except UnicodeDecodeError:
                    st.error(f"Failed to read file {file.name} with UTF-8, UTF-8-SIG, or GB2312 encoding.")
                    continue  # Skip this file and move to the next

        # Standardize column names: convert to uppercase and apply specific mappings
        df.columns = df.columns.str.upper()
        df = df.rename(columns=column_name_mapping)
        
        # Data Cleaning: Remove any non-numeric characters in dollar columns and convert them to floats
        for col in numeric_columns:
            if col in df.columns:
                # Remove $ symbols, commas, and other non-numeric characters, then convert to numeric
                df[col] = pd.to_numeric(df[col].replace({r'[^\d.-]': ''}, regex=True), errors='coerce')

        # Combine COLOR and PID NUMBER for COLOR (PID NUMBER) level if they exist in columns
        if 'COLOR' in df.columns and 'PID NUMBER' in df.columns:
            df['COLOR (PID NUMBER)'] = df['COLOR'] + " (" + df['PID NUMBER'] + ")"
        
        data_frames.append(df)
        
        # Update the progress bar
        progress_text.text(f"Loading file {i+1} of {len(uploaded_files)}")
        progress_bar.progress((i + 1) / len(uploaded_files))
    
    data = pd.concat(data_frames, ignore_index=True)
    progress_text.text("Files loaded successfully.")
    progress_bar.empty()

    # Ensure 'DATE' column is in the correct format by converting it, if it exists
    if 'DATE' in data.columns:
        # Preserve the full dataset before any filtering
        full_data = data.copy()

        # Flag to check if any January dates were adjusted
        jan_adjustment_made = any(data['DATE'].str.startswith("Jan"))
        
        # Add a progress bar specifically for date conversion
        date_progress = st.progress(0)
        
        # Apply the date conversion to both full_data and data
        full_data['Converted_Date'] = full_data['DATE'].apply(convert_to_date_fixed)
        data['Converted_Date'] = data['DATE'].apply(convert_to_date_fixed)

        # Define the date range for filtering
        if 'Converted_Date' in data.columns:
            min_date, max_date = data['Converted_Date'].min(), data['Converted_Date'].max()
            selected_date_range = st.sidebar.date_input(
                "Select Date Range",
                [min_date, max_date],
                min_value=min_date,
                max_value=max_date
            )

            # Filter data based on the selected date range
            data = data[(data['Converted_Date'] >= pd.Timestamp(selected_date_range[0])) &
                        (data['Converted_Date'] <= pd.Timestamp(selected_date_range[1]))]

        # Check if 'Converted_Date' has been created successfully
        if 'Converted_Date' in data.columns:
            # Inform the user if any January dates were adjusted
            if jan_adjustment_made:
                st.write("Year of January corrected")
            
            # Drop rows where 'Converted_Date' could not be parsed for both data and full_data
            full_data = full_data.dropna(subset=['Converted_Date']).sort_values(by='Converted_Date').reset_index(drop=True)
            data = data.dropna(subset=['Converted_Date']).sort_values(by='Converted_Date').reset_index(drop=True)

            # Create SUM_SELECTED column for full_data and data
            if "SUM_SELECTED" not in full_data.columns:
                full_data['SUM_SELECTED'] = full_data[["REG SALES U", "PROMO SALES U", "CLEAR SALES U"]].sum(axis=1)

            if "SUM_SELECTED" not in data.columns:
                data['SUM_SELECTED'] = data[["REG SALES U", "PROMO SALES U", "CLEAR SALES U"]].sum(axis=1)

            date_progress.empty()
        else:
            st.warning("Converted_Date column could not be created.")
    else:
        st.warning("DATE column not found. Skipping date conversion.")

    # Convert expected numeric columns to numeric values, handling non-numeric data by setting it to NaN
    numeric_columns = ["REG SALES U", "PROMO SALES U", "CLEAR SALES U", "EOH+OT U", "REG SALES $", "PROMO SALES $", "CLEAR SALES $", "EOH+OT $", "GROSS MARGIN $"]
    for col in numeric_columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    
    # Fill NaN values in numeric columns with 0 to avoid issues in summing
    data[numeric_columns] = data[numeric_columns].fillna(0)

    # Sidebar: Filter Type Selection
    filter_type = st.sidebar.selectbox("Select Filter Type", ["DPCI", "PID NUMBER"], index=1)

    if filter_type == "PID NUMBER":
        # Load the PID NUMBER list from the raw data
        pid_number_list = data["PID NUMBER"].dropna().unique().tolist()
        # Autocomplete input for PID NUMBER
        filter_values = st.sidebar.multiselect(
            "Select PID NUMBER(s):", options=pid_number_list, default=[]
        )
    else:
        # Text area input for DPCI (one code per line)
        filter_input = st.sidebar.text_area("Enter DPCI codes (one per line):")
        filter_values = [code.strip() for code in filter_input.split("\n") if code.strip()]

    # Apply the filter if values are provided
    if filter_values:
        data = data[data[filter_type].isin(filter_values)]

    # Aggregation Level Logic
    aggregation_levels = {
        "DPCI": ["DPCI"],
        "COLOR (PID NUMBER)": ["COLOR (PID NUMBER)"],
        "PID NUMBER": ["PID NUMBER"],
        "DEPARTMENT": ["DEPARTMENT"],
    }

    # Main aggregation level selection with horizontal bar format
    selected_level = st.sidebar.select_slider(
        "Select Aggregation Level",
        options=list(aggregation_levels.keys()),
        value="COLOR (PID NUMBER)"  # Default selection
    )

    # Filtered data grouping by the selected aggregation level
    group_columns = aggregation_levels[selected_level]

    # Dropdown for COLOR (PID NUMBER) if applicable
    if selected_level in ["COLOR (PID NUMBER)", "PID NUMBER", "DEPARTMENT"]:
        # Group by COLOR (PID NUMBER) and sort by SUM_SELECTED
        color_grouped = data.groupby("COLOR (PID NUMBER)")["SUM_SELECTED"].sum().sort_values(ascending=False)
        color_options = color_grouped.index.tolist()
        selected_colors = st.sidebar.multiselect(
            "Select COLOR (PID NUMBER):",
            options=color_options,
            format_func=lambda x: f"{x} ({color_grouped[x]:,.0f} units)",
        )
        # Note for COLOR (PID NUMBER) selection
        st.sidebar.markdown(
            """
            <span style="color: orange; font-weight: bold;">Note:</span> 
            This function is limited in recognizing individual colors when GMS colors are duplicated within the same style.
            """,
            unsafe_allow_html=True
        )
        # Filter data based on selected colors
        if selected_colors:
            data = data[data["COLOR (PID NUMBER)"].isin(selected_colors)]

    # Dropdown for DPCI (always visible)
    dpci_grouped = data.groupby("DPCI")["SUM_SELECTED"].sum().sort_values(ascending=False)
    dpci_options = dpci_grouped.index.tolist()
    selected_dpcis = st.sidebar.multiselect(
        "Select DPCI:",
        options=dpci_options,
        format_func=lambda x: f"{x} ({dpci_grouped[x]:,.0f} units)",
    )
    # Filter data based on selected DPCIs
    if selected_dpcis:
        data = data[data["DPCI"].isin(selected_dpcis)]
        
    # Sidebar Expander with custom styling
    with st.sidebar.expander("Purpose of This App 📨", expanded=False):
        st.markdown("""
        The purpose of this app is to enable efficient analysis and forecasting of product trends within specific groups.  

        I created this tool to support all my colleagues in our "3A 본부 😊", helping you uncover valuable insights to strengthen persuasive discussions with our buyers.  

        This application was developed using Streamlit instead of Power BI to offer a more dynamic and interactive approach to analyzing sales data. With customized formulas, it is tailored to address our specific needs.  

        *As a vendor, we don’t “pricing” our products the way most companies do. Instead, we approach it as “costing” to align with the buyer’s perspective. Let’s work together to make our buyer’s overhead cost valuable.*    

        If this app contributes to simplifying your work or even once supports Hansae’s sales success, I would consider it a meaningful achievement.  

        **Please feel free to share any ideas or suggestions for improvement. Together, we can make this tool even better!**  

        **Thank you for your support,**  
        Jung Hyeongjin  
        *December 5th, 2024*  
        hjjung@hansae.com  
        """)



    # Create the first graph with the sum of selected values
    data['SUM_SELECTED'] = data[["REG SALES U", "PROMO SALES U", "CLEAR SALES U"]].sum(axis=1)

    # Grouped data for totals (used for total sum visualizations)
    grouped_data = data.groupby(['Converted_Date'])[numeric_columns + ['SUM_SELECTED']].sum().reset_index()

    # Grouped data for aggregation level (used for Graph 1 & 2)
    if group_columns:
        grouped_data_with_groups = data.groupby(group_columns + ['Converted_Date'])[numeric_columns + ['SUM_SELECTED']].sum().reset_index()
    else:
        grouped_data_with_groups = None  # No grouping columns available
    
    # Determine the loaded date range
    if 'Converted_Date' in data.columns:
        start_date = data['Converted_Date'].min().strftime("%B %d, %Y")
        end_date = data['Converted_Date'].max().strftime("%B %d, %Y")
        loaded_period_text = f"<span style='font-size: 18px; color: #489CD3; font-weight: bold;'>Loaded Period: {start_date} to {end_date}</span>"
    else:
        loaded_period_text = "<span style='font-size: 18px; color: #D9534F;'>Date information is not available.</span>"
    st.divider()

    # Display the loaded period text
    st.markdown(
        f"""
        <div style='text-align: right; font-size:16px; color:orange;'>
            📅 {loaded_period_text}
        </div>
        """, 
        unsafe_allow_html=True
    )
    st.divider()

    # Create two columns for horizontal layout
    col1, col2 = st.columns(2)

    # Add English instructions to an expander in the first column
    with col1:
        with st.expander("Help (English)"):
            st.markdown("""
            - Double click a legend item to isolate a single graph.
            - Click on legend items to toggle visibility of individual traces.
            - Select a region of the graph to zoom into that area.
            - Double click the graph to reset the zoom.
            - Use the toolbar above the graph to download as a PNG or adjust other settings.
            - Use the toolbar in the top-right corner of data tables to download them as a CSV file or search through the content.
            - Some graphs may have multiple Y-axes (e.g., units, $, %).
            - Click the ⋮ button in the top-right corner to use the "Record a screencast" feature and create a video report with voice.
            """)

    # Add Korean instructions to an expander in the second column
    with col2:
        with st.expander("도움말 (Korean)"):
            st.markdown("""
            - 범례(legend)를 더블 클릭하면 단일 그래프만 표시됩니다.
            - 범례 항목을 클릭하여 특정 그래프를 켜거나 끌 수 있습니다.
            - 그래프의 특정 영역을 드래그하여 선택하면 해당 부분만 확대됩니다.
            - 그래프를 더블 클릭하면 초기화됩니다.
            - 그래프 우측 상단의 툴바를 이용해 PNG로 다운로드하거나 기타 설정을 조정할 수 있습니다.
            - 데이터 테이블 우측 상단의 툴바를 이용해 CSV 파일로 다운로드 하거나 내용을 검색 할 수 있습니다.
            - 일부 그래프는 여러 Y축(예: 단위, $, %)을 포함할 수 있습니다.
            - 우측 상단의 ⋮ 버튼을 눌러 "Record a screencast" 기능을 사용 하시어 음성과 함께 레포트를 영상으로 만들 수 있습니다. 
            """)

    st.divider()
            

    #========= Display Line Graph for Graph #1
    col1, col2 = st.columns([1, 8])  # Adjust column width ratio as needed

    with col1:
        image_path = "data\sales2.png"
        try:
            image = Image.open(image_path)
            st.image(image, caption=None, use_container_width=True)
        except FileNotFoundError:
            st.error("Image not found. Please check the path.")

    with col2:
        st.markdown(
            """
            <div style='font-size:36px; font-weight:bold; color:black;'>
                Sales Units
            </div>
            """, 
            unsafe_allow_html=True
        )
        # Title or introductory text
        st.write("Sum of selected sales units over time. REG SALES U, PROMO SALES U, CLEAR SALES U")

    # Conditionally set color based on grouping column
    if grouped_data_with_groups is not None:
        # Calculate total sales for each category to determine the order
        category_totals = grouped_data_with_groups.groupby(group_columns[0])['SUM_SELECTED'].sum().sort_values(ascending=False)
        sorted_categories = category_totals.index.tolist()

        fig = px.line(
            grouped_data_with_groups,
            x='Converted_Date',
            y='SUM_SELECTED',
            color=group_columns[0],
            labels={"SUM_SELECTED": "Sales Units", "Converted_Date": "Date"},
            category_orders={group_columns[0]: sorted_categories}  # Apply custom ordering
        )
    else:
        # No color set for Total Sum (no grouping column)
        fig = px.line(
            grouped_data,
            x='Converted_Date',
            y='SUM_SELECTED',
            labels={"SUM_SELECTED": "Sales Units", "Converted_Date": "Date"}
        )

    fig.update_traces(mode="lines+markers", hovertemplate="%{y}")
    fig.update_xaxes(dtick="M1", tickformat="%b %Y")  # Monthly tick with month and year
    st.plotly_chart(fig, use_container_width=True)

    # Data Preparation for Datatable
    # Use grouped data based on aggregation level if available
    data_for_table = grouped_data_with_groups if grouped_data_with_groups is not None else grouped_data

    # Step 1: Add a column for 'Month' in data_for_table
    data_for_table['Month'] = data_for_table['Converted_Date'].dt.to_period('M')

    # Step 2: Aggregate data by month and line category (based on grouping column)
    if grouped_data_with_groups is not None:
        monthly_data = (
            data_for_table.groupby([group_columns[0], 'Month'])['SUM_SELECTED']
            .sum()
            .reset_index()
            .pivot(index=group_columns[0], columns='Month', values='SUM_SELECTED')
            .fillna(0)
            .astype(int)  # Convert values to integers for better display formatting
        )
    else:
        # Total aggregation without grouping column
        monthly_data = (
            data_for_table.groupby('Month')['SUM_SELECTED']
            .sum()
            .reset_index()
            .pivot(index=None, columns='Month', values='SUM_SELECTED')
            .fillna(0)
            .astype(int)
        )

    # Step 3: Calculate total sales across months and % contribution
    monthly_data['Total'] = monthly_data.sum(axis=1)
    total_sales_sum = monthly_data['Total'].sum()  # Sum of all totals for percentage calculation
    monthly_data['% (portion)'] = monthly_data['Total'] / total_sales_sum

    # Step 4: Sort in ascending order (highest total at the top)
    monthly_data = monthly_data.sort_values(by='Total', ascending=False)

    # Step 5: Handle cases with more than 5 lines
    if len(monthly_data) > 10:
        top_five = monthly_data.iloc[:10]  # Select top 5 rows
        others = pd.DataFrame(monthly_data.iloc[10:].sum(axis=0)).T  # Sum the rest as "Others"
        others.index = ['Others']
        others['% (portion)'] = others['Total'] / total_sales_sum
        monthly_data = pd.concat([top_five, others])  # Concatenate top 5 with "Others"

    # Step 6: Rename month columns to a more readable format
    monthly_data.columns = [col.strftime('%b %Y') if isinstance(col, pd.Period) else col for col in monthly_data.columns]

    # Step 7: Add "Total Sum" row at the bottom
    total_sum_row = monthly_data.sum(numeric_only=True).astype(int)  # Convert to int for formatting
    total_sum_row['% (portion)'] = 1.0  # 100% for the Total Sum row
    monthly_data.loc['Total Sum'] = total_sum_row

    # Function to apply bold styling to the "Total Sum" row
    def highlight_total_sum(row):
        return ['font-weight: bold' if row.name == 'Total Sum' else '' for _ in row]

    # Display Datatable under Graph #1 with formatting
    st.write("Monthly Sales Units(filtered) - Top 10")
    st.dataframe(
        monthly_data.style.format({
            **{col: "{:,.0f}" for col in monthly_data.columns if col not in ["% (portion)"]},  # Format integers with commas
            "% (portion)": "{:.2%}"
        }).apply(highlight_total_sum, axis=1)
    )
    st.divider()

    #======== GRAPH#1-1
    # Display Graph #1-1: Multi-Year Sales Comparison
    col1, col2 = st.columns([1, 8])  # Adjust column width ratio as needed

    with col1:
        image_path = r"data\year.png"
        try:
            image = Image.open(image_path)
            st.image(image, caption=None, use_container_width=True)
        except FileNotFoundError:
            st.error("Image not found. Please check the path.")

    with col2:
        st.markdown(
            """
            <div style='font-size:36px; font-weight:bold; color:black;'>
                Multi-Year Sales Comparison (Over loaded period from raw data)
            </div>
            """, 
            unsafe_allow_html=True
        )
        st.write("""
        Displays all available data across the loaded period, irrespective of the selected date range.
        It compares sales data for the same months across different years. If there is no data spanning more than one year, the graph will not be displayed.
        """)

    # Apply the same filtering logic to full_data
    filtered_full_data = full_data.copy()
    if filter_values:  # Apply filter if values were input
        filtered_full_data = filtered_full_data[filtered_full_data[filter_type].isin(filter_values)]

    # Add 'Year' and 'Month' columns for multi-year comparison
    filtered_full_data['Year'] = filtered_full_data['Converted_Date'].dt.year
    filtered_full_data['Month'] = filtered_full_data['Converted_Date'].dt.month

    # Check if data spans over more than one year
    if filtered_full_data['Year'].nunique() > 1:
        # Aggregate data by year and month
        multi_year_data = filtered_full_data.groupby(['Year', 'Month'])['SUM_SELECTED'].sum().reset_index()

        # Convert month to string for display on x-axis
        multi_year_data['Month_Name'] = pd.to_datetime(multi_year_data['Month'], format='%m').dt.strftime('%b')

        # Plot Multi-Year Sales Comparison
        fig_multi_year = px.line(
            multi_year_data,
            x='Month_Name',
            y='SUM_SELECTED',
            color='Year',
            labels={"SUM_SELECTED": "Sales Units", "Month_Name": "Month"}
        )
        fig_multi_year.update_xaxes(categoryorder='array', categoryarray=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
        fig_multi_year.update_traces(mode="lines+markers", hovertemplate="%{y}")
        st.plotly_chart(fig_multi_year, use_container_width=True)

        # Data Preparation for Multi-Year Data Table
        multi_year_table = multi_year_data.pivot(index='Year', columns='Month_Name', values='SUM_SELECTED').fillna(0).astype(int)

        # Reorder columns by month from Jan to Dec
        month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        multi_year_table = multi_year_table.reindex(columns=month_order)

        # Add total sales column
        multi_year_table['Total'] = multi_year_table.sum(axis=1)

        # Calculate percentage change compared to the previous year
        comparison_rows = {}
        years = sorted(multi_year_table.index.tolist())  # Ensure years are sorted numerically
        for i in range(1, len(years)):  # Start from the second year
            prev_year = years[i - 1]
            curr_year = years[i]
            comparison_rows[curr_year] = ((multi_year_table.loc[curr_year] - multi_year_table.loc[prev_year]) / multi_year_table.loc[prev_year]).fillna(0) * 100

        # Add comparison rows to the table
        for year, comparison in comparison_rows.items():
            # Replace infinite values caused by division by zero with NaN
            comparison.replace([float('inf'), float('-inf')], float('nan'), inplace=True)
            multi_year_table.loc[f"{year} Change (%)"] = comparison.round(2)

        # Sort rows by year (including comparison rows)
        sorted_rows = sorted(
            multi_year_table.index,
            key=lambda x: int(str(x).split(" ")[0]) if str(x).split(" ")[0].isdigit() else int(str(x).split(" ")[0].replace(" Change (%)", ""))
        )
        multi_year_table = multi_year_table.loc[sorted_rows]

        # Add "Total Sum" row at the bottom
        total_sum_row = multi_year_table.loc[~multi_year_table.index.to_series().astype(str).str.contains("Change")].sum(numeric_only=True).astype(int)
        total_sum_row.name = "Total Sum"
        multi_year_table = pd.concat([multi_year_table, pd.DataFrame([total_sum_row], columns=multi_year_table.columns)], axis=0)

        # Copy of the original data for styling purposes
        original_multi_year_table = multi_year_table.copy()

        # Format the data table (convert to strings)
        formatted_multi_year_table = multi_year_table.apply(
            lambda col: [
                f"{value:.2f}%" if "Change (%)" in str(row_name) and pd.notna(value) else
                "n/a" if "Change (%)" in str(row_name) and pd.isna(value) else
                f"{value:,.0f}" if pd.notna(value) else "-"
                for value, row_name in zip(col, multi_year_table.index)
            ],
            axis=0
        )

        # Function to determine if a row represents a year (units)
        def is_year_row(row_name):
            return str(row_name).isdigit()

        # Function to apply styling based on the original numeric data
        def highlight_rows(row):
            if "Change (%)" in str(row.name):
                return [
                    'color: red; font-weight: bold' if isinstance(x, (float, int)) and x > 0 else
                    'color: blue; font-weight: bold' if isinstance(x, (float, int)) and x < 0 else
                    'color: gray; font-weight: bold' if pd.isna(x) else ''  # Handle "n/a" explicitly
                    for x in original_multi_year_table.loc[row.name]
                ]
            if row.name == 'Total Sum':
                return ['font-weight: bold; background-color: #f5f5f5; color: black' for _ in row]
            if is_year_row(row.name):
                return ['background-color: #f5f5f5; color: black' for _ in row]
            return ['' for _ in row]

        # Display the styled and formatted data table
        st.write("Multi-Year Sales Units(filtered) with Yearly Changes")
        st.dataframe(
            pd.DataFrame(formatted_multi_year_table, index=multi_year_table.index, columns=multi_year_table.columns)
            .style.apply(highlight_rows, axis=1)
        )

    st.divider()

    #========= Create an additional graph for "EOH+OT U" as Graph #2
    if "EOH+OT U" in data.columns:
        col1, col2 = st.columns([1, 8])  # Adjust column width ratio as needed

        with col1:
            image_path = r"data\warehouse.png"
            try:
                image = Image.open(image_path)
                st.image(image, caption=None, use_container_width=True)
            except FileNotFoundError:
                st.error("Image not found. Please check the path.")

        with col2:
            st.markdown(
                """
                <div style='font-size:36px; font-weight:bold; color:black;'>
                EOH+OT Units
                </div>
                """, 
                unsafe_allow_html=True
            )
            st.write("Sum of EOH+OT units over time.")

        # Use aggregated data with grouping or default to total data
        if grouped_data_with_groups is not None:
            grouped_data_eoh_ot = grouped_data_with_groups.copy()
        else:
            grouped_data_eoh_ot = grouped_data.copy()

        # Calculate total EOH+OT U for each category to sort the legend
        if grouped_data_with_groups is not None:
            eoh_totals = grouped_data_eoh_ot.groupby(group_columns[0])['EOH+OT U'].sum().sort_values(ascending=False)
            sorted_categories = eoh_totals.index.tolist()
            fig_eoh_ot = px.line(
                grouped_data_eoh_ot,
                x='Converted_Date',
                y='EOH+OT U',
                color=group_columns[0],
                labels={"EOH+OT U": "EOH+OT Units", "Converted_Date": "Date"},
                category_orders={group_columns[0]: sorted_categories}  # Order categories by total EOH+OT U
            )
        else:
            # No grouping column; use total data
            fig_eoh_ot = px.line(
                grouped_data_eoh_ot,
                x='Converted_Date',
                y='EOH+OT U',
                labels={"EOH+OT U": "EOH+OT Units", "Converted_Date": "Date"}
            )

        # Customize the graph
        fig_eoh_ot.update_traces(mode="lines+markers", hovertemplate="%{y}")
        fig_eoh_ot.update_xaxes(dtick="M1", tickformat="%b %Y")  # Monthly ticks with month and year format

        # Display the graph
        st.plotly_chart(fig_eoh_ot, use_container_width=True)

        #========= Data Table for EOH+OT Units
        grouped_data_eoh_ot['Month'] = grouped_data_eoh_ot['Converted_Date'].dt.to_period('M')

        # Aggregate data by month and line category (based on grouping column)
        if grouped_data_with_groups is not None:
            monthly_data = (
                grouped_data_eoh_ot.groupby([group_columns[0], 'Month'])['EOH+OT U']
                .sum()
                .reset_index()
                .pivot(index=group_columns[0], columns='Month', values='EOH+OT U')
                .fillna(0)
                .astype(int)  # Convert values to integers for better display formatting
            )
        else:
            monthly_data = (
                grouped_data_eoh_ot.groupby(['Month'])['EOH+OT U']
                .sum()
                .reset_index()
                .pivot(index='Month', values='EOH+OT U')
                .fillna(0)
                .astype(int)  # Convert values to integers for better display formatting
            )

        # Calculate total EOH+OT U across months and % contribution
        monthly_data['Total'] = monthly_data.sum(axis=1)
        total_units_sum = monthly_data['Total'].sum()  # Sum of all totals for percentage calculation
        monthly_data['% (portion)'] = monthly_data['Total'] / total_units_sum

        # Sort in descending order by 'Total'
        monthly_data = monthly_data.sort_values(by='Total', ascending=False)

        # Handle cases with more than 10 lines
        if len(monthly_data) > 10:
            top_five = monthly_data.iloc[:10]  # Select top 10 rows
            others = pd.DataFrame(monthly_data.iloc[10:].sum(axis=0)).T  # Sum the rest as "Others"
            others.index = ['Others']
            others['% (portion)'] = others['Total'] / total_units_sum
            monthly_data = pd.concat([top_five, others])  # Concatenate top 10 with "Others"

        # Rename month columns to a more readable format
        monthly_data.columns = [col.strftime('%b %Y') if isinstance(col, pd.Period) else col for col in monthly_data.columns]

        # Add "Total Sum" row at the bottom
        total_sum_row = monthly_data.sum(numeric_only=True).astype(int)  # Convert to int for formatting
        total_sum_row['% (portion)'] = 1.0  # 100% for the Total Sum row
        monthly_data.loc['Total Sum'] = total_sum_row

        # Function to apply bold styling to the "Total Sum" row
        def highlight_total_sum(row):
            return ['font-weight: bold' if row.name == 'Total Sum' else '' for _ in row]

        # Display Datatable under Graph #2 with formatting
        st.write("Monthly EOH+OT Units(filtered) - Top 10")
        st.dataframe(
            monthly_data.style.format({
                **{col: "{:,.0f}" for col in monthly_data.columns if col not in ["% (portion)"]},  # Format integers with commas
                "% (portion)": "{:.2%}"
            }).apply(highlight_total_sum, axis=1)
        )

        st.divider()


        #======================================= Create Graph #3 for Sales % (Using Total Sum of Filtered Data)
        col1, col2 = st.columns([1, 8])  # Adjust column width ratio as needed

        with col1:
            image_path = r"data\interest-rate.png"
            try:
                image = Image.open(image_path)
                st.image(image, caption=None, use_container_width=True)
            except FileNotFoundError:
                st.error("Image not found. Please check the path.")

        with col2:
            st.markdown(
                """
                <div style='font-size:36px; font-weight:bold; color:black;'>
                Sales % (Compare to EOH+OT U)
                </div>
                """, 
                unsafe_allow_html=True
            )
            st.write("Sum of REG SALES U, PROMO SALES U, CLEAR SALES U / Sum of REG SALES U, PROMO SALES U, CLEAR SALES U + EOH+OT U")

        # Calculate Sales % using total sum of filtered data
        grouped_data['Total_Units'] = grouped_data['SUM_SELECTED'] + grouped_data['EOH+OT U']
        grouped_data['Sales %'] = (grouped_data['SUM_SELECTED'] / grouped_data['Total_Units']) * 100

        # Create figure for Sales % visualization
        fig_graph_3 = go.Figure()

        # Add EOH+OT U fill (Default color with opacity)
        fig_graph_3.add_trace(go.Scatter(
            x=grouped_data['Converted_Date'],
            y=grouped_data['EOH+OT U'],
            fill='tozeroy',
            mode='none',
            name='EOH+OT U',
            opacity=0.3
        ))

        # Add Sales Units line (Red)
        fig_graph_3.add_trace(go.Scatter(
            x=grouped_data['Converted_Date'],
            y=grouped_data['SUM_SELECTED'],
            mode='lines+markers',
            name='Sales Units (SUM_SELECTED)',
            line=dict(color='red')  # Set line color to red
        ))

        # Add Sales % line (Blue)
        fig_graph_3.add_trace(go.Scatter(
            x=grouped_data['Converted_Date'],
            y=grouped_data['Sales %'],
            mode='lines+markers',
            name='Sales %',
            yaxis="y2",
            line=dict(dash='dash', color='blue')  # Set line color to blue
        ))

        # Update layout with dual y-axes
        fig_graph_3.update_layout(
            xaxis_title="Date",
            yaxis_title="Sales Units",
            yaxis2=dict(
                title="Sales %",
                overlaying="y",
                side="right"
            ),
            xaxis=dict(dtick="M1", tickformat="%b %Y")
        )
        st.plotly_chart(fig_graph_3, use_container_width=True)

        # Prepare Data Table for Graph #3
        # Step 1: Add a column for 'Month' in grouped_data
        grouped_data['Month'] = grouped_data['Converted_Date'].dt.to_period('M')

        # Step 2: Aggregate data by month
        monthly_sales_data = grouped_data.groupby('Month').agg(
            Sales_Units=("SUM_SELECTED", 'sum'),
            EOH_OT_Units=("EOH+OT U", 'sum')  # Corrected column name
        ).reset_index()

        # Ensure all months in the range are present
        all_months = pd.period_range(start=grouped_data['Month'].min(), end=grouped_data['Month'].max(), freq='M')
        monthly_sales_data = monthly_sales_data.set_index('Month').reindex(all_months, fill_value=0).reset_index()
        monthly_sales_data.rename(columns={'index': 'Month'}, inplace=True)

        # Step 3: Calculate total units and Sales % for each month
        monthly_sales_data['Total_Units'] = monthly_sales_data['Sales_Units'] + monthly_sales_data['EOH_OT_Units']
        monthly_sales_data['Sales %'] = (
            (monthly_sales_data['Sales_Units'] / monthly_sales_data['Total_Units']) * 100
        ).fillna(0)  # Fill NaN with 0 for cases where Total_Units is 0

        # Step 4: Rename 'Month' to a string format for display
        monthly_sales_data['Month'] = monthly_sales_data['Month'].dt.strftime('%b %Y')

        # Step 5: Transform the table so rows are metrics and columns are months
        pivoted_data = monthly_sales_data.set_index('Month').T  # Transpose the data
        pivoted_data.index.name = 'Metrics'  # Set row index as 'Metrics'

        # Reorder rows to match the specified order
        pivoted_data = pivoted_data.reindex(['Total_Units', 'EOH_OT_Units', 'Sales_Units', 'Sales %'])

        # Step 6: Add a "Total Sum" column for the data
        total_units_sum = int(monthly_sales_data['Total_Units'].sum())  # Ensure integer format
        eoh_ot_units_sum = int(monthly_sales_data['EOH_OT_Units'].sum())  # Ensure integer format
        sales_units_sum = int(monthly_sales_data['Sales_Units'].sum())  # Ensure integer format
        sales_percent_sum = (sales_units_sum / total_units_sum * 100) if total_units_sum != 0 else 0  # Avoid division by 0

        # Add "Total Sum" as a final row
        pivoted_data['Total Sum'] = [
            f"{total_units_sum:,}",
            f"{eoh_ot_units_sum:,}",
            f"{sales_units_sum:,}",
            f"{sales_percent_sum:.2f}%"  # Percentage with two decimals
        ]

        # Format all monthly columns
        for col in pivoted_data.columns[:-1]:  # Skip "Total Sum"
            pivoted_data[col] = pivoted_data[col].apply(
                lambda x: f"{x:,.0f}" if isinstance(x, (int, float)) and not pd.isna(x) and x == int(x) 
                else f"{x:.2f}%" if isinstance(x, float) and not pd.isna(x) 
                else "0" if pd.isna(x) 
                else x
            )

        # Highlight the "Total Sum" column
        def highlight_total_sum(col):
            return ['font-weight: bold; background-color: #f5f5f5; color: black;' if col.name == 'Total Sum' else '' for _ in col]

        # Display the Data Table
        st.write("Monthly Data Table for Sales % (Compared to EOH+OT Units)")
        st.dataframe(
            pivoted_data.style.apply(highlight_total_sum, axis=1)
        )

        st.divider()
        #====================Gross Margin part====================
        # Section: Gross Margin Introduction
        if "REG SALES $" in grouped_data.columns and "PROMO SALES $" in grouped_data.columns and "CLEAR SALES $" in grouped_data.columns:
            col1, col2 = st.columns([1, 8])  # Adjust column width ratio as needed

            with col1:
                image_path = r"data\margin.png"
                try:
                    image = Image.open(image_path)
                    st.image(image, caption=None, use_container_width=True)
                except FileNotFoundError:
                    st.error("Image not found. Please check the path.")

            with col2:
                st.markdown(
                    """
                    <div style='font-size:36px; font-weight:bold; color:black;'>
                        Gross Margin
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
                st.write("Gross Margin: Total sales less product costs, or costs directly tied to the item sold in stores or online "
                        "(which includes cost of goods sold and other costs of sales). These costs include raw material charges, "
                        "freight, and more transportation and sourcing cost.")

            # Add calculated Total Sales $ to the grouped data
            grouped_data["Total Sales $"] = grouped_data[["REG SALES $", "PROMO SALES $", "CLEAR SALES $"]].sum(axis=1)

            # Gross Margin $ Line Graph
            fig_gross_margin_dollar = px.line(
                grouped_data,
                x='Converted_Date',
                y='GROSS MARGIN $',
                title="Gross Margin $ Over Time",
                labels={"GROSS MARGIN $": "Gross Margin ($)", "Converted_Date": "Date"},
                template="plotly_white"
            )
            # Gross Margin % Line Graph
            grouped_data["Gross Margin %"] = (grouped_data["GROSS MARGIN $"] / grouped_data["Total Sales $"]) * 100
            fig_gross_margin_percent = px.line(
                grouped_data,
                x='Converted_Date',
                y='Gross Margin %',
                title="Gross Margin % Over Time",
                labels={"Gross Margin %": "Gross Margin (%)", "Converted_Date": "Date"},
                template="plotly_white"
            )
            # Monthly Summary Preparation
            grouped_data['Month'] = grouped_data['Converted_Date'].dt.to_period('M')
            monthly_summary = grouped_data.groupby('Month').agg(
                Sales_Amount=("Total Sales $", 'sum'),
                Gross_Margin_Dollar=('GROSS MARGIN $', 'sum'),
                Sales_Units=('SUM_SELECTED', 'sum')
            ).reset_index()

            # Calculate Average Selling Price (ASP)
            monthly_summary['ASP'] = monthly_summary['Sales_Amount'] / monthly_summary['Sales_Units']

            # Add Gross Margin % and anomalies
            monthly_summary['Gross Margin %'] = (monthly_summary['Gross_Margin_Dollar'] / monthly_summary['Sales_Amount']) * 100
            monthly_summary['Anomaly'] = monthly_summary['Gross_Margin_Dollar'] > monthly_summary['Sales_Amount']
            monthly_summary['Month'] = monthly_summary['Month'].dt.strftime('%b %Y')

            # Separate normal and anomalous months
            normal_months = monthly_summary[~monthly_summary['Anomaly']]
            anomalous_months = monthly_summary[monthly_summary['Anomaly']]

            # Initialize figure
            fig = go.Figure()

            # Normal months: Sales Amount (Orange) backward, Gross Margin (Blue) forward
            fig.add_trace(go.Bar(
                x=normal_months['Month'],
                y=normal_months['Sales_Amount'],
                name='Total Sales $ (Normal)',
                marker_color='orange',
                opacity=0.8,
                yaxis='y'
            ))

            fig.add_trace(go.Bar(
                x=normal_months['Month'],
                y=normal_months['Gross_Margin_Dollar'],
                name='Gross Margin $ (Normal)',
                marker_color='blue',
                opacity=1,
                yaxis='y'
            ))

            # Anomalous months: Gross Margin (Blue) backward, Sales Amount (Orange) forward
            fig.add_trace(go.Bar(
                x=anomalous_months['Month'],
                y=anomalous_months['Gross_Margin_Dollar'],
                name='Gross Margin $ (Anomaly)',
                marker_color='blue',
                opacity=0.8,
                yaxis='y'
            ))

            fig.add_trace(go.Bar(
                x=anomalous_months['Month'],
                y=anomalous_months['Sales_Amount'],
                name='Total Sales $ (Anomaly)',
                marker_color='orange',
                opacity=1,
                yaxis='y'
            ))

            # Add Sales Units as a line on the secondary Y-axis
            fig.add_trace(go.Scatter(
                x=monthly_summary['Month'],
                y=monthly_summary['Sales_Units'],
                name='Sales Units',
                mode='lines+markers',
                line=dict(color='green', width=2),
                yaxis='y2'
            ))

            # Annotate Gross Margin % and ASP
            for i, row in monthly_summary.iterrows():
                # Annotate Gross Margin %
                fig.add_annotation(
                    x=row['Month'],
                    y=row['Sales_Units'] + 10,  # Above the Sales Units line
                    text=f"{row['Gross Margin %']:.2f}%",
                    showarrow=False,
                    font=dict(size=12, color="white"),  # White text
                    align="center",
                    bgcolor="black"
                )

                # Annotate ASP
                fig.add_annotation(
                    x=row['Month'],
                    y=row['Gross_Margin_Dollar'] + row['Sales_Amount'] + 10,  # Add offset for better readability
                    text=f"ASP: ${row['ASP']:.2f}",
                    showarrow=False,
                    font=dict(size=12, color="white"),  # Black text
                    align="center",
                    bgcolor="black"

                )

            # Warning Annotations for Anomalies
            for i, row in anomalous_months.iterrows():
                fig.add_annotation(
                    x=row['Month'],
                    y=row['Gross_Margin_Dollar'],  # Position warning near the Gross Margin
                    text="⚠️",
                    showarrow=False,
                    font=dict(size=16, color="red"),  # Red warning indicator
                    align="center"
                )

            # Update layout for dual Y-axis
            fig.update_layout(
                barmode='overlay',  # Overlay bars
                title="Monthly Total Sales $, Gross Margin $, and Sales Units with Anomalies",
                xaxis=dict(title="Month"),
                yaxis=dict(
                    title="Amount ($)",
                    titlefont=dict(color="black"),
                    tickfont=dict(color="black"),
                ),
                yaxis2=dict(
                    title="Sales Units",
                    titlefont=dict(color="green"),
                    tickfont=dict(color="green"),
                    overlaying="y",
                    side="right"
                ),
                legend=dict(title="Legend"),
                template="plotly_white"
            )

            # Display the chart
            st.plotly_chart(fig, use_container_width=True)
            
            # Check the structure of monthly_summary
            if 'Month' in monthly_summary.columns:
                # Ensure 'Month' is a string
                monthly_summary['Month'] = monthly_summary['Month'].astype(str)

                # Ensure numeric columns are properly formatted
                numeric_columns = ['Sales_Amount', 'Gross_Margin_Dollar', 'Sales_Units', 'ASP', 'Gross Margin %']
                for col in numeric_columns:
                    if col in monthly_summary.columns:
                        monthly_summary[col] = pd.to_numeric(monthly_summary[col], errors='coerce').fillna(0)

                # Add "Total Sum" row, ensuring no KeyErrors
                total_sum_row = {
                    'Month': 'Total',
                    'Sales_Amount': int(monthly_summary['Sales_Amount'].sum()),
                    'Gross_Margin_Dollar': int(monthly_summary['Gross_Margin_Dollar'].sum()),
                    'Sales_Units': int(monthly_summary['Sales_Units'].sum()),
                    'ASP': monthly_summary['Sales_Amount'].sum() / monthly_summary['Sales_Units'].sum()
                    if monthly_summary['Sales_Units'].sum() != 0 else 0,
                    'Gross Margin %': (monthly_summary['Gross_Margin_Dollar'].sum() /
                                    monthly_summary['Sales_Amount'].sum()) * 100
                    if monthly_summary['Sales_Amount'].sum() != 0 else 0
                }
                monthly_summary = pd.concat(
                    [monthly_summary, pd.DataFrame([total_sum_row])],
                    ignore_index=True
                )

                # Pivot the data
                pivoted_summary = monthly_summary.set_index('Month').T

                # Reorder rows and rename for display clarity
                pivoted_summary = pivoted_summary.reindex(
                    ['Sales_Amount', 'Gross_Margin_Dollar', 'Sales_Units', 'ASP', 'Gross Margin %']
                )
                pivoted_summary.rename(index={
                    'Sales_Amount': 'Total Sales ($)',
                    'Gross_Margin_Dollar': 'Gross Margin ($)',
                    'Sales_Units': 'Sales Units',
                    'ASP': 'Average Selling Price (ASP)',
                    'Gross Margin %': 'Gross Margin (%)'
                }, inplace=True)

                # Apply row-specific formatting
                def format_row(value, row_name):
                    if row_name in ['Total Sales ($)', 'Gross Margin ($)']:
                        return f"${value:,.0f}"  # Integer with $
                    elif row_name == 'Sales Units':
                        return f"{int(value):,}"  # Integer with pcs
                    elif row_name == 'Average Selling Price (ASP)':
                        return f"${value:,.2f}"  # Float with $ and 2 decimals
                    elif row_name == 'Gross Margin (%)':
                        return f"{value:.2f}%"  # Float with % and 2 decimals
                    else:
                        return value  # Default

                # Apply formatting to each row in pivoted_summary
                formatted_summary = pivoted_summary.apply(
                    lambda col: [format_row(value, row_name) for value, row_name in zip(col, pivoted_summary.index)]
                )

                # Display the final table
                st.write("Monthly Gross Margin $(filtered)")
                st.dataframe(formatted_summary)
            else:
                st.error("The 'Month' column is missing in the DataFrame.")


        st.divider()
        #=====================================Graph #4 Forecasting=====================================
        def calculate_total_eoh_ot(data):
            """
            Calculate the total EOH+OT U across all rows, regardless of aggregation level.

            Parameters:
            - data (DataFrame): The full dataset with 'Converted_Date' and 'EOH+OT U'.

            Returns:
            - DataFrame: Aggregated total EOH+OT U over time.
            """
            if 'Converted_Date' in data.columns and 'EOH+OT U' in data.columns:
                # Aggregate EOH+OT U by Converted_Date
                total_eoh_ot = (
                    data.groupby('Converted_Date', as_index=False)['EOH+OT U']
                    .sum()
                    .sort_values(by='Converted_Date')
                )
                return total_eoh_ot
            else:
                raise ValueError("Columns 'Converted_Date' and 'EOH+OT U' are required in the dataset.")

        # Display formula description for forecasting
        col1, col2 = st.columns([1, 8])  # Adjust column width ratio as needed

        with col1:
            image_path = r"data\forecasting.png"
            try:
                image = Image.open(image_path)
                st.image(image, caption=None, use_container_width=True)
            except FileNotFoundError:
                st.error("Image not found. Please check the path.")

        with col2:
            st.markdown(
                """
                <div style='font-size:36px; font-weight:bold; color:black;'>
                Sales & EOH+OT U Forecasting
                </div>
                """, 
                unsafe_allow_html=True
            )
            st.write("""
            **Forecasted Sales Quantity Calculation Explanation:**
            1. You can calculate the forecasted sales quantity using:
            - The average weekly sales from a default period of the last 3 months.
            - A user-specified date range, independent of the selected date range.
            2. Additionally, you can apply an increase or decrease percentage to adjust the forecasted sales quantity dynamically.
            """)

        # Set colors for consistency
        sales_qty_color = "red"
        eoh_ot_color = "skyblue"  # For historical EOH+OT U
        forecast_eoh_ot_fill_color = "lightskyblue"  # For forecasted EOH+OT U with translucent color
        sales_pct_color = "blue"  # Sales % should be red as per instruction
      
        # Create three columns with equal width
        col1, col2, col3 = st.columns(3)
        # Column 1: Forecasting period, date range, and sales adjustment
        with col1:
            st.subheader("Forecasting Settings")
            # Allow manual input for the forecasting period in months
            forecast_months = st.number_input("Enter Forecasting Period (in months):", min_value=1, max_value=24, value=3)
            forecast_weeks = forecast_months * 4  # Approximate weeks in the forecast period

            # Apply the same filtering logic to the full_data for calculating weekly sales
            filtered_full_data = full_data.copy()
            if filter_values:  # Apply filter if values were input
                filtered_full_data = filtered_full_data[filtered_full_data[filter_type].isin(filter_values)]

            # Allow user to select custom sales period or use default 3 months
            st.write("Select period for calculating average weekly sales:")
            start_date = pd.to_datetime(
                st.date_input("Start date", value=pd.Timestamp(filtered_full_data['Converted_Date'].max() - pd.DateOffset(months=3)))
            )
            end_date = pd.to_datetime(
                st.date_input("End date", value=pd.Timestamp(filtered_full_data['Converted_Date'].max()))
            )

            # Filter data for the selected period from the filtered dataset
            selected_period_data = filtered_full_data[(filtered_full_data['Converted_Date'] >= start_date) & (filtered_full_data['Converted_Date'] <= end_date)]

            # Calculate average weekly sales
            if not selected_period_data.empty:
                total_sales = selected_period_data['SUM_SELECTED'].sum()
                num_weeks = len(selected_period_data['Converted_Date'].dt.to_period('W').unique())
                average_weekly_sales = total_sales / num_weeks

                # Prepare a detailed data table
                weekly_sales_table = (
                    selected_period_data.groupby(selected_period_data['Converted_Date'].dt.to_period('W'))['SUM_SELECTED']
                    .sum()
                    .reset_index()
                    .rename(columns={'Converted_Date': 'Week', 'SUM_SELECTED': 'Sales'})
                )
                weekly_sales_table['Week'] = weekly_sales_table['Week'].astype(str)  # Convert week to string for display
            else:
                st.warning("No data available for the selected period.")
                total_sales = 0
                num_weeks = 0
                average_weekly_sales = 0
                weekly_sales_table = None

            # Adjust sales based on user-defined percentage
            sales_adjustment_percentage = st.number_input(
                "Adjust sales by percentage (%)", 
                min_value=-100.0,  # Float type
                max_value=100.0,   # Float type
                value=0.0          # Float type
            )
            adjusted_average_weekly_sales = average_weekly_sales * (1 + sales_adjustment_percentage / 100)

            # Generate forecast sales quantities based on the adjusted average
            forecast_weeks = forecast_months * 4  # Ensure forecast_weeks is defined earlier in the code
            forecasted_sales_qty = [adjusted_average_weekly_sales] * forecast_weeks

            # "Show Detail" button to display detailed calculation
            with st.expander("Show Details of Forecasting Sales Quantity"):
                st.markdown("### Detailed Calculation of Forecasted Sales Quantity")
                
                if not selected_period_data.empty:
                    st.markdown(
                        f"""
                        - **Total Sales**: {total_sales:,.0f} units  
                        - **Number of Counted Weeks**: {num_weeks}  
                        - **Average Weekly Sales**: {average_weekly_sales:,.2f} units  
                        """
                    )
                    st.markdown("#### Weekly Sales Breakdown:")
                    st.dataframe(weekly_sales_table.style.format({'Sales': '{:,.0f}'}))
                else:
                    st.warning("No data available for detailed calculation.")
                    
            # Forecasted EOH+OT U calculation
            total_eoh_ot = calculate_total_eoh_ot(data)
            last_eoh_ot = total_eoh_ot['EOH+OT U'].iloc[-1]
            forecasted_eoh_ot = [last_eoh_ot]

            # Starting date for forecasting - connect from the last historical date directly
            last_date = grouped_data['Converted_Date'].max()
            forecast_dates = pd.date_range(start=last_date, periods=forecast_weeks, freq='W')
        with col2, col3:
            monthly_stock_inputs = []

            # Loop through each forecast month
            for month_idx in range(forecast_months):
                # Define the start and end week for this month
                start_idx = month_idx * 4
                end_idx = min((month_idx + 1) * 4 - 1, len(forecast_dates) - 1)  # Avoid index out of range
                month_start_date = forecast_dates[start_idx]
                month_end_date = forecast_dates[end_idx]

                # Format label for the month
                label = f"Month {month_idx + 1} ({month_start_date.strftime('%b %Y')} ~ {month_end_date.strftime('%b %Y')})"

                # Use separate `with` blocks for col2 and col3 based on the month index
                if month_idx < 8:
                    with col2:
                        stock_input = st.number_input(f"Input Stock for {label}", min_value=0, value=0)
                else:
                    with col3:
                        stock_input = st.number_input(f"Input Stock for {label}", min_value=0, value=0)
                
                # Append the stock input value to the list
                monthly_stock_inputs.append(stock_input)


        # Apply monthly stock inputs only once per month
        forecasted_eoh_ot = [last_eoh_ot]
        for week_index in range(forecast_weeks):
            # Determine the current month index
            current_month_idx = week_index // 4
            if current_month_idx < len(monthly_stock_inputs):
                # Apply the stock input at the first week of each month
                stock_to_apply = monthly_stock_inputs[current_month_idx] if week_index % 4 == 0 else 0
            else:
                stock_to_apply = 0

            # Calculate new EOH+OT U for each week using the forecasted sales and monthly stock input
            new_eoh_ot = forecasted_eoh_ot[-1] + stock_to_apply - forecasted_sales_qty[week_index]
            forecasted_eoh_ot.append(new_eoh_ot)

        # Sales % Calculation
        last_sales_pct = grouped_data['Sales %'].iloc[-1]
        forecasted_sales_pct = [last_sales_pct]  # Initialize with last actual value

        for qty, eoh in zip(forecasted_sales_qty, forecasted_eoh_ot[1:]):
            forecasted_pct = (qty / (qty + eoh)) * 100
            forecasted_sales_pct.append(forecasted_pct)

        # Initialize variables to store the depletion and negative stock week information
        depletion_found = False
        negative_stock_found = False
        depletion_date = None
        negative_stock_date = None

        # Calculate the intersection week and the first negative stock week
        for i in range(len(forecasted_sales_qty)):
            # Intersection point where forecasted sales meets or exceeds forecasted EOH+OT U
            if not depletion_found and forecasted_eoh_ot[i] <= forecasted_sales_qty[i]:
                depletion_date = forecast_dates[i] - timedelta(weeks=1)  # Adjust result by 1 week earlier
                depletion_found = True  # Mark that we found the depletion point

            # Point where forecasted EOH+OT U turns negative
            if not negative_stock_found and forecasted_eoh_ot[i] < 0:
                negative_stock_date = forecast_dates[i] - timedelta(weeks=1)  # Adjust result by 1 week earlier
                negative_stock_found = True  # Mark that we found the first negative stock point
        # Plotting Graph #4 with forecasted sales and EOH+OT U levels
        fig_graph_4 = go.Figure()

        # Historical EOH+OT U (EOH+OT Fill Added First for Background)
        fig_graph_4.add_trace(go.Scatter(
            x=total_eoh_ot['Converted_Date'],
            y=total_eoh_ot['EOH+OT U'],
            mode='none',  # No markers/lines, only fill
            name='EOH+OT U (Historical Fill)',
            fill='tozeroy',
            fillcolor='lightskyblue',
            opacity=0.3
        ))

        # Historical Sales Units
        fig_graph_4.add_trace(go.Scatter(
            x=grouped_data['Converted_Date'],
            y=grouped_data['SUM_SELECTED'],
            mode='lines+markers',
            name='Sales Units (Historical)',
            line=dict(color=sales_qty_color)
        ))

        # Historical EOH+OT U (Line)
        fig_graph_4.add_trace(go.Scatter(
            x=total_eoh_ot['Converted_Date'],
            y=total_eoh_ot['EOH+OT U'],
            mode='lines+markers',
            name='EOH+OT U (Historical Line)',
            line=dict(color=eoh_ot_color)
        ))

        # Forecasted EOH+OT U
        fig_graph_4.add_trace(go.Scatter(
            x=[last_date] + list(forecast_dates),
            y=[last_eoh_ot] + forecasted_eoh_ot[1:],
            mode='lines+markers',
            name='Forecasted EOH+OT U',
            line=dict(dash="dash", color="lightskyblue"),
            fill='tozeroy',
            fillcolor='lightblue',
            opacity=0.1
        ))

        # Forecasted Sales Quantity
        fig_graph_4.add_trace(go.Scatter(
            x=[last_date] + list(forecast_dates),
            y=[grouped_data['SUM_SELECTED'].iloc[-1]] + forecasted_sales_qty,
            mode='lines+markers',
            name='Forecasted Sales Qty',
            line=dict(dash="dash", color="gold")
        ))


        # Add Annotations for Inventory Depletion Point and Negative Stock Point
        annotations = []

        # Ensure depletion_date is valid and exists in forecast_dates
        if depletion_found and depletion_date in forecast_dates:
            depletion_index = forecast_dates.tolist().index(depletion_date)
            depletion_y = forecasted_eoh_ot[depletion_index]

            annotations.append(dict(
                x=depletion_date,  # Use the valid depletion date
                y=depletion_y,  # Align annotation with the EOH+OT U line
                text="📉 Inventory Depletion Point",
                showarrow=True,
                arrowhead=2,
                ax=20,
                ay=-40,
                font=dict(color="white"),
                bgcolor="black"
            ))
        else:
            st.warning("Depletion date is not within the forecast dates range.")

        # Ensure negative_stock_date is valid and exists in forecast_dates
        if negative_stock_found and negative_stock_date in forecast_dates:
            negative_stock_index = forecast_dates.tolist().index(negative_stock_date)
            negative_stock_y = forecasted_eoh_ot[negative_stock_index]

            annotations.append(dict(
                x=negative_stock_date,  # Use the valid negative stock date
                y=negative_stock_y,  # Align annotation with the EOH+OT U line
                text="🚨 Negative Stock Point",
                showarrow=True,
                arrowhead=2,
                ax=20,
                ay=-40,
                font=dict(color="white"),
                bgcolor="red"
            ))
        else:
            st.warning("Negative stock date is not within the forecast dates range.")

        # Update layout and add annotations
        fig_graph_4.update_layout(
            xaxis_title="Date",
            yaxis_title="Sales Units",
            yaxis2=dict(
                title="Sales %",
                overlaying="y",
                side="right",
                range=[0, 100]
            ),
            xaxis=dict(dtick="M1", tickformat="%b %Y"),
            annotations=annotations
        )

        # Display the graph
        st.plotly_chart(fig_graph_4, use_container_width=True)

        # Display explanatory note and dynamically include the depletion week information in styled format
        depletion_week_text = "<span style='color: #4B9CD3; font-size:16px;'>No intersection point within the forecasted period.</span>"
        negative_stock_week_text = "<span style='color: #D9534F; font-size:16px;'>No negative stock forecasted within the forecasted period.</span>"

        # Helper function to determine the week of the month
        def week_of_month(date):
            first_day = date.replace(day=1)
            adjusted_day = date.day + first_day.weekday()  # Adjust based on the weekday of the first day of the month
            return (adjusted_day - 1) // 7 + 1

        # Flags to ensure we capture only the first occurrence of each event
        depletion_found = False
        negative_stock_found = False

        # Adjust dates for depletion and negative stock points
        for i in range(len(forecasted_sales_qty)):
            # Intersection point where forecasted sales meets or exceeds forecasted EOH+OT U
            if not depletion_found and forecasted_eoh_ot[i] <= forecasted_sales_qty[i]:
                depletion_date = forecast_dates[i] - timedelta(weeks=1)  # Adjust result by 1 week earlier
                week_num = week_of_month(depletion_date)
                month_name = depletion_date.strftime("%B")
                year = depletion_date.strftime("%Y")

                # Format week as "1st", "2nd", etc.
                week_suffix = ["th", "st", "nd", "rd", "th"][min(week_num, 4)]
                depletion_week_text = f"<span style='color: #4B9CD3; font-size:18px;'>📉 Estimated Inventory Depletion Week: {week_num}{week_suffix} week of {month_name} {year}</span>"
                depletion_found = True  # Mark that we found the depletion point

            # Point where forecasted EOH+OT U turns negative
            if not negative_stock_found and forecasted_eoh_ot[i] < 0:
                negative_stock_date = forecast_dates[i] - timedelta(weeks=1)  # Adjust result by 1 week earlier
                week_num_neg = week_of_month(negative_stock_date)
                month_name_neg = negative_stock_date.strftime("%B")
                year_neg = negative_stock_date.strftime("%Y")

                # Format week as "1st", "2nd", etc.
                week_suffix_neg = ["th", "st", "nd", "rd", "th"][min(week_num_neg, 4)]
                negative_stock_week_text = f"<span style='color: #D9534F; font-size:18px;'>🚨 Forecasted Negative Stock Week: {week_num_neg}{week_suffix_neg} week of {month_name_neg} {year_neg}</span>"
                negative_stock_found = True  # Mark that we found the first negative stock point
        
        # Prepare the data for the table======================
        forecast_table_data = {
            "Week": [],
            "Sales (Units)": [],
            "EOH+OT U (Units)": [],
            "Balance (EOH+OT - Sales)": []
        }

        # Include last 3 weeks of historical data
        historical_weeks = grouped_data.tail(3)

        for index, row in historical_weeks.iterrows():
            # Calculate the week number within the month
            start_of_month = row["Converted_Date"].replace(day=1)
            week_number = ((row["Converted_Date"] - start_of_month).days // 7) + 1
            week_label = f"{row['Converted_Date'].strftime('%b')} wk {week_number} {row['Converted_Date'].year}"
            
            forecast_table_data["Week"].append(week_label)
            forecast_table_data["Sales (Units)"].append(row["SUM_SELECTED"])
            forecast_table_data["EOH+OT U (Units)"].append(total_eoh_ot.loc[index, "EOH+OT U"])
            forecast_table_data["Balance (EOH+OT - Sales)"].append(total_eoh_ot.loc[index, "EOH+OT U"] - row["SUM_SELECTED"])

        # Add forecasted weeks
        for i, date in enumerate(forecast_dates):
            # Calculate the week number within the month for forecasted weeks
            start_of_month = date.replace(day=1)
            week_number = ((date - start_of_month).days // 7) + 1
            week_label = f"{date.strftime('%b')} wk {week_number} {date.year} (Forecasted)"
            
            forecast_table_data["Week"].append(week_label)
            forecast_table_data["Sales (Units)"].append(forecasted_sales_qty[i])
            forecast_table_data["EOH+OT U (Units)"].append(forecasted_eoh_ot[i])
            forecast_table_data["Balance (EOH+OT - Sales)"].append(forecasted_eoh_ot[i] - forecasted_sales_qty[i])

        # Convert to DataFrame for display
        forecast_table_df = pd.DataFrame(forecast_table_data)

        # Highlight negative balances and forecasted weeks
        def style_table(row):
            styles = []
            for val, col in zip(row, forecast_table_df.columns):
                if col == "Week" and "(Forecasted)" in str(val):
                    styles.append("text-decoration: underline; color: #0056b3;")
                elif col == "Balance (EOH+OT - Sales)" and val < 0:
                    styles.append("color: blue; font-weight: bold;")
                else:
                    styles.append("")
            return styles

        # Display the table with styling applied row-wise
        st.write("Weekly Sales and Inventory Forecast Table")
        st.dataframe(
            forecast_table_df.style.apply(style_table, axis=1).format("{:,.0f}", subset=["Sales (Units)", "EOH+OT U (Units)", "Balance (EOH+OT - Sales)"])
        )
        #====================================================================================================
        # Combine main explanation with depletion week and negative stock week info in one styled markdown
        st.markdown(
            f"<span style='font-size:18px; font-weight:bold;'>Inventory Depletion Point:</span> "
            "<span style='font-size:18px;'>Displaying the point at which the forecasted Sales line intersects with the forecasted EOH+OT U line, indicating the expected timing of inventory exhaustion.</span><br>"
            f"{depletion_week_text}<br><br>"
            f"<span style='font-size:18px; font-weight:bold;'>Negative Stock Point:</span> "
            "<span style='font-size:18px;'>Displaying the first point at which forecasted EOH+OT U turns negative, indicating potential stock shortages.</span><br>"
            f"{negative_stock_week_text}",
            unsafe_allow_html=True
        )
        st.divider()

        #====================================Contribution Calculation====================================
        # Function to calculate metrics including ASP, contributions, and averages by 4 weeks
        def calculate_metrics(grouped_data):
            # Ensure required columns exist
            required_columns = ["REG SALES U", "PROMO SALES U", "CLEAR SALES U", "REG SALES $", "PROMO SALES $", "CLEAR SALES $", 
                                "EOH+OT $", "EOH+OT U", "GROSS MARGIN $"]
            for col in required_columns:
                if col not in grouped_data.columns:
                    grouped_data[col] = 0

            # Convert numeric columns to numeric types and handle missing values
            grouped_data[["REG SALES U", "PROMO SALES U", "CLEAR SALES U"]] = grouped_data[["REG SALES U", "PROMO SALES U", "CLEAR SALES U"]].apply(pd.to_numeric, errors='coerce').fillna(0)
            grouped_data[["REG SALES $", "PROMO SALES $", "CLEAR SALES $", "EOH+OT $", "GROSS MARGIN $"]] = grouped_data[["REG SALES $", "PROMO SALES $", "CLEAR SALES $", "EOH+OT $", "GROSS MARGIN $"]].apply(pd.to_numeric, errors='coerce').fillna(0)

            # Total Sales Units and Value
            total_sales_units = grouped_data[['REG SALES U', 'PROMO SALES U', 'CLEAR SALES U']].sum().sum()
            total_sales_value = grouped_data[['REG SALES $', 'PROMO SALES $', 'CLEAR SALES $']].sum().sum()

            # Gross Margin Calculations
            total_gross_margin = grouped_data['GROSS MARGIN $'].sum()
            gross_margin_percent = (total_gross_margin / total_sales_value) * 100 if total_sales_value != 0 else 0

            # ASP Calculation
            asp = total_sales_value / total_sales_units if total_sales_units != 0 else 0

            # 4-week Averages Calculation (average per week * 4)
            average_inventory_value_4w = grouped_data['EOH+OT $'].mean() * 4
            average_inventory_units_4w = grouped_data['EOH+OT U'].mean() * 4
            average_sales_value_4w = grouped_data[['REG SALES $', 'PROMO SALES $', 'CLEAR SALES $']].mean().sum() * 4
            average_sales_units_4w = grouped_data[['REG SALES U', 'PROMO SALES U', 'CLEAR SALES U']].mean().sum() * 4

            # Average Sales % Calculation
            total_eoh_sales_units = total_sales_units + grouped_data['EOH+OT U'].sum()
            average_sales_percent = (total_sales_units / total_eoh_sales_units) * 100 if total_eoh_sales_units != 0 else 0

            # Contribution Calculations
            total_sales_df = grouped_data['REG SALES $'] + grouped_data['PROMO SALES $'] + grouped_data['CLEAR SALES $']
            total_sales_df = total_sales_df.replace(0, 1)  # Avoid division by zero
            average_reg_contribution = (grouped_data['REG SALES $'].sum() / total_sales_value) * 100 if total_sales_value > 0 else 0
            average_promo_contribution = (grouped_data['PROMO SALES $'].sum() / total_sales_value) * 100 if total_sales_value > 0 else 0
            average_clear_contribution = (grouped_data['CLEAR SALES $'].sum() / total_sales_value) * 100 if total_sales_value > 0 else 0
            
            # Avoid division by zero in row-wise contributions
            grouped_data['PROMO_CONTRIBUTION_%'] = np.where(
                total_sales_df > 0,
                (grouped_data['PROMO SALES $'] / total_sales_df) * 100,
                0
            )
            grouped_data['CLEAR_CONTRIBUTION_%'] = np.where(
                total_sales_df > 0,
                (grouped_data['CLEAR SALES $'] / total_sales_df) * 100,
                0
            )
            grouped_data['REG_CONTRIBUTION_%'] = np.where(
                total_sales_df > 0,
                (grouped_data['REG SALES $'] / total_sales_df) * 100,
                0
            )

            # Final dictionary of metrics
            return {
                "total_sales_value": total_sales_value,
                "total_sales_units": total_sales_units,
                "total_gross_margin": total_gross_margin,
                "gross_margin_percent": gross_margin_percent,
                "asp": asp,
                "average_inventory_value_4w": average_inventory_value_4w,
                "average_inventory_units_4w": average_inventory_units_4w,
                "average_sales_value_4w": average_sales_value_4w,
                "average_sales_units_4w": average_sales_units_4w,
                "average_sales_percent": average_sales_percent,
                "average_promo_contribution": average_promo_contribution,
                "average_clear_contribution": average_clear_contribution,
                "average_reg_contribution": average_reg_contribution,
                "promo_contribution_%": grouped_data['PROMO_CONTRIBUTION_%'].mean(),
                "clear_contribution_%": grouped_data['CLEAR_CONTRIBUTION_%'].mean(),
                "reg_contribution_%": grouped_data['REG_CONTRIBUTION_%'].mean()
            }

        # Calculate and retrieve metrics
        metrics = calculate_metrics(grouped_data)

        # Create two columns for the layout
        col1, col2 = st.columns([1, 8])  # Adjust column width ratio as needed

        with col1:
            image_path = r"data\contribution.png"
            try:
                image = Image.open(image_path)
                st.image(image, caption=None, use_container_width=True)
            except FileNotFoundError:
                st.error("Image not found. Please check the path.")

        with col2:
            st.markdown(
                """
                <div style='font-size:36px; font-weight:bold; color:black;'>
                Sales $ Contribution Over Time
                </div>
                """, 
                unsafe_allow_html=True
            )


        # Aggregate data by month for bar graph
        monthly_summary = grouped_data.groupby(grouped_data['Converted_Date'].dt.to_period("M")).agg(
            Reg_Sales_Value=("REG SALES $", "sum"),
            Promo_Sales_Value=("PROMO SALES $", "sum"),
            Clear_Sales_Value=("CLEAR SALES $", "sum"),
            EOH_OT_Units=("EOH+OT U", "sum")  # Adding EOH+OT U
        ).reset_index()

        monthly_summary['Sales_Units'] = (
            grouped_data.groupby(grouped_data['Converted_Date'].dt.to_period("M"))
            [['REG SALES U', 'PROMO SALES U', 'CLEAR SALES U']].sum().sum(axis=1).values
        )

        # Add Total Value and Units
        monthly_summary['Total_Sales_Amount'] = (
            monthly_summary['Reg_Sales_Value'] +
            monthly_summary['Promo_Sales_Value'] +
            monthly_summary['Clear_Sales_Value']
        )
        monthly_summary['Total_Units'] = monthly_summary['Sales_Units'] + monthly_summary['EOH_OT_Units']

        # Convert the period to timestamp for visualization
        monthly_summary['Converted_Date'] = monthly_summary['Converted_Date'].dt.to_timestamp()

        # Calculate contributions
        monthly_summary['Promo_Contribution_%'] = (
            monthly_summary['Promo_Sales_Value'] / monthly_summary['Total_Sales_Amount']
        ) * 100
        monthly_summary['Clear_Contribution_%'] = (
            monthly_summary['Clear_Sales_Value'] / monthly_summary['Total_Sales_Amount']
        ) * 100
        monthly_summary['Reg_Contribution_%'] = (
            monthly_summary['Reg_Sales_Value'] / monthly_summary['Total_Sales_Amount']
        ) * 100

        # Initialize the figure
        fig_combined = go.Figure()

        # Add stacked bar traces for sales contributions
        fig_combined.add_trace(go.Bar(
            x=monthly_summary['Converted_Date'],
            y=monthly_summary['Reg_Sales_Value'],
            name='Regular Sales ($)',
            marker=dict(color="#2ecc71")  # Emerald
        ))
        fig_combined.add_trace(go.Bar(
            x=monthly_summary['Converted_Date'],
            y=monthly_summary['Promo_Sales_Value'],
            name='Promo Sales ($)',
            marker=dict(color="#3498db")  # Sapphire
        ))
        fig_combined.add_trace(go.Bar(
            x=monthly_summary['Converted_Date'],
            y=monthly_summary['Clear_Sales_Value'],
            name='Clear Sales ($)',
            marker=dict(color="#e67e22")  # Orange
        ))

        # Add Total Sales Units as line graph on the second Y-axis
        fig_combined.add_trace(go.Scatter(
            x=monthly_summary['Converted_Date'],
            y=monthly_summary['Sales_Units'],
            mode='lines+markers',
            name='Total Sales Units (pcs)',
            line=dict(color="orange", width=2),
            yaxis="y2"
        ))

        # Add EOH+OT U as an additional line graph
        fig_combined.add_trace(go.Scatter(
            x=monthly_summary['Converted_Date'],
            y=monthly_summary['EOH_OT_Units'],
            mode='lines+markers',
            name='EOH+OT U (pcs)',
            line=dict(color="skyblue", width=2),
            yaxis="y2"
        ))

        # Add annotations for contributions
        for i, row in monthly_summary.iterrows():
            fig_combined.add_annotation(
                x=row['Converted_Date'],
                y=row['Reg_Sales_Value'] / 2,
                text=f"{row['Reg_Contribution_%']:.1f}%",
                showarrow=False,
                font=dict(color="white", size=10),
                bgcolor="black"
            )
            fig_combined.add_annotation(
                x=row['Converted_Date'],
                y=row['Reg_Sales_Value'] + row['Promo_Sales_Value'] / 2,
                text=f"{row['Promo_Contribution_%']:.1f}%",
                showarrow=False,
                font=dict(color="white", size=10),
                bgcolor="black"
            )
            fig_combined.add_annotation(
                x=row['Converted_Date'],
                y=row['Reg_Sales_Value'] + row['Promo_Sales_Value'] + row['Clear_Sales_Value'] / 2,
                text=f"{row['Clear_Contribution_%']:.1f}%",
                showarrow=False,
                font=dict(color="white", size=10),
                bgcolor="black"
            )

        # Update layout
        fig_combined.update_layout(
            barmode='stack',
            xaxis=dict(
                title="Date",
                tickformat="%b %Y",
                tickmode='linear',
                dtick="M1"  # Ensure every month appears
            ),
            yaxis=dict(
                title="Sales ($)",
                side="left",
            ),
            yaxis2=dict(
                title="Units (pcs)",
                overlaying="y",
                side="right"
            ),
            title="Monthly Sales, EOH+OT U, and Contribution Analysis",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            template="plotly_white"
        )

        # Display the updated graph
        st.plotly_chart(fig_combined, use_container_width=True)

        # Prepare data for the pivot table
        pivot_table = monthly_summary.pivot_table(
            index=None,
            columns=monthly_summary['Converted_Date'].dt.strftime("%b %Y"),  # Include year
            values=[
                'Total_Sales_Amount',  # Total Sales $
                'Sales_Units',  # Total Sales Units
                'EOH_OT_Units',  # EOH+OT U
                'Reg_Contribution_%',  # Regular Contribution %
                'Promo_Contribution_%',  # Promo Contribution %
                'Clear_Contribution_%'  # Clear Contribution %
            ],
            aggfunc='sum'
        )


        # Rearrange months in chronological order (Jan to Dec)
        month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        pivot_table = pivot_table.sort_index(axis=1, key=lambda x: pd.to_datetime(x, format="%b %Y"))

        # Drop columns with all NaN values
        pivot_table = pivot_table.dropna(how='all', axis=1)

        # Reorder rows to match the desired order
        row_order = ['Total_Sales_Amount', 'Sales_Units', 'EOH_OT_Units', 'Reg_Contribution_%', 'Promo_Contribution_%', 'Clear_Contribution_%']
        pivot_table = pivot_table.loc[row_order]

        # Calculate Total column for value-based rows
        pivot_table.loc['Total_Sales_Amount', 'Total'] = pivot_table.loc['Total_Sales_Amount'].sum()
        pivot_table.loc['Sales_Units', 'Total'] = pivot_table.loc['Sales_Units'].sum()
        pivot_table.loc['EOH_OT_Units', 'Total'] = pivot_table.loc['EOH_OT_Units'].sum()

        # Calculate Total column for percentage-based rows based on total sales amounts
        total_sales_amount = pivot_table.loc['Total_Sales_Amount', 'Total']
        if total_sales_amount > 0:
            pivot_table.loc['Reg_Contribution_%', 'Total'] = (
                (pivot_table.loc['Total_Sales_Amount'] * pivot_table.loc['Reg_Contribution_%'] / 100).sum() / total_sales_amount
            ) * 100
            pivot_table.loc['Promo_Contribution_%', 'Total'] = (
                (pivot_table.loc['Total_Sales_Amount'] * pivot_table.loc['Promo_Contribution_%'] / 100).sum() / total_sales_amount
            ) * 100
            pivot_table.loc['Clear_Contribution_%', 'Total'] = (
                (pivot_table.loc['Total_Sales_Amount'] * pivot_table.loc['Clear_Contribution_%'] / 100).sum() / total_sales_amount
            ) * 100
        else:
            pivot_table.loc['Reg_Contribution_%', 'Total'] = 0
            pivot_table.loc['Promo_Contribution_%', 'Total'] = 0
            pivot_table.loc['Clear_Contribution_%', 'Total'] = 0

        # Format the pivot table
        formatted_table = pivot_table.copy()

        # Format rows explicitly
        formatted_table.loc['Total_Sales_Amount'] = formatted_table.loc['Total_Sales_Amount'].apply(
            lambda x: f"${x:,.0f}" if pd.notnull(x) else "")
        formatted_table.loc['Sales_Units'] = formatted_table.loc['Sales_Units'].apply(
            lambda x: f"{x:,.0f}" if pd.notnull(x) else "")
        formatted_table.loc['EOH_OT_Units'] = formatted_table.loc['EOH_OT_Units'].apply(
            lambda x: f"{x:,.0f}" if pd.notnull(x) else "")
        formatted_table.loc['Reg_Contribution_%'] = formatted_table.loc['Reg_Contribution_%'].apply(
            lambda x: f"{x:.1f}%" if pd.notnull(x) else "")
        formatted_table.loc['Promo_Contribution_%'] = formatted_table.loc['Promo_Contribution_%'].apply(
            lambda x: f"{x:.1f}%" if pd.notnull(x) else "")
        formatted_table.loc['Clear_Contribution_%'] = formatted_table.loc['Clear_Contribution_%'].apply(
            lambda x: f"{x:.1f}%" if pd.notnull(x) else "")

        # Display the formatted table with a smaller heading
        st.markdown(
            "<div style='font-size:18px; font-weight:bold;'>Monthly Contribution (filtered)</div>",
            unsafe_allow_html=True
        )
        st.dataframe(formatted_table)
        st.divider()

        #key metrics=======================================================
        # Create two columns for the layout
        col1, col2 = st.columns([1, 8])  # Adjust column width ratio as needed

        with col1:
            image_path = r"data\key.png"
            try:
                image = Image.open(image_path)
                st.image(image, caption=None, use_container_width=True)
            except FileNotFoundError:
                st.error("Image not found. Please check the path.")

        with col2:
            st.markdown(
                """
                <div style='font-size:36px; font-weight:bold; color:black;'>
                Key Metrics Overview
                </div>
                """, unsafe_allow_html=True)
            st.write("""
                This Key Metrics section provides a comprehensive view of sales performance and inventory metrics with gross margin.
            """)

        # Define CSS styles
        st.markdown("""
            <style>
            .metric-container {
                display: flex;
                align-items: center;
                padding: 10px;
                margin-bottom: 10px;
                border-radius: 8px;
                background-color: #f5faff;
                border: 1px solid #1E90FF;
            }
            .metric-label {
                font-weight: bold;
                font-size: 1.1em;
                color: #1E90FF;
                flex: 1;
            }
            .metric-value {
                font-size: 1.4em;
                font-weight: bold;
                color: #104E8B;
            }
            .metric-arrow {
                margin-left: 5px;
                color: #1E90FF;
                font-size: 1.2em;
            }
            .metric-tooltip {
                font-size: 0.9em;
                color: #555;
                margin-left: 10px;
                font-style: italic;
            }
            </style>
        """, unsafe_allow_html=True)

        # Add custom CSS for styling
        st.markdown("""
            <style>
                .metric-container {
                    display: flex;
                    flex-direction: column;
                    background-color: #f9f9f9;
                    border: 1px solid #ddd;
                    border-radius: 10px;
                    padding: 15px;
                    margin-bottom: 10px;
                    box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
                }
                .metric-label {
                    font-size: 1rem;
                    font-weight: 600;
                    color: #333;
                }
                .metric-value {
                    font-size: 1.5rem;
                    font-weight: bold;
                    color: #007BFF;
                }
                .metric-tooltip {
                    font-size: 0.9rem;
                    color: #6c757d;
                    margin-top: 5px;
                }
            </style>
        """, unsafe_allow_html=True)

    # Create 4 columns for displaying metrics
    col1, col2, col3, col4 = st.columns(4)

    # Define the layout for key metrics
    st.markdown("<div style='font-size:20px; font-weight:bold;'>Key Metrics(filtered)</div>", unsafe_allow_html=True)

    # First Row
    row1_col1, row1_col2, row1_col3, row1_col4 = st.columns(4)
    with row1_col1:
        st.markdown(f"""
            <div class="metric-container">
                <span class="metric-label">Total Sales Value:</span>
                <span class="metric-value">${metrics['total_sales_value']:,.2f}</span>
            </div>
        """, unsafe_allow_html=True)

    with row1_col2:
        st.markdown(f"""
            <div class="metric-container">
                <span class="metric-label">Total Sales Units:</span>
                <span class="metric-value">{metrics['total_sales_units']:,.0f} units</span>
            </div>
        """, unsafe_allow_html=True)

    with row1_col3:
        st.markdown(f"""
            <div class="metric-container">
                <span class="metric-label">Total Gross Margin $:</span>
                <span class="metric-value">${metrics['total_gross_margin']:,.2f}</span>
            </div>
        """, unsafe_allow_html=True)

    with row1_col4:
        st.markdown(f"""
            <div class="metric-container">
                <span class="metric-label">Gross Margin %:</span>
                <span class="metric-value">{metrics['gross_margin_percent']:,.2f}%</span>
            </div>
        """, unsafe_allow_html=True)

    # Second Row
    row2_col1, row2_col2, row2_col3, row2_col4 = st.columns(4)
    with row2_col1:
        st.markdown(f"""
            <div class="metric-container">
                <span class="metric-label">Total Regular Sales Contribution:</span>
                <span class="metric-value">{metrics['average_reg_contribution']:,.2f}%</span>
            </div>
        """, unsafe_allow_html=True)

    with row2_col2:
        st.markdown(f"""
            <div class="metric-container">
                <span class="metric-label">Total Promo Sales Contribution:</span>
                <span class="metric-value">{metrics['average_promo_contribution']:,.2f}%</span>
            </div>
        """, unsafe_allow_html=True)

    with row2_col3:
        st.markdown(f"""
            <div class="metric-container">
                <span class="metric-label">Total Clear Sales Contribution:</span>
                <span class="metric-value">{metrics['average_clear_contribution']:,.2f}%</span>
            </div>
        """, unsafe_allow_html=True)

    with row2_col4:
        st.markdown(f"""
            <div class="metric-container">
                <span class="metric-label">Average Selling Price (ASP):</span>
                <span class="metric-value">${metrics['asp']:,.2f}</span>
            </div>
        """, unsafe_allow_html=True)

    # Third Row
    row3_col1, row3_col2, row3_col3, row3_col4 = st.columns(4)
    with row3_col1:
        st.markdown(f"""
            <div class="metric-container">
                <span class="metric-label">Average Sales Value by 4weeks:</span>
                <span class="metric-value">${metrics['average_sales_value_4w']:,.2f}</span>
            </div>
        """, unsafe_allow_html=True)

    with row3_col2:
        st.markdown(f"""
            <div class="metric-container">
                <span class="metric-label">Average Sales Units by 4weeks:</span>
                <span class="metric-value">{metrics['average_sales_units_4w']:,.0f} units</span>
            </div>
        """, unsafe_allow_html=True)

    with row3_col3:
        st.markdown(f"""
            <div class="metric-container">
                <span class="metric-label">Average Inventory Value (EOH+OT $) by 4weeks:</span>
                <span class="metric-value">${metrics['average_inventory_value_4w']:,.2f}</span>
            </div>
        """, unsafe_allow_html=True)

    with row3_col4:
        st.markdown(f"""
            <div class="metric-container">
                <span class="metric-label">Average Inventory Units (EOH+OT U) by 4weeks:</span>
                <span class="metric-value">{metrics['average_inventory_units_4w']:,.0f} units</span>
            </div>
        """, unsafe_allow_html=True)
            
    # ==================================== Export Section ====================================

    st.divider()

    # Export Filtered Data to Excel
    if 'data' in locals() and not data.empty:
        # Define a buffer to write the Excel file to
        buffer = io.BytesIO()

        # Write the DataFrame to an Excel file
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            # Write the filtered data to the Excel sheet
            data.to_excel(writer, index=False, sheet_name='Filtered Data')

            # Set column width for better readability in Excel
            worksheet = writer.sheets['Filtered Data']
            for i, col in enumerate(data.columns):
                column_width = max(data[col].astype(str).map(len).max(), len(col)) + 2
                worksheet.set_column(i, i, column_width)

        # Use columns to control button alignment
        export_col1, export_col2 = st.columns([1, 4])  # Adjust proportions as needed

        with export_col1:
            st.download_button(
                label="📥 Export to Excel (filtered raw data)",
                data=buffer,
                file_name="Filtered_Sales_Report.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    else:
        st.warning("No data available to export.")

else:
    st.info("Please upload one or more Sales report raw data CSV files from Green-field to proceed.")

