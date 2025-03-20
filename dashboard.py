import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from lifelines import KaplanMeierFitter
# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv(r"customer_churn.csv")  # Change to your file
    return df

df = load_data()
# Main Title  
st.markdown("""
    <h1 style='text-align: center; color: white; background-color: #1e3c72; 
    padding: 10px; border-radius: 10px; background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);'>
        Telecom Customer Churn Dashboard
    </h1>
""", unsafe_allow_html=True)
# About 
st.markdown("""
<div style='text-align: right; font-size: 0.8em; color: #888;'>
    Created by Zeeshan Akram
</div>
""", unsafe_allow_html=True)

st.markdown("""
**Welcome!** This dashboard helps analyze customer churn trends, service adoption, and financial metrics.
Use the sidebar to explore different sections and gain insights into customer behavior.
""")

with st.expander("ℹ️ About This Dashboard"):
    st.markdown("""
    ##  Project Overview  
    This interactive dashboard provides a **comprehensive analysis** of telecom customer churn,  
    helping businesses **identify patterns, understand risk factors, and make data-driven retention strategies**.  
    
    ###  Features:
      **Interactive visualizations** powered by **Plotly**
      **Advanced analytics** including **clustering & feature importance**
      **Predictive insights** from **machine learning models**
      **Responsive & dynamic design** built with **Streamlit**
    
    ###  Technical Implementation:
      **Python** (Pandas, NumPy, Scikit-learn) for **data processing & ML**
      **Plotly** for **high-quality interactive visualizations**
      **Streamlit** for **seamless web app integration**
    
    
    ** Created by [Zeeshan Akram](https://www.linkedin.com/in/zeeshan-akram-572bbb34a/details/skills/)**  
     Data Science |  Machine Learning |  Analytics  
    """)



# Sidebar Title  
st.sidebar.markdown("## Telecom Customer Churn Dashboard")

# Sidebar Description  
st.sidebar.markdown("""
This app is a **Streamlit dashboard** designed to **analyze customer churn data**.  
Explore key insights, trends, and predictive analytics to understand churn behavior.
""")

if st.sidebar.checkbox("Show Raw Dataset"):
    st.write("### Full Dataset Preview")
    st.dataframe(df)

st.sidebar.write('# Enter Customer ID')
customer_id = st.sidebar.text_input("Customer ID", "")
if customer_id:
    customer_data = df[df["customerID"] == customer_id]
    if not customer_data.empty:
        st.dataframe(customer_data)
    else:
        st.warning("No customer found with this ID.")

# Sidebar for Section Selection
section = st.sidebar.selectbox(
    "Select Section", 
    [
        "Data Distribution Analysis",
        "Customer Count by Demographics",
        "Customer Count by Service Type",
        "Customer Distribution by Account Features",
        "Customer Status Filter",
        "Churn Distribution by Demographics",
        "Customer Tenure & Churn Trends",
        "Service Adoption & Churn Analysis",
        "Churn Analysis by Account Features",
        "Financial Metrics, Contract & Billing",
        "Advanced Insights"
    ],
    index= 4
)

if section == "Data Distribution Analysis":
    # Sidebar Section for Data Distribution Analysis
    st.sidebar.subheader("Data Distribution Analysis")

    # Selectbox for choosing a distribution chart
    distribution_chart_type = st.sidebar.selectbox(
        "Select a distribution chart:",
        [
            "Monthly Charges Distribution",
            "Total Charges Distribution",
            "Service Usage Count"
        ],
        key="distribution_chart_select"
    )

    
    if distribution_chart_type == "Monthly Charges Distribution":
        st.subheader("Monthly Charges Distribution")
        
        fig_monthly_charges = plt.figure(figsize=(10, 5))
        sns.histplot(df["MonthlyCharges"], bins=30, kde=True, color="#E67E22")  # Orange for good visibility
        plt.xlabel("Monthly Charges ($)")
        plt.ylabel("Customer Count")
        plt.title("Monthly Charges Distribution with KDE")
        plt.grid(True)
        st.pyplot(fig_monthly_charges)

    elif distribution_chart_type == "Total Charges Distribution":
        st.subheader("Total Charges Distribution")
        
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")  # Ensure numeric values
        fig_total_charges = plt.figure(figsize=(10, 5))
        sns.histplot(df["TotalCharges"].dropna(), bins=30, kde=True, color="#1ABC9C")  # Teal color
        plt.xlabel("Total Charges ($)")
        plt.ylabel("Customer Count")
        plt.title("Total Charges Distribution with KDE")
        plt.grid(True)
        st.pyplot(fig_total_charges)

    
    elif distribution_chart_type == "Service Usage Count":
        st.subheader("Service Usage Count")
        
        service_columns = [
            "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity", 
            "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"
        ]
        service_usage = df[service_columns].apply(lambda x: x.value_counts()).T
        service_usage.reset_index(inplace=True)
        service_usage.rename(columns={"index": "Service"}, inplace=True)

        fig_service_count = px.bar(
            service_usage.melt(id_vars=["Service"], var_name="Category", value_name="Count"),
            x="Service",
            y="Count",
            color="Category",
            barmode="group",
            title="Service Usage Count",
            color_discrete_map={"Yes": "#2ECC71", "No": "#E74C3C"}  
        )
        st.plotly_chart(fig_service_count, use_container_width=True)

    elif distribution_chart_type == "Churn Rate by Tenure Bins":
        st.subheader("Churn Rate by Tenure Bins")
        
        df["TenureBin"] = pd.cut(df["tenure"], bins=[0, 12, 24, 36, 48, 60, 72], labels=["0-12", "13-24", "25-36", "37-48", "49-60", "61+"])
        tenure_churn = df.groupby("TenureBin")["Churn"].mean().reset_index()
        tenure_churn["Churn"] *= 100 

elif section == "Customer Count by Demographics":
    # Subheader
    st.sidebar.subheader("Customer Count by Demographics")

    # Selectbox for Demographic Feature
    demographic_feature = st.sidebar.selectbox(
        "Select a Demographic Feature:", 
        ["gender", "SeniorCitizen", "Partner", "Dependents", "Churn"]
    )

    # Selectbox for Visualization Type
    chart_type = st.sidebar.selectbox(
        "Select Visualization Type:", 
        ["Bar Chart", "Pie Chart"]  
    )

    # Compute counts for selected feature
    demographic_counts = df[demographic_feature].value_counts().reset_index()
    demographic_counts.columns = [demographic_feature, "Count"]

    # Define Different Color Themes for Each Feature
    color_map = {
        "Gender": px.colors.qualitative.Pastel,
        "SeniorCitizen": px.colors.qualitative.Bold,
        "Partner": px.colors.qualitative.Dark24,
        "Dependents": px.colors.qualitative.Vivid,
        "Churn": px.colors.qualitative.Set2
    }

    if chart_type == "Bar Chart":
        fig_demo = px.bar(
            demographic_counts, 
            x=demographic_feature, 
            y="Count", 
            title=f"Customer Count by {demographic_feature}",
            color=demographic_feature,
            text="Count",
            color_discrete_sequence=color_map.get(demographic_feature, px.colors.qualitative.Set2)  # Dynamic Color
        )
        fig_demo.update_traces(texttemplate='%{text}', textposition='outside')
        fig_demo.update_layout(xaxis_title=demographic_feature, yaxis_title="Customer Count")
        st.plotly_chart(fig_demo, use_container_width=True)

    elif chart_type == "Pie Chart":
        fig_demo = px.pie(
            demographic_counts, 
            names=demographic_feature, 
            values="Count", 
            title=f"Customer Distribution by {demographic_feature}",
            color=demographic_feature,
            color_discrete_sequence=color_map.get(demographic_feature, px.colors.qualitative.Set2)  # Dynamic Color
        )
        fig_demo.update_traces(textinfo='percent+label')
        st.plotly_chart(fig_demo, use_container_width=True)

elif section == "Customer Count by Service Type":
    # Subheader
    st.sidebar.subheader("Customer Count by Service Type")

    # Selectbox for Service Feature
    service_feature = st.sidebar.selectbox(
        "Select a Service Type:",
        ["PhoneService", "MultipleLines", "InternetService", "OnlineSecurity", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"],
        key="service_feature_select"
    )

    # Selectbox for Vizz Type
    chart_type = st.sidebar.selectbox(
        "Select Visualization Type:",
        ["Bar Chart", "Pie Chart"],
        key="service_chart_type"
    )

    # Compute counts
    service_counts = df[service_feature].value_counts().reset_index()
    service_counts.columns = [service_feature, "Count"]

    # Define colors
    color_map = {
        "PhoneService": ["#1F77B4"],       
        "MultipleLines": ["#FF7F0E"],      
        "InternetService": ["#2CA02C"],    
        "OnlineSecurity": ["#D62728"],     
        "DeviceProtection": ["#9467BD"],   
        "TechSupport": ["#17BECF"],        
        "StreamingTV": ["#E377C2"],        
        "StreamingMovies": ["#8C564B"]     
    }

    # Display Chart
    if chart_type == "Bar Chart":
        fig_service = px.bar(
            service_counts, 
            y=service_feature, 
            x="Count", 
            title=f"Customer Count by {service_feature}",
            color=service_feature,
            text="Count",
            color_discrete_sequence=color_map.get(service_feature, px.colors.qualitative.Bold)  # Dynamic Colors
        )
        fig_service.update_traces(texttemplate='%{text}', textposition='outside')
        fig_service.update_layout(yaxis_title=service_feature, xaxis_title="Customer Count", bargap=0.3)
        st.plotly_chart(fig_service, use_container_width=True)

    elif chart_type == "Pie Chart":
        fig_service = px.pie(
            service_counts, 
            names=service_feature, 
            values="Count", 
            title=f"Customer Distribution by {service_feature}",
            color=service_feature,
            color_discrete_sequence=color_map.get(service_feature, px.colors.qualitative.Bold)  # Dynamic Colors
        )
        fig_service.update_traces(textinfo='percent+label')
        st.plotly_chart(fig_service, use_container_width=True)

elif section == "Customer Distribution by Account Features":
    # Customer Count by Account Information
    st.sidebar.subheader("Customer Distribution by Account Features")

    # Select Feature
    account_feature = st.sidebar.selectbox(
        "Select Account Feature:", 
        ["Contract", "PaperlessBilling", "PaymentMethod"],
        key="account_feature_select"
    )

    # Select Vizz Type
    viz_type_account = st.sidebar.selectbox(
        "Select Visualization Type:", 
        ["Bar Chart", "Pie Chart"], 
        key="account_viz_select"
    )

    # Colors
    color_map_account = {
        "Contract": ["#6A0572"],         
        "PaperlessBilling": ["#D90368"], 
        "PaymentMethod": ["#FF9800"]     
    }

    # Count the selected feature
    account_count = df[account_feature].value_counts().reset_index()
    account_count.columns = [account_feature, "Count"]
    
    # Horizontal Bar Chart
    if viz_type_account == "Bar Chart":
        fig_account = px.bar(
            account_count, 
            x="Count", 
            y=account_feature, 
            orientation="h",
            text="Count",
            title=f"Customer Count by {account_feature}",
            color=account_feature, 
            color_discrete_sequence=color_map_account.get(account_feature, ["#636EFA"])
        )
        fig_account.update_traces(texttemplate='%{text}', textposition='inside')
        fig_account.update_layout(
            xaxis_title="Number of Customers", 
            yaxis_title=account_feature,
            margin=dict(r=200),  
            legend=dict(
                yanchor="top",
                y=1,
                xanchor="right",
                x=1.5  
            )
        )
        st.plotly_chart(fig_account, use_container_width=True)

    # Pie Chart
    elif viz_type_account == "Pie Chart":
        fig_pie = px.pie(
            account_count, 
            names=account_feature, 
            values="Count", 
            title=f"Distribution of {account_feature}",
            color=account_feature, 
            color_discrete_sequence=color_map_account.get(account_feature, ["#636EFA"])
        )
        st.plotly_chart(fig_pie, use_container_width=True)

elif section == "Customer Status Filter":
    # Sidebar 
    st.sidebar.subheader("Customer Status Filter")
    customer_type = st.sidebar.selectbox("Select Customer Type:", ["All Customers", "Churned Customers", "Retained Customers"])

    # Apply Filter
    if customer_type == "Churned Customers":
        filtered_df = df[df["Churn"] == "Yes"]
    elif customer_type == "Retained Customers":
        filtered_df = df[df["Churn"] == "No"]
    else:
        filtered_df = df  # All Customers

    # Show More Filters
    apply_filters = st.sidebar.checkbox("Apply More Filters")

    if apply_filters:
        # Demographic Filters
        gender_filter = st.sidebar.multiselect("Select Gender", df["gender"].unique(), default=df["gender"].unique())
        senior_filter = st.sidebar.multiselect("Senior Citizen Status", df["SeniorCitizen"].unique(), default=df["SeniorCitizen"].unique())

        # Service Filters
        internet_filter = st.sidebar.multiselect("Internet Service", df["InternetService"].unique(), default=df["InternetService"].unique())
        tech_support_filter = st.sidebar.multiselect("Tech Support", df["TechSupport"].unique(), default=df["TechSupport"].unique())

        # Contract Filters
        contract_filter = st.sidebar.multiselect("Contract Type", df["Contract"].unique(), default=df["Contract"].unique())
        payment_filter = st.sidebar.multiselect("Payment Method", df["PaymentMethod"].unique(), default=df["PaymentMethod"].unique())

        # Applying Additional Filters
        filtered_df = filtered_df[
            (filtered_df["gender"].isin(gender_filter)) &
            (filtered_df["SeniorCitizen"].isin(senior_filter)) &
            (filtered_df["InternetService"].isin(internet_filter)) &
            (filtered_df["TechSupport"].isin(tech_support_filter)) &
            (filtered_df["Contract"].isin(contract_filter)) &
            (filtered_df["PaymentMethod"].isin(payment_filter))
        ]

    # Computing KPIs 
    total_customers = len(filtered_df)
    churned_customers = filtered_df[filtered_df['Churn'] == 'Yes'].shape[0]
    churn_rate = (churned_customers / total_customers) * 100 if total_customers > 0 else 0
    average_monthly_charges = filtered_df['MonthlyCharges'].mean()
    total_revenue = filtered_df['MonthlyCharges'].sum()

    # Displaying KPIs
    st.markdown("### Key Metrics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(" Total Customers", total_customers)

    with col2:
        st.metric(" Churn Rate", f"{churn_rate:.1f}%")

    with col3:
        st.metric(" Avg Monthly Charges", f"${average_monthly_charges:.2f}")

    with col4:
        st.metric(" Total Revenue", f"${total_revenue:.2f}")

elif section == "Churn Distribution by Demographics":
    # Select Demographic Feature  
    st.sidebar.subheader("Churn Distribution by Demographics")  
    selected_demo = st.sidebar.selectbox(
        "Select a Demographic Feature:",
        ["gender", "SeniorCitizen", "Partner", "Dependents"]
    )

    # Converting Churn to numeric
    df["Churn"] = df["Churn"].apply(lambda x: 1 if x == "Yes" else 0)

    # calculate churn rate
    def calculate_churn_rate(df, column):
        churn_rates = df.groupby(column)["Churn"].mean() * 100  # to percentage
        return churn_rates.index, churn_rates.values

    # Defining colors
    color_map = {
        "gender": "#1f77b4",
        "SeniorCitizen": "#ff7f0e",
        "Partner": "#2ca02c",
        "Dependents": "#d62728"
    }

    # Display chart
    x_values, y_values = calculate_churn_rate(df, selected_demo)

    text_position = ["inside" if val > 10 else "outside" for val in y_values]
    text_color = ["white" if val > 10 else "black" for val in y_values]

    # Creating the plot  
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=x_values,
            y=y_values,
            marker_color=color_map[selected_demo],  
            text=[f"{val:.1f}%" for val in y_values],  
            textposition=text_position,  
            insidetextfont=dict(color=text_color),  
            name=selected_demo
        )
    )

    # Updating layout  
    fig.update_layout(
        title_text=f"Churn Rate by {selected_demo}",
        xaxis_title=selected_demo,
        yaxis_title="Churn Rate (%)",
        height=500,
        width=800
    )

    st.plotly_chart(fig)

    # Show Insights   
    st.markdown("## **Key Insights from Demographic Churn Analysis**")

    if selected_demo == "Gender":
        st.markdown("""
        **Gender & Churn:** There is **no significant difference** in churn rate between **males (26.2%) and females (26.9%)**.  
        **Insight:** **Gender does not have a major impact** on churn behavior.  
        """)

    elif selected_demo == "Senior Citizen":
        st.markdown("""
        **Senior Citizen Status & Churn:** **Senior Citizens (41.7%)** have a **much higher churn rate** than **non-Senior Citizens (23.6%)**.  
        **Insight:** Older customers may leave due to **financial constraints, lack of need, or difficulty in adapting to services**.  
        """)

    elif selected_demo == "Partner":
        st.markdown("""
        **Partner Status & Churn:** Customers **without a partner (33%)** churn **significantly more** than those **with a partner (19.7%)**.  
        **Insight:** Having a **partner may provide financial stability** or encourage **shared service plans**, reducing churn.  
        """)

    elif selected_demo == "Dependents":
        st.markdown("""
        **Dependents & Churn:** **Independent customers (31.3%)** have almost **twice the churn rate** compared to those **with dependents (15.5%)**.  
        **Insight:** Customers **with dependents may prefer stability**, making them **less likely to switch providers**.  
        """)

    # Summary
    st.markdown("""
    ### **Key Takeaway:**
    - **Senior citizens, customers without partners, and independent individuals** are at **higher churn risk**.  
    - **Retention strategies** should focus on **personalized discounts, better customer support, and value-added services** for these groups.  
    """)

elif section == "Customer Tenure & Churn Trends":
    # Customer Tenure & Churn Trends
    st.sidebar.subheader("Customer Tenure & Churn Trends")

    # Selectbox to choose vizz
    tenure_chart_type = st.sidebar.selectbox(
        "Select a chart:",
        ["Churn Rate vs. Tenure", "Tenure Distribution (Churned vs. Retained)", "Kaplan-Meier Survival Curve"]
    )

    # Display 
    if tenure_chart_type == "Churn Rate vs. Tenure":
        # Churn Rate vs. Tenure (Line Chart)
        fig_line = plt.figure(figsize=(10, 5))
        sns.lineplot(data=df, x='tenure', y='Churn', estimator='mean', ci=None, color='red')
        plt.xlabel("Tenure (Months)")
        plt.ylabel("Churn Rate")
        plt.title("Churn Rate vs. Tenure")
        plt.grid(True)
        st.pyplot(fig_line)

        # Insights
        st.subheader("Churn Rate vs. Tenure")
        st.markdown(
            """
            - **New customers (tenure < 1 month) have the highest churn rate**.
            - As tenure increases, churn rate declines, suggesting **long-term customers are more loyal**.
            - **Customers staying for 3-4 years tend to remain for even longer**.
            """
        )

    elif tenure_chart_type == "Tenure Distribution (Churned vs. Retained)":
        # Tenure Distribution (Histogram)
        # 🔹 Fix: Convert Churn column if it's stored as "Yes" / "No"
        if df['Churn'].dtype == 'object':
            df['Churn'] = df['Churn'].str.strip()  # Remove spaces
            df['Churn'] = df['Churn'].replace({'Yes': 1, 'No': 0})  # Convert to 0/1

        # Ensure 'Churn' is numeric
        df['Churn'] = pd.to_numeric(df['Churn'], errors='coerce').fillna(0).astype(int)

        # 🔹 Check if tenure column exists and is not empty
        if 'tenure' not in df.columns or df['tenure'].dropna().empty:
            st.error("Error: 'tenure' column is missing or contains only NaN values.")
            st.stop()

        retained_count = df[df['Churn'] == 0].shape[0]
        churned_count = df[df['Churn'] == 1].shape[0]

        if retained_count > 0 and churned_count > 0:
            fig_hist = plt.figure(figsize=(10, 5))
            sns.histplot(df[df['Churn'] == 0]['tenure'], bins=30, color='blue', alpha=0.5, label='Retained')
            sns.histplot(df[df['Churn'] == 1]['tenure'], bins=30, color='red', alpha=0.5, label='Churned')

            plt.xlabel("Tenure (Months)")
            plt.ylabel("Count")
            plt.title("Tenure Distribution: Churned vs. Retained Customers")
            plt.legend()
            plt.grid(True)

            st.pyplot(fig_hist)
        else:
            st.warning("Not enough data to generate tenure distribution.")
        # Insights
        st.subheader("Tenure Distribution (Churned vs. Retained)")
        st.markdown(
            """
            - **Churned customers have shorter tenure**, meaning they leave early.
            - **Retained customers have longer tenure**, showing better customer retention over time.
            - **Mid-range tenure (10-50 months) has a mix of both churned and retained customers**.
            """
        )

    elif tenure_chart_type == "Kaplan-Meier Survival Curve":
        # Kaplan-Meier Survival Curve
        if df['Churn'].dtype == 'object':
            df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
        kmf = KaplanMeierFitter()
        kmf_fig = plt.figure(figsize=(10, 5))
        kmf.fit(durations=df['tenure'], event_observed=df['Churn'])
        kmf.plot_survival_function(color='green', label='Customer Retention Probability')
        plt.xlabel("Tenure (Months)")
        plt.ylabel("Survival Probability")
        plt.title("Kaplan-Meier Survival Curve for Customer Retention")
        plt.grid(True)
        st.pyplot(kmf_fig)

        # Insights
        st.subheader("Kaplan-Meier Survival Curve (Customer Retention)")
        st.markdown(
            """
            - **Customers with low tenure have the highest churn risk**.
            - **Long-term customers are far less likely to leave**.
            - **Focusing on retaining new customers early can significantly boost long-term retention**.
            """
        )

elif section == "Service Adoption & Churn Analysis":
    # Service Selection
    st.sidebar.subheader("Service Adoption & Churn Analysis")
    service_options = ["PhoneService", "MultipleLines", "InternetService", "OnlineSecurity", 
                    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]

    selected_service = st.sidebar.selectbox("Select a Service to Analyze:", service_options)

    # Defining colors 
    color_map = {
        "PhoneService": "#1f77b4",  
        "MultipleLines": "#ff7f0e",  
        "InternetService": "#2ca02c",  
        "OnlineSecurity": "#d62728",  
        "DeviceProtection": "#9467bd",  
        "TechSupport": "#8c564b",  
        "StreamingTV": "#e377c2",  
        "StreamingMovies": "#7f7f7f"  
    }

    # Function to create a horizontal bar chart
    def create_service_chart(df, service, title, color):
        service_churn = df.groupby([service, "Churn"]).size().reset_index(name="Count")
        service_churn["Churn"] = service_churn["Churn"].astype(str)

        fig = px.bar(service_churn, 
                    y=service, 
                    x="Count", 
                    color="Churn",
                    orientation="h", 
                    text="Count",
                    title=title,
                    color_discrete_map={"1": color, "0": "#17becf"})  # Use custom color for churn & blue for non-churn
        
        fig.update_traces(texttemplate='%{text}', textposition='outside')
        fig.update_layout(xaxis_title="Number of Customers", yaxis_title="Service Availability")
        
        return fig

    # Show chart
    st.plotly_chart(create_service_chart(df, selected_service, f"Churn Impact on {selected_service}", color_map[selected_service]))

    # Insights based on selected service
    st.markdown("## Key Insights")

    if selected_service == "PhoneService":
        st.markdown("""
        - **Over 70% of customers with phone service do not churn.**
        - Phone service alone does not appear to be a strong factor in churn.
        """)

    elif selected_service == "MultipleLines":
        st.markdown("""
        - **Churn rate is slightly higher for customers with multiple lines.**
        - Customers with single lines may be more stable.
        """)

    elif selected_service == "InternetService":
        st.markdown("""
        - **DSL and Fiber customers show high retention (~75%).**
        - **Customers without internet service have the highest retention (~90%)**.
        """)

    elif selected_service == "OnlineSecurity":
        st.markdown("""
        - **Customers with Online Security have significantly lower churn.**
        - This suggests security features improve retention.
        """)

    elif selected_service == "DeviceProtection":
        st.markdown("""
        - **Device protection does not significantly affect churn.**
        - Customers with or without this service churn at similar rates.
        """)

    elif selected_service == "TechSupport":
        st.markdown("""
        - **Customers with Tech Support have noticeably lower churn.**
        - Indicates that strong support services may encourage retention.
        """)

    elif selected_service == "StreamingTV":
        st.markdown("""
        - **Churn rates are similar regardless of TV streaming availability.**
        - Streaming services do not appear to be a major factor in churn.
        """)

    elif selected_service == "StreamingMovies":
        st.markdown("""
        - **Like StreamingTV, movies do not strongly influence churn.**
        - Customers may not see these services as essential.
        """)


elif section == "Churn Analysis by Account Features":
    # Sidebar Selection  
    st.sidebar.subheader("Churn Analysis by Account Features")  
    account_feature = st.sidebar.selectbox(  
        "Select an Account Feature:",  
        ["Contract", "PaperlessBilling", "PaymentMethod"]  
    )  
 

    # Compute Churn Rate 
    churn_data = df.groupby([account_feature, "Churn"]).size().reset_index(name="Count")  
    churn_data["Churn"] = churn_data["Churn"].astype(str)  # Convert to string 

    # Defining colors 
    color_map = {"0": "#3498DB", "1": "#E74C3C"}  

    # Create Horizontal Bar Chart  
    fig = px.bar(  
        churn_data,  
        x="Count",  
        y=account_feature,  
        color="Churn",  
        text="Count",  
        barmode="group",  
        orientation="h",  
        color_discrete_map=color_map,  
        title=f"Churn Rate by {account_feature}"  
    )  

    fig.update_traces(texttemplate='%{text}', textposition='outside')  
    fig.update_layout(xaxis_title="Number of Customers", yaxis_title=account_feature)  

    # Display Chart  
    st.plotly_chart(fig, use_container_width=True)  

    # Display Insights  
    st.markdown(f"### Key Insights: {account_feature} & Churn")  
    if account_feature == "Contract":  
        st.markdown(  
            """  
            - **Month-to-month contracts** have the highest churn rate, indicating that short-term plans are risky.  
            - **Long-term contracts (1-year, 2-year)** have significantly lower churn, suggesting customer commitment.  
            - **Encouraging customers to switch to longer contracts may improve retention.**  
            """  
        )  
    elif account_feature == "PaperlessBilling":  
        st.markdown(  
            """  
            - Customers using **Paperless Billing** have a **higher churn rate**, possibly due to online billing concerns.  
            - Customers with **physical bills have lower churn**, indicating trust in traditional payment methods.  
            - **Improving the digital billing experience might reduce churn.**  
            """  
        )  
    elif account_feature == "PaymentMethod":  
        st.markdown(  
            """  
            - Customers using **Electronic Checks have the highest churn rate (45%)**, possibly due to failed transactions.  
            - **Automatic payments (Credit Card, Bank Transfer) have the lowest churn**, suggesting that auto-pay reduces customer loss.  
            - **Encouraging automatic payments can enhance retention.**  
            """  
        )  
elif section == "Financial Metrics, Contract & Billing":
    # Financial Metrics
    st.sidebar.subheader("Financial Metrics")

    # Selectbox for choosing a visualization
    financial_chart_type = st.sidebar.selectbox(
        "Select a chart:",
        ["Monthly Charges Distribution by Churn", "Total Charges vs. Tenure (Trend Line)", "Churn Rate by Payment Method (Normalized)"],
        key="financial_chart_select"
    )

    # Ensure numeric conversion
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["tenure"] = pd.to_numeric(df["tenure"], errors="coerce")
    df_clean = df.dropna(subset=["TotalCharges", "tenure"])

    # Display
    if financial_chart_type == "Monthly Charges Distribution by Churn":
        st.subheader("Monthly Charges & Churn")

        fig_monthly_charges = px.violin(df, 
            x="Churn", 
            y="MonthlyCharges", 
            box=True, 
            points="all", 
            color="Churn",
            title="Monthly Charges Distribution by Churn",
            color_discrete_map={"Yes": "#E74C3C", "No": "#3498DB"}
        )

        fig_monthly_charges.update_layout(
            yaxis_title="Monthly Charges ($)", 
            xaxis_title="Churn", 
            violinmode="overlay"
        )

        st.plotly_chart(fig_monthly_charges, use_container_width=True)

        # Insights 
        st.markdown(
            """
            - **Higher Monthly Charges Correlate with Higher Churn**: Customers with higher monthly charges tend to churn more.
            - **Price Sensitivity Impact**: The highest churn is in the **$60-$100 range**, indicating cost concerns.
            - **Retention Strategy**: Consider **discounts, loyalty rewards, or tiered pricing plans** to retain high-paying customers.
            """
        )

    elif financial_chart_type == "Total Charges vs. Tenure (Trend Line)":
        st.subheader("Total Charges vs. Tenure")

        fig, ax = plt.subplots(figsize=(8, 5))
        sns.regplot(data=df_clean, 
            x="tenure", 
            y="TotalCharges", 
            scatter_kws={"alpha": 0.5}, 
            line_kws={"color": "red"}
        )
        plt.title("Total Charges vs. Tenure (With Trend Line)")
        plt.xlabel("Tenure (Months)")
        plt.ylabel("Total Charges ($)")
        st.pyplot(fig)

        # Insights 
        st.markdown(
            """
            - **Strong Positive Correlation**: Higher tenure customers have **higher total charges**.
            - **Billing Anomalies**: Some long-tenure customers have **low total charges**, possibly due to discounts.
            - **Retention Strategy**: Identifying customers with **unexpectedly low charges** could uncover **billing errors or discount strategies**.
            """
        )

    elif financial_chart_type == "Churn Rate by Payment Method (Normalized)":
        st.subheader("Churn Rate by Payment Method")

        payment_method_churn = df.groupby(["PaymentMethod", "Churn"]).size().unstack()
        payment_method_churn = payment_method_churn.div(payment_method_churn.sum(axis=1), axis=0) * 100  

        payment_method_churn = payment_method_churn.reset_index().melt(id_vars="PaymentMethod")

        fig_payment_method = px.bar(payment_method_churn, 
            x="PaymentMethod", 
            y="value", 
            color="Churn", 
            barmode="stack",
            text="value",
            title="Churn Rate by Payment Method (Normalized)",
            labels={"value": "Percentage (%)"},
            color_discrete_map={"Yes": "#E74C3C", "No": "#3498DB"}
        )

        fig_payment_method.update_traces(texttemplate='%{text:.2f}%', textposition='inside')
        fig_payment_method.update_layout(yaxis_title="Percentage (%)", xaxis_title="Payment Method")

        st.plotly_chart(fig_payment_method, use_container_width=True)

        # Insights
        st.markdown(
            """
            - **Electronic Check Users Have Highest Churn (~45%)**: Possible issues with payment convenience or failed transactions.
            - **Automatic Payments (Bank Transfer & Credit Card) Reduce Churn**: Churn rate is significantly lower (~16%).
            - **Retention Strategy**: Encouraging **automatic payments** could help **reduce churn rates**.
            """
        )



    # Contract & Billing
    st.sidebar.subheader("Contract & Billing")

    # Dropdown for selecting chart
    contract_chart_option = st.sidebar.selectbox(
        "Select a Chart to View:",
        [
            "Churn Rate by Contract Type",
            "Churn Rate by Paperless Billing",
            "Churn Rate by Contract Type & Paperless Billing",
        ],
        key="contract_chart_selection",
    )

    if contract_chart_option == "Churn Rate by Contract Type":
        # Compute Churn Rate
        contract_churn = df.groupby(["Contract", "Churn"]).size().unstack(fill_value=0)
        contract_churn_percentage = contract_churn.div(contract_churn.sum(axis=1), axis=0) * 100  # Normalize

        contract_churn_percentage.columns = contract_churn_percentage.columns.astype(str)
        # Bar Chart
        fig_contract = px.bar(
            contract_churn_percentage.reset_index(),
            x="Contract",
            y=contract_churn_percentage.columns.tolist(),  # Retained (0) and Churned (1)
            barmode="stack",
            title="Churn Rate by Contract Type",
            labels={"value": "Percentage", "Contract": "Contract Type"},
            color_discrete_map={'0': "#3498DB", '1': "#F39C12"}  # Blue for retained, Orange for churned
        )

        fig_contract.update_layout(yaxis_title="Churn Rate (%)", xaxis_title="Contract Type")
        st.plotly_chart(fig_contract, use_container_width=True)

        # Insights
        st.markdown(
            """
            **Churn Rate by Contract Duration**  
            -  **Month-to-month contracts** have the **highest churn rate**, meaning those with the shortest contracts are most likely to churn.  
            -  **One-year & two-year contracts** show **significantly lower churn**, confirming that longer contracts help with retention.  
            -  **Customers with longer contracts** likely feel more committed or incentivized to stay.  

            **Conclusion:**  
            - Encouraging **month-to-month users to switch to long-term contracts** can help **reduce churn**.  
            - Offering **better incentives for long-term plans** could further enhance retention.
            """
        )

    elif contract_chart_option == "Churn Rate by Paperless Billing":
        # Compute Churn Rate by Paperless Billing
        paperless_churn = df.groupby(["PaperlessBilling", "Churn"]).size().unstack(fill_value=0)
        paperless_churn_percentage = paperless_churn.div(paperless_churn.sum(axis=1), axis=0) * 100  # Normalize to %

        # Stacked Bar Chart
        fig_paperless = px.bar(
            paperless_churn_percentage.reset_index(),
            x="PaperlessBilling",
            y=paperless_churn_percentage.columns.tolist(),  # Retained (0) and Churned (1)
            barmode="stack",
            title="Churn Rate by Paperless Billing",
            labels={"value": "Percentage", "PaperlessBilling": "Paperless Billing"},
            color_discrete_map={"0": "#9B59B6", "1": "#1ABC9C"}  # Purple for retained, Teal for churned
        )

        fig_paperless.update_layout(yaxis_title="Churn Rate (%)", xaxis_title="Paperless Billing")
        st.plotly_chart(fig_paperless, use_container_width=True)

        # Insights
        st.markdown(
            """
            **Churn Rate by Paperless Billing**  
            -  **Customers with paperless billing churn more**, which may indicate disengagement or **missed auto-payments**.  
            -  **Traditional billing users** show lower churn, suggesting they may be **more engaged** or aware of their bills.  
            -  **Paperless customers might need better communication or reminders** to improve retention.  

            **Conclusion:**  
            - Improving **paperless billing experience** with **clearer notifications** could reduce churn.  
            - **Offering incentives for online billing adoption** while addressing common pain points may enhance retention.
            """
        )

    elif contract_chart_option == "Churn Rate by Contract Type & Paperless Billing":
        # Compute Churn Rate by Contract & Paperless Billing
        df['Churn'] = df['Churn'].replace({'Yes': 1, 'No': 0})
        contract_paperless_churn = df.groupby(["Contract", "PaperlessBilling"])["Churn"].mean().unstack() * 100

        # Seaborn Heatmap
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.heatmap(contract_paperless_churn, annot=True, cmap="coolwarm", fmt=".1f", linewidths=0.5, ax=ax)
        ax.set_title("Churn Rate by Contract Type & Paperless Billing")
        ax.set_xlabel("Paperless Billing")
        ax.set_ylabel("Contract Type")

        st.pyplot(fig)

        # Insights
        st.markdown(
            """
            **Churn Rate by Contract Type & Paperless Billing**  
            -  **Month-to-month contracts have the highest churn**, especially with **paperless billing (48.3%)**.  
            -  **One-year contracts show moderate churn**, but churn is **higher for paperless users (7.1% → 14.8%)**.  
            -  **Two-year contracts have the lowest churn**, showing strong customer retention regardless of billing type.  

            **Conclusion:**  
            - Encouraging **longer-term contracts** can significantly **reduce churn**.  
            - **Paperless billing users in short-term contracts** may need better **engagement strategies or incentives**.
            """
        )
elif section == "Advanced Insights":
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import LabelEncoder
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    import plotly.figure_factory as ff
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression

    # Section Header
    st.sidebar.subheader("Advanced Insights")

    # Dropdown for selecting visualization
    advanced_chart_option = st.sidebar.selectbox(
        "Select a Chart to View:",
        [
            "Feature Importance for Churn Prediction",
            "Customer Segmentation Based on Charges",
            "Correlation Heatmap",
            "Churn Probability Distribution",
        ],
        key="advanced_chart_selection",
    )

    if advanced_chart_option == "Feature Importance for Churn Prediction":
        # Train Random Forest Classifier
        df_ml = df.copy()

        # Encode categorical features
        for col in df_ml.select_dtypes(include=['object']).columns:
            if col != "Churn":
                df_ml[col] = LabelEncoder().fit_transform(df_ml[col])

        X = df_ml.drop(columns=['Churn', 'customerID'])
        y = df_ml['Churn']

        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X, y)

        # Feature Importance
        feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': rf_model.feature_importances_})
        feature_importance = feature_importance.sort_values(by="Importance", ascending=True)

        # Ploting Feature Importance
        fig_feature_importance = px.bar(
            feature_importance,
            x="Importance",
            y="Feature",
            orientation='h',
            title="Feature Importance for Churn Prediction",
            text="Importance",
            color="Importance",
            color_continuous_scale="viridis"
        )

        fig_feature_importance.update_traces(texttemplate='%{text:.2f}', textposition='outside')
        fig_feature_importance.update_layout(xaxis_title="Importance Score", yaxis_title="Feature", showlegend=False)

        st.plotly_chart(fig_feature_importance, use_container_width=True)

        # Insights
        st.markdown(
            """
            **Feature Importance for Churn Prediction**  
            -  **Total Charges (0.19)** is the most significant factor impacting churn.  
            -  **Monthly Charges (0.18)** follow closely, showing price sensitivity in churn behavior.  
            -  **Tenure (0.16)** indicates that **longer-tenured customers are less likely to churn**.  
            -  **Contract Type (0.08)** highlights the importance of **long-term agreements** in retention.  
            -  **Security & Support Services (0.05)** have a moderate influence on churn behavior.  

            **Conclusion:**  
            - Financial factors dominate churn prediction.  
            - Encouraging **longer contracts** and **improving service quality** can help reduce churn.  
            """
        )

    elif advanced_chart_option == "Customer Segmentation Based on Charges":
        # K-Means Clustering
        df_cluster = df[['tenure', 'MonthlyCharges', 'TotalCharges']].copy()
        df_cluster['TotalCharges'] = pd.to_numeric(df_cluster['TotalCharges'], errors='coerce')
        df_cluster.fillna(df_cluster.median(), inplace=True)

        scaler = StandardScaler()
        df_cluster_scaled = scaler.fit_transform(df_cluster)

        kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
        df['Cluster'] = kmeans.fit_predict(df_cluster_scaled)

        # Scatter plot
        fig_clusters = px.scatter(
            df,
            x="TotalCharges",
            y="MonthlyCharges",
            color="Cluster",
            hover_data=['tenure'],
            title="Customer Segments Based on Charges",
            color_continuous_scale="plasma"
        )

        fig_clusters.update_layout(xaxis_title="Total Charges", yaxis_title="Monthly Charges", showlegend=True)
        st.plotly_chart(fig_clusters, use_container_width=True)

        # Insights
        st.markdown(
            """
            **Customer Segmentation by Charges**  
            - **Cluster 0:** Long-term customers with high total charges but low monthly charges.  
            - **Cluster 1:** Valuable customers with high recurring spending.  
            - **Cluster 2:** New customers with low total and monthly charges.  
            - **Cluster 3:** High monthly spenders but newer to the service.  

            **Conclusion:**  
            - **Retain high-value customers (Clusters 0 & 1).**  
            - **Upsell opportunities exist for Cluster 2 customers.**  
            - **Monitor Cluster 3 for potential churn risk.**  
            """
        )

    elif advanced_chart_option == "Correlation Heatmap":
        # Ensure 'Churn' is numeric before correlation computation
        df['Churn'] = df['Churn'].replace({'Yes': 1, 'No': 0}).astype(int)

        # Compute correlation matrix
        corr_matrix = df.select_dtypes(include=['number']).corr()

        # Filter key correlations with Churn
        if "Churn" not in corr_matrix.columns:
            st.error("Error: 'Churn' column is missing in correlation matrix.")
        else:
            corr_target = corr_matrix["Churn"].drop("Churn").sort_values(ascending=False)

            # Select top 8-10 features
            top_features = corr_target[abs(corr_target) > 0.1].index.tolist()[:10]  

            # Ensure 'Churn' is always included in the heatmap
            if "Churn" not in top_features:
                top_features.append("Churn")

            # Compute filtered correlation matrix
            filtered_corr_matrix = df[top_features].corr().fillna(0)

            # Round values for annotation
            annotation_text = [[f"{val:.2f}" for val in row] for row in filtered_corr_matrix.values]

            # Create Heatmap
            fig_corr = ff.create_annotated_heatmap(
                z=filtered_corr_matrix.values,
                x=filtered_corr_matrix.columns.tolist(),
                y=filtered_corr_matrix.columns.tolist(),
                annotation_text=annotation_text,  
                colorscale="RdBu",
                showscale=True,
                autocolorscale=False
            )

            fig_corr.update_layout(
                title="Correlation Heatmap (Top Features)",
                xaxis_title="Features",
                yaxis_title="Features",
                autosize=True
            )

            st.plotly_chart(fig_corr, use_container_width=True)




        st.markdown("""
## Analysis of Correlation Heatmap

The correlation heatmap highlights key relationships between important customer features and churn. The darker the shade, the stronger the correlation.

### **Key Observations:**

- **Churn Factors:**
- **Negative correlation with tenure (-0.35):** Longer-tenured customers are less likely to churn.
- **Weak negative correlation with Total Charges (-0.20):** Higher accumulated spending slightly reduces churn risk.
- **Weak positive correlation with Monthly Charges (0.19):** Customers paying higher monthly fees have a slightly higher likelihood of churn.
- **Weak positive correlation with Senior Citizen (0.15):** Senior citizens show a marginally higher churn tendency.

- **Feature Interactions:**
- **Strong correlation between Tenure & Total Charges (0.83):** Customers with longer tenures naturally accumulate higher total charges.
- **Moderate correlation between Monthly Charges & Total Charges (0.65):** Higher monthly fees contribute to higher total spending.
- **Weak correlation between Senior Citizen & Monthly Charges (0.22):** Senior citizens tend to have slightly higher monthly fees.

### **Summary & Business Takeaways:**
- Customers with **longer tenure tend to stay longer**, reinforcing the importance of early retention efforts.
- **Higher monthly charges slightly increase churn risk**, suggesting the need for pricing strategies like loyalty discounts.
- **Senior citizens have slightly higher churn**, indicating that specialized retention efforts may be required for this group.
""")


    elif advanced_chart_option == "Churn Probability Distribution":
        # Churn Prediction
        features = df.drop(columns=["customerID", "Churn"])
        features = pd.get_dummies(features, drop_first=True)
        features.fillna(features.median(numeric_only=True), inplace=True)
        target = df["Churn"]

        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = LogisticRegression()
        model.fit(X_train_scaled, y_train)
        churn_probabilities = model.predict_proba(X_test_scaled)[:, 1]

        # Plot Histogram
        fig_churn_prob = px.histogram(
            pd.DataFrame({'Churn Probability': churn_probabilities}),
            x="Churn Probability",
            nbins=20,
            title="Churn Probability Distribution",
            labels={"Churn Probability": "Predicted Churn Probability"},
            color_discrete_sequence=["#E74C3C"]
        )

        fig_churn_prob.update_layout(
            xaxis_title="Predicted Churn Probability",
            yaxis_title="Number of Customers",
            bargap=0.1
        )

        st.plotly_chart(fig_churn_prob, use_container_width=True)

        # Insights
        st.markdown(
            """
            **Churn Probability Distribution**  
            -  Most customers have a **low churn probability (~0.0-0.2)**.  
            -  **Some customers fall in the 0.6+ range**, indicating **high churn risk**.  
            -  **Identifying high-risk customers can help in early intervention.**  

            **Business Strategy:**  
            - **Focus on moderate-risk customers (0.1 - 0.2) with engagement strategies.**  
            - **Target high-risk customers (0.6+) with retention offers & personalized outreach.**  
            """
        )
    
