import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant # Added for proper VIF calculation
import matplotlib.pyplot as plt

@st.cache_data
def load_data():
    """Load the dataset"""
    data = {'Month': ["Jan-24","Feb-24","Mar-24","Apr-24","May-24","Jun-24","Jul-24",
                   "Aug-24","Sep-24","Oct-24","Nov-24","Dec-24","Jan-25","Feb-25","Mar-25",
                   "Apr-25","May-25","Jun-25"],
        'Dyed_Garment': [81296.7,26800.7,23528.9,37113.6,52909.0,66963.6,
                         69193.5,95459.1,67589.0,86473.8,89504.0,68004.8,116687.6,84678.7,
                         123700.5,93761.7,93139.4,113947.4],
        'Reknit_Garment': [11834.6,3218.5,997.2,744.8,1793.3,3515.3,7239,4940.1,
                          10291.3,6209,4836.8,4875.8,7163,17319.8,12799.9,7800.8,3883.1,
                           6280.5],
        'Additional_Garment': [30047,10915,3718,12632.2,19484.3,22333.3,
                              28335,23265.6,18574.6,23287.4,31385.6,42982.9,29800,32172,
                              20810,17316,18412,27934],
        'Hotwater_usage': [3257,1672,1170,1931,1795,2560,3075,3265,2288,3008,
                           2670,3574,3669,3432,4501,4064,4054,8081],
        'Process_Water': [23302,8604,6079,9989,16933,20358,19794,23342,21692,
                         27054,24116,24648,26335,24662,29216,21099,22670,24617]}
    
    return pd.DataFrame(data)

@st.cache_resource
def train_model(data):
    """Train the regression model"""
    X = data[['Dyed_Garment','Reknit_Garment','Additional_Garment','Hotwater_usage']]
    Y = data['Process_Water']

    model = LinearRegression()
    model.fit(X,Y)

    return model, X, Y

def calculate_metrics(model, X, Y):
    """Calculate model performance metrics"""
    Y_pred = model.predict(X)
    residuals = Y - Y_pred

    r2 = r2_score(Y,Y_pred)
    adj_r2 = 1 - (1-r2) *(len(Y)-1) / (len(Y) - X.shape[1] - 1)
    rmse = np.sqrt(mean_squared_error(Y,Y_pred))
    mae = np.mean(np.abs(residuals))

    shapiro_stat, shapiro_p = stats.shapiro(residuals)

    # Calculate VIF properly - add constant for intercept
    X_with_const = add_constant(X.values)
    vif_values = [variance_inflation_factor(X_with_const, i+1) for i in range(len(X.columns))]
    
    return{
        'r2': r2,
        'adj_r2': adj_r2,
        'rmse': rmse,
        'mae': mae,
        'shapiro_stat': shapiro_stat,
        'shapiro_p': shapiro_p,
        'vif': dict(zip(X.columns, vif_values)),
        'residuals': residuals,
        'predictions': Y_pred
    }

# --- START NEW HELPER FUNCTION DEFINITION ---
def display_actual_vs_low_savings(low_pred, actual_water):
    """Calculates and displays the Actual Water vs Low Scenario savings."""
    
    # Only calculate if actual water is provided and comparison is possible
    if actual_water > 0 and actual_water != low_pred:
        
        # Calculations using Actual Water as the starting point (Actual - Low Prediction)
        actual_water_saving = low_pred - actual_water
        actual_waste_water_saving = actual_water_saving * 0.9 # Using your 90% waste factor
        actual_water_saving_usd = (actual_water_saving * 165) / 290
        actual_waste_water_saving_usd = actual_waste_water_saving * 1.15

        st.markdown("---")
        st.markdown("<h3 style='text-align:left;font-size:24px;'>💸 Potential Savings (Actual vs Low Scenario) :</h3>", unsafe_allow_html=True)
        #st.caption(f"Savings if you had run the process using **Low Scenario** parameters ({low_pred:,.0f} m³) instead of the **Actual Recorded Usage** ({actual_water:,.0f} m³).")

        col_a1, col_a2, col_a3, col_a4 = st.columns(4)
        
        with col_a1:
            st.metric("🔹💧 Water Saving", f"{actual_water_saving:,.0f} m³")
        with col_a2:
            st.metric("🔹🗑️ Waste Water Saving", f"{actual_waste_water_saving:,.0f} m³")
        with col_a3:
            st.metric("🔸💰 Saving USD (Water)", f"${actual_water_saving_usd:,.2f}")
        with col_a4:
            st.metric("🔸💰 Saving USD (Waste Water)", f"${actual_waste_water_saving_usd:,.2f}")
            
        st.markdown("---")
# --- END NEW HELPER FUNCTION DEFINITION ---


#Page configuration
st.set_page_config(
    page_title = "Water Usage Prediction",
    page_icon = "💧",
    layout = "wide"
)

# Custom CSS styling
st.markdown("""
<style>
    /*Reduce white space at the top*/
    .main.block-container {
        padding-top:1rem;
        padding-bottom:2rem;
        min-height:100vh;
    }
    /* Target only the main content area, not sidebar */
    [data-testid="stMain"] {
        margin-top: -70px;
    }

    /* Keep the sidebar intact */
    [data-testid = "stSidebar"]{
        margin-top: 0px;
    }
    /* Button styling */
    .stButton button{
        width: 100%;
        height: 50px;
        font-size: 16px;
        font-weight: bold;
    }
</style>
""",unsafe_allow_html=True)

#Dashboard Title
st.markdown("""
    <h1 style = "text-align: center; font-family: 'Lato',
    sans-serif; color: #fafafa;">
    💧 Water Consumption Optimization
    </h1>
""",unsafe_allow_html=True)

data = load_data()
model, X, Y = train_model(data)
metrics = calculate_metrics(model, X, Y)

#Sidebar
st.sidebar.title("Menu")

#Initialize session state for selected menu
if 'selected_menu' not in st.session_state:
    st.session_state.selected_menu = "Model"

#Menu options
menu_options = ["Model","Prediction","Saving"]

#create menu buttons
for option in menu_options:
    is_active = st.session_state.selected_menu == option
    button_type = "primary" if is_active else "secondary"

    if st.sidebar.button(option,key = option,type=button_type,use_container_width=True):
        st.session_state.selected_menu = option
        st.rerun()

#Main content based on selected menu
if st.session_state.selected_menu == "Model":

    st.markdown("<h2 style='text-align:center;font-size:30px;'> ⚛ Model Overview </h2>",unsafe_allow_html=True)

    #Display Dataset
    with st.expander("📑 View Dataset", expanded=False):
        st.dataframe(
            data.style.format({
                'Dyed_Garment': '{:.1f}',
                'Reknit_Garment': '{:.1f}',
                'Additional_Garment': '{:.1f}',
                'Hotwater_usage': '{:.0f}',
                'Process_Water': '{:.0f}'
        }),
        use_container_width=True,
        hide_index=True
        )

    st.caption(f"**Total Records**:{len(data)}months&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**Dyed Garment** in Kg&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**Reknit Garment** in Kg&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**Additional Garment** in Kg&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**Hotwater Usage** in m³&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**Process Water** in m³")

    st.markdown("---")

    #Regression Equation
    st.markdown("<h2 style='text-align:left;font-size:24px;margin-bottom:-60px;'>♾️ Regression Equation : <h2>",unsafe_allow_html=True)

    st.latex(r"\text{Process Water} = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \beta_3 X_3 + \beta_4 X_4")
    st.write("**Expanded Form:**")
    st.code(
        f"Process Water = {model.intercept_:.2f}\n"
        f"              + ({model.coef_[0]:.6f}) × Dyed Garment\n"
        f"              + ({model.coef_[1]:.6f}) × Reknit Garment\n"
        f"              + ({model.coef_[2]:.6f}) × Additional Garment\n"
        f"              + ({model.coef_[3]:.6f}) × Hotwater usage"
    )

    st.latex(r"\text{Process Water (m³)} = \beta_0 + \beta_1 \text{Dyed (kg)} + \beta_2 \text{Reknit (kg)} + \beta_3 \text{Additional (kg)} + \beta_4 \text{Hotwater (m³)}")

    st.markdown("---")

    #Coefficients Table
    st.markdown("<h3 style='text-align:left;font-size:24px;margin-bottom:20px;'>💡 Model Coefficients and Explanation : </h3>",unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Intercept (β₀):**")
        st.info(f"**{model.intercept_:.2f}**\n\nThis is the baseline **Process Water** when all **Other variables(Kg)** and **hot water usage(m³)** are zero.")
        
        st.write("**Dyed Garment (β₁):**")
        if model.coef_[0] > 0:
            st.success(f"**{model.coef_[0]:.6f}** (Positive)\n\nFor every 1Kg of additional **Dyed Garment**, **Process Water** increases by {model.coef_[0]:.6f} m³.")
        else:
            st.warning(f"**{model.coef_[0]:.6f}** (Negative)\n\nFor every 1kg of additional **Dyed Garment**, **Process Water** decreases by {abs(model.coef_[0]):.6f} m³.")
    
    with col2:
        st.write("**Reknit Garment (β₂):**")
        if model.coef_[1] > 0:
            st.success(f"**{model.coef_[1]:.6f}** (Positive)\n\nFor every 1kg of additional **Reknit Garment**, **Process Water** increases by {model.coef_[1]:.6f} m³.")
        else:
            st.warning(f"**{model.coef_[1]:.6f}** (Negative)\n\nFor every 1kg of additional **Reknit Garment**, **Process Water** decreases by {abs(model.coef_[1]):.6f} m³.")
        
        st.write("**Additional Garment (β₃):**")
        if model.coef_[2] > 0:
            st.success(f"**{model.coef_[2]:.6f}** (Positive)\n\nFor every 1kg of **Additional Garment**, **Process Water** increases by {model.coef_[2]:.6f} m³.")
        else:
            st.warning(f"**{model.coef_[2]:.6f}** (Negative)\n\nFor every 1kg of **Additional Garment**, **Process Water** decreases by {abs(model.coef_[2]):.6f} m³.")
    
    st.write("**Hotwater Usage (β₄):**")
    if model.coef_[3] > 0:
        st.success(f"**{model.coef_[3]:.6f}** (Positive)\n\nFor every 1m³ increase in **Hot Water**, **Process Water** increases by {model.coef_[3]:.6f} m³.")
    else:
        st.warning(f"**{model.coef_[3]:.6f}** (Negative)\n\nFor every 1m³ unit increase in **Hot Water**, we can reduce **Process Water** by {abs(model.coef_[3]):.6f} m³.")
    
    st.markdown("---")

    #Performance Metrics
    st.markdown("<h3 style='text-align:left;font-size:24px;margin-bottom:20px;'>🌟Model Performance Metrics : </h3>",unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="R² Score",
            value=f"{metrics['r2']:.4f}",
            help="Proportion of variance explained by the model"
        )
        st.caption(f"**{metrics['r2']*100:.2f}%** variance explained")
    
    with col2:
        st.metric(
            label="Adjusted R²",
            value=f"{metrics['adj_r2']:.4f}",
            help="R² adjusted for number of predictors"
        )
    
    with col3:
        st.metric(
            label="RMSE",
            value=f"{metrics['rmse']:.2f}",
            help="Root Mean Squared Error (lower is better)"
        )
        st.caption("Lower is better")
    
    with col4:
        st.metric(
            label="MAE",
            value=f"{metrics['mae']:.2f}",
            help="Mean Absolute Error"
        )
        st.caption("Mean error")
    
    #Performance interpretation
    if metrics['r2'] > 0.9:
        st.success("✅ **Excellent model fit** - R² > 0.9")
    elif metrics['r2'] > 0.7:
        st.info("✅ **Good model fit** - R² > 0.7")
    elif metrics['r2'] > 0.5:
        st.warning("⚠️ **Moderate model fit** - R² > 0.5")
    else:
        st.error("❌ **Poor model fit** - R² < 0.5")
    
    st.markdown("---")

    # Residual Diagnostics
    st.markdown("<h3 style='text-align:left;font-size:24px;margin-bottom:20px;'>🧐 Residual Diagnostics : </h3>",unsafe_allow_html=True)
    
    st.caption("Residuals are the differences between actual and predicted values. They should be normally distributed for a good model.")

    col1, col2 = st.columns(2)

    with col1:
        #Histogram of residuals
        fig1, ax1 = plt.subplots(figsize=(8,6))
    
        ax1.hist(metrics['residuals'],bins=15,color='lightblue',
            edgecolor='black',alpha=0.7,density=True)
    
        #Add normal curve
        mu = np.mean(metrics['residuals'])
        sigma = np.std(metrics['residuals'])
        x = np.linspace(metrics['residuals'].min(),metrics['residuals'].max(),100)
        ax1.plot(x, stats.norm.pdf(x, mu, sigma), 'r--',linewidth=2,
            label='Normal Distribution')
    
        ax1.axvline(mu, color='darkgreen', linestyle='-', linewidth=2, 
                    label=f'Mean = {mu:.2f}')
    
        ax1.set_xlabel('Residuals',fontsize=11)
        ax1.set_ylabel('Density',fontsize=11)
        ax1.set_title('Histogram of Residuals', fontsize=13, fontweight='bold')
        ax1.legend()
        ax1.grid(True,alpha=0.3)

        st.pyplot(fig1)
        plt.close()
    
    with col2:
        #Q-Q Plot
        fig2, ax2 = plt.subplots(figsize=(8,6))

        stats.probplot(metrics['residuals'],dist="norm",plot=ax2)
        ax2.set_title('Q-Q Plot of Residuals', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.get_lines()[0].set_markerfacecolor('lightblue')
        ax2.get_lines()[0].set_markeredgecolor('darkblue')
        ax2.get_lines()[1].set_color('darkred')

        st.pyplot(fig2)
        plt.close()
    
    st.write("**Normality Test (Shapiro-Wilk):**")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Test Statistic", f"{metrics['shapiro_stat']:.6f}")
    
    with col2:
        st.metric("P-value", f"{metrics['shapiro_p']:.6f}")
    
    with col3:
        if metrics['shapiro_p'] > 0.05:
            st.success("✅ Normal")
        else:
            st.warning("⚠️ Non-normal")
    
    if metrics['shapiro_p'] > 0.05:
        st.success(f"✅ Residuals are normally distributed (p-value = {metrics['shapiro_p']:.4f} > 0.05)")
    else:
        st.warning(f"⚠️ Residuals may not be normally distributed (p-value = {metrics['shapiro_p']:.4f} < 0.05)")
    
    with st.expander("ℹ️ How to interpret these plots?"):
        st.write("**Histogram:**")
        st.write("• Should look like a bell curve (normal distribution)")
        st.write("• Red dashed line shows the ideal normal distribution")
        st.write("• Green line shows the mean of residuals (should be close to 0)")
        
        st.write("\n**Q-Q Plot:**")
        st.write("• Points should follow the red diagonal line")
        st.write("• Deviations indicate non-normality")
        
        st.write("\n**Shapiro-Wilk Test:**")
        st.write("• Tests if residuals are normally distributed")
        st.write("• p-value > 0.05: Residuals are normal ✅")
        st.write("• p-value < 0.05: Residuals may not be normal ⚠️")

    st.markdown("---")

    #Multicollinearity check (VIF)
    st.markdown("<h2 style='text-align:left;font-size:24px;margin-top:-10px'>👉 Multicollinearity Check (VIF)</h2>",unsafe_allow_html=True)

    st.caption("VIF (Variance Inflation Factor) checks if predictor variables are too correlated with each other.")

    #Create VIF dataframe
    vif_df = pd.DataFrame({
        'Variables': list(metrics['vif'].keys()),
        'VIF': list(metrics['vif'].values()),
        'Status': ['✅ Good' if v < 5 else ('⚠️ Moderate' if v < 10 else '❌ High') 
                    for v in metrics['vif'].values()]
    })

    col1, col2 = st.columns([2,1])

    with col1:
        st.dataframe(
            vif_df.style.format({'VIF':'{:.2f}'}),
            use_container_width=True,
            hide_index=True
        )
    
    with col2:
        st.write("**VIF Guidelines:**")
        st.write("✅ **VIF < 5:** No problem")
        st.write("⚠️ **VIF 5-10:** Moderate")
        st.write("❌ **VIF > 10:** High correlation")
    
    # Check if any VIF is problematic
    max_vif = max(metrics['vif'].values())
    if max_vif < 5:
        st.success("✅ No multicollinearity detected - All VIF values are below 5")
    elif max_vif < 10:
        st.warning("⚠️ Moderate multicollinearity detected - Some VIF values between 5-10")
    else:
        st.error("❌ High multicollinearity detected - Some VIF values above 10")
    
    with st.expander("ℹ️ What is VIF and why does it matter?"):
        st.write("**VIF (Variance Inflation Factor):**")
        st.write("• Measures how much the variance of a coefficient is inflated due to correlation with other predictors")
        st.write("• VIF = 1: No correlation with other variables")
        st.write("• VIF > 10: High correlation (problematic)")
        
        st.write("\n**Why it matters:**")
        st.write("• High VIF means predictor variables are correlated")
        st.write("• This makes it hard to determine individual variable effects")
        st.write("• Can lead to unstable coefficient estimates")
        
        st.write("\n**What to do if VIF is high:**")
        st.write("• Remove one of the correlated variables")
        st.write("• Combine correlated variables")
        st.write("• Use regularization techniques")


elif st.session_state.selected_menu == "Prediction":

    st.markdown("<h2 style='text-align:center;font-size:30px;'> ✦︎ Prediction</h2>",unsafe_allow_html=True)

    st.markdown("<br>",unsafe_allow_html=True)

    #Initialize reset counter and store prediction
    if 'reset_counter' not in st.session_state:
        st.session_state.reset_counter = 0
    if 'predicted_water' not in st.session_state:
        st.session_state.predicted_water = None
    if 'prediction_made' not in st.session_state:
        st.session_state.prediction_made = False
    
    #Default values
    default_dyed = 80000.0
    default_reknit = 5000.0
    default_additional = 20000.0
    default_hotwater = 3000.0

    #Create input fields in columns
    col1, col2 = st.columns(2)

    with col1:
        dyed = st.number_input("Dyed Garment (Kg)", min_value=0.0, value=default_dyed, step=1.0, key=f'dyed_{st.session_state.reset_counter}')
        reknit = st.number_input("Reknit Garment (Kg)", min_value=0.0, value=default_reknit, step=1.0, key=f'reknit_{st.session_state.reset_counter}')
    
    with col2:
        additional = st.number_input("Additional Garment (Kg)", min_value=0.0, value=default_additional, step=1.0, key=f'additional_{st.session_state.reset_counter}')
        hotwater = st.number_input("Hotwater Usage (m³)", min_value=0.0, value=default_hotwater, step=1.0, key=f'hotwater_{st.session_state.reset_counter}')
    
    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1,1,1])

    with col2:
        predict_btn = st.button(" ✦︎ Predict Water Usage", type="primary", use_container_width=True)
    
    with col3:
        if st.button("🔄 Reset", type="secondary", use_container_width=True):
            st.session_state.reset_counter += 1
            st.session_state.predicted_water = None
            st.session_state.prediction_made = False
            st.rerun()

    #When predict button is clicked
    if predict_btn:
        #Prepare input data for prediction
        input_data = np.array([[dyed,reknit,additional,hotwater]])

        #Make prediction and store in session state
        st.session_state.predicted_water = model.predict(input_data)[0]
        st.session_state.prediction_made = True

    #Display prediction result if available
    if st.session_state.prediction_made:
        #Display result
        st.markdown("---")
        st.markdown("<h3 style='text-align:left;font-size:24px;'>✨ Prediction Result : </h3>",unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)

        with col2:
            st.metric(
                label="Predicted Process Water Usage",
                value=f"{st.session_state.predicted_water:,.1f} m³",
                help="Predicted water consumption based on your inputs"
            )

        st.markdown("---")
        st.markdown("<h3 style='text-align:left;font-size:24px;'> ⇄ Compare with Actual Value : </h3>",unsafe_allow_html=True)

        col1,col2,col3 = st.columns(3)

        with col2:
            actual_water = st.number_input(
                "Enter Actual Process Water (m³)",
                min_value=0.0,
                value=0.0,
                step=1.0,
                help="Enter the actual water consumption after the month ends"
            )
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            compare_btn = st.button("⇄ Compare Prediction vs Actual : ", type="primary", use_container_width=True)

        # Compare button logic
        if compare_btn and actual_water > 0:
            st.markdown("<br>",unsafe_allow_html=True)

            #Calculate error metrics
            error = actual_water - st.session_state.predicted_water
            error_percentage = (error/actual_water) * 100
            absolute_error = abs(error)

            #Display metrics
            st.markdown("---")
            st.markdown("<h4 style='text-align:left;'>💭 Comparison Results : </h4>",unsafe_allow_html=True)

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    label="Predicted Water",
                    value=f"{st.session_state.predicted_water:,.2f}m³"
                )
            
            with col3:
                st.metric(
                    label="Actual Water",
                    value=f"{actual_water:,.2f}m³"
                )
            
            #Create comparison bar chart
            col1, col2, col3 = st.columns([1,2,1])

            with col2:
                fig, ax = plt.subplots(figsize=(8,6))

                categories = ['Predicted','Actual']
                values = [st.session_state.predicted_water,actual_water]
                colors = ['#64b5f6','#81c784']

                bars = ax.bar(categories, values, color=colors, edgecolor='black', linewidth=1.5)
            
                # Add value labels on bars
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height - (height*0.05),
                        f'{height:,.0f} m³',
                        ha='center', va='bottom', fontsize=12, fontweight='bold')
            
                ax.set_ylabel('Process Water (m³)', fontsize=12, fontweight='bold')
                ax.set_title('Predicted vs Actual Water Usage', fontsize=14, fontweight='bold')
                ax.grid(True, alpha=0.3, axis='y')
                ax.tick_params(axis='x',labelsize=12)
                ax.tick_params(axis='y',labelsize=12)
            
                st.pyplot(fig)
                plt.close()
            
            # Error analysis
            st.markdown("<br>", unsafe_allow_html=True)
            
            if absolute_error / actual_water < 0.05:
                st.success(f"✅ **Excellent Prediction!** Error is only {abs(error_percentage):.2f}% ({absolute_error:,.2f} m³)")
            elif absolute_error / actual_water < 0.10:
                st.info(f"✅ **Good Prediction!** Error is {abs(error_percentage):.2f}% ({absolute_error:,.2f} m³)")
            else:
                st.warning(f"⚠️ **Moderate Error:** {error_percentage:.2f}% ({absolute_error:,.2f} m³)")

elif st.session_state.selected_menu == "Saving":

    st.markdown("<h2 style='text-align:center;font-size:30px;'>💰 Saving </h2>", unsafe_allow_html=True)

    # Two main columns for scenarios
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<h3 style='text-align:center;font-size:18px'>💧 Low Hot Water Scenario</h3>", unsafe_allow_html=True)

        # First row of inputs
        col1a, col1b = st.columns(2)
        with col1a:
            low_dyed = st.number_input("Dyed (Kg)", min_value=0.0, value=st.session_state.get('saved_low_dyed', 80000.0), step=1000.0, key='low_dyed')
        with col1b:
            low_reknit = st.number_input("Reknit (Kg)", min_value=0.0, value=st.session_state.get('saved_low_reknit', 5000.0), step=100.0, key='low_reknit')
        
        # Second row of inputs
        col1c, col1d = st.columns(2)
        with col1c:
            low_additional = st.number_input("Additional (Kg)", min_value=0.0, value=st.session_state.get('saved_low_additional', 20000.0), step=1000.0, key='low_additional')
        with col1d:
            low_hotwater = st.number_input("Hot Water (m³)", min_value=0.0, value=st.session_state.get('saved_low_hotwater', 2000.0), step=100.0, key='low_hotwater')
    
    with col2:
        st.markdown("<h3 style='text-align:center;font-size:18px;'>🔥 High Hot Water Scenario</h3>", unsafe_allow_html=True)

        # First row of inputs
        col2a, col2b = st.columns(2)
        with col2a:
            # We enforce High Scenario inputs to match Low Scenario for fair comparison, except for Hot Water
            high_dyed = st.number_input("Dyed (Kg)", min_value=0.0, value=low_dyed, step=1000.0, key='high_dyed')
        with col2b:
            high_reknit = st.number_input("Reknit (Kg)", min_value=0.0, value=low_reknit, step=100.0, key='high_reknit')
            
        # second row of inputs
        col2c, col2d = st.columns(2)
        with col2c:
            high_additional = st.number_input("Additional (Kg)", min_value=0.0, value=low_additional, step=1000.0, key='high_additional')
        with col2d:
            high_hotwater = st.number_input("Hot Water (m³)", min_value=0.0, value=st.session_state.get('saved_high_hotwater', 4000.0), step=100.0, key='high_hotwater')
            
    #Input for Actual Water Consumption ---
    st.markdown("<br>", unsafe_allow_html=True)
    col_ac1, col_ac2, col_ac3 = st.columns([1,2,1])
    with col_ac2:
        # Use a temporary local variable to capture the input
        actual_water_input_temp = st.number_input(
            "Actual Process Water (m³) - Optional",
            min_value=0.0,
            value=st.session_state.get('actual_water_input', 0.0), # Use stored value as default
            step=1.0,
            key='actual_water_input_key',
            help="Leave as 0 if not available. This value is used for savings comparison when you click 'Predict Scenarios'."
        )

    st.markdown("<br>", unsafe_allow_html=True)

    #TWO BUTTONS ONLY
    col_btn1, col_btn2, col_btn3 = st.columns([1,1,1])

    with col_btn2:
        predict_btn = st.button(" ✦︎ Predict Scenarios", type="primary", use_container_width=True)
    with col_btn3:
        reset_btn = st.button("🔄 Reset", type="secondary", use_container_width=True)
    
    if reset_btn:
        # Clear all saving-related session state variables
        for key in ['charts_ready', 'saved_low_prediction', 'saved_high_prediction', 
                    'saved_low_hotwater', 'saved_high_hotwater', 'saved_low_dyed', 
                    'saved_low_reknit', 'saved_low_additional', 'actual_water_input']:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()
    
    if predict_btn:
        # Make predictions for both scenarios
        low_input = np.array([[low_dyed, low_reknit, low_additional, low_hotwater]])
        high_input = np.array([[high_dyed, high_reknit, high_additional, high_hotwater]])
    
        low_prediction = model.predict(low_input)[0]
        high_prediction = model.predict(high_input)[0]

        # Store ALL necessary data in session state, including the actual water input
        st.session_state.saved_low_prediction = low_prediction
        st.session_state.saved_high_prediction = high_prediction
        st.session_state.saved_low_hotwater = low_hotwater
        st.session_state.saved_high_hotwater = high_hotwater
        st.session_state.saved_low_dyed = low_dyed
        st.session_state.saved_low_reknit = low_reknit
        st.session_state.saved_low_additional = low_additional
        
        # Store the current actual water input
        st.session_state.actual_water_input = actual_water_input_temp

        st.session_state.charts_ready = True
        
        #Prediction Results and High vs Low Savings
        
        # Calculate savings (High vs Low)
        water_saving =  low_prediction - high_prediction
        waste_water_saving = water_saving * 0.9
        water_saving_usd = (water_saving * 165) / 290
        waste_water_saving_usd = waste_water_saving * 1.15
        
        st.markdown("---")
        st.markdown("<h3 style='text-align:left;font-size:24px;'>🏭 Predicted Process Water Consumption :</h3>", unsafe_allow_html=True)
        
        # Display Prediction Results
        col_res1, col_res2, col_res3, col_res4 = st.columns(4)
        
        with col_res2:
            st.metric(
                "🟢 Low Hot Water Usage Scenario",
                f"{low_prediction:,.0f} m³",
            )
        
        with col_res3:
            st.metric(
                "🔴 High Hot Water Usage Scenario ", 
                f"{high_prediction:,.0f} m³",
            )

        st.markdown("---")
        # Display Savings (High vs Low)
        st.markdown("<h4 style = 'text-align:left;'>💵 Predicted Savings :</h4>",unsafe_allow_html=True)

        col_s1,col_s2,col_s3,col_s4 = st.columns(4)

        with col_s1:
            st.metric("🔹💧 Water Saving",f"{water_saving:,.0f} m³")
        with col_s2:
            st.metric("🔹🗑️ Waste Water Saving",f"{waste_water_saving:,.0f} m³")
        with col_s3:
            st.metric("🔸💰 Saving USD (Water)", f"${water_saving_usd:,.2f}")
        with col_s4:
            st.metric("🔸💰 Saving USD (Waste Water)",f"${waste_water_saving_usd:,.2f}")
        
        
        # Call the helper function - THIS IS THE SAVINGS METRICS YOU WANTED UPWARD
        display_actual_vs_low_savings(
            low_prediction, # Use the freshly calculated low_prediction
            actual_water_input_temp # Use the freshly entered actual water input
        )
        # --- END MOVED SECTION 2 UP ---
        
        st.markdown("<h3 style = 'text-align:center;'>📊 Visualization</h3>",unsafe_allow_html=True)


    # Create two charts side by side - ALWAYS show charts when prediction is made
    if st.session_state.get('charts_ready', False):

        # USE SESSION STATE VARIABLES (These are already calculated and stored above)
        low_pred = st.session_state.saved_low_prediction
        high_pred = st.session_state.saved_high_prediction
        low_hw = st.session_state.saved_low_hotwater
        high_hw = st.session_state.saved_high_hotwater
        low_d = st.session_state.saved_low_dyed
        low_r = st.session_state.saved_low_reknit
        low_a = st.session_state.saved_low_additional
        current_actual_water = st.session_state.actual_water_input # Retrieve current actual water input

        # Create two charts side by side
        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            # Bar chart for comparing scenarios
            fig1, ax1 = plt.subplots(figsize=(6,4))

            # Prepare data for bar chart - USE SESSION STATE VARIABLES
            categories = ['Predicted - Low \nHot Water Scenario','Predicted - High \nHot Water Scenario']
            values = [low_pred, high_pred]
            colors = ['#1f77b4', '#ff7f0e']

            # Add actual water if provided
            if current_actual_water > 0:
                categories.append('Actual - High Hot\n Water Scenario')
                values.append(current_actual_water)
                colors.append('#2ca02c')
            
            bars = ax1.bar(categories, values, color=colors, width=0.6, edgecolor='black', linewidth=1)
    
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + (height * 0.01),
                        f'{height:,.0f} m³',
                        ha='center', va='bottom', fontsize=9, fontweight='bold')

            ax1.set_ylabel('Process Water (m³)', fontsize=10)
            ax1.set_title('Scenario Comparison', fontsize=12, fontweight='bold')
            ax1.grid(True, alpha=0.3, axis='y')
            ax1.tick_params(axis='x', labelsize=9)
            ax1.tick_params(axis='y', labelsize=8)

            st.pyplot(fig1)
            plt.close(fig1)

        with chart_col2:
            # Line chart showing prediction vs hot water usage
            fig2, ax2 = plt.subplots(figsize=(6,4))

            # --- 1. Generate and Plot Prediction Line ---
            min_hw_val = min(low_hw, high_hw) 
            max_hw_val = max(low_hw, high_hw) 
            
            range_start = max(0, min_hw_val - 500)
            range_end = max_hw_val + 500
            
            if range_start >= range_end:
                 range_end = range_start + 100 
            
            hotwater_range = np.linspace(range_start, range_end, 50) 
            # Note: We must use the *Low Scenario* garment inputs for the line calculation, as only Hotwater changes
            predictions = [model.predict([[low_d, low_r, low_a, hw]])[0] for hw in hotwater_range]

            ax2.plot(hotwater_range, predictions, 'b-', linewidth=2, label='Predicted Process Water')

            # --- 2. Mark Scenarios ---
            ax2.scatter([low_hw, high_hw], [low_pred, high_pred], 
                color=['blue', 'orange'], s=100, zorder=5, edgecolors='black')

            # --- 3. Mark Actual Process Water as a Point ---
            if current_actual_water > 0:
                # X-coordinate = High Hot Water Input (as per request)
                actual_hot_water_x = high_hw 
                # Y-coordinate = Actual Process Water Input
                actual_process_water_y = current_actual_water
                
                ax2.scatter([actual_hot_water_x], [actual_process_water_y], 
                            color='red', 
                            marker='X', # Distinct marker
                            s=150, 
                            zorder=6, 
                            edgecolors='black', 
                            label='Actual Point')
                
                ax2.annotate('Actual', (actual_hot_water_x, actual_process_water_y), 
                            textcoords="offset points", xytext=(-5,-20), ha='center', fontsize=9, color='red', fontweight='bold')
            
            # --- 4. Chart Cosmetics ---
            ax2.set_xlabel('Hot Water Usage (m³)', fontsize=10)
            ax2.set_ylabel('Process Water (m³)', fontsize=10)
            ax2.set_title('Process Water vs Hot Water Usage', fontsize=12, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            ax2.legend(fontsize=8)
            ax2.tick_params(axis='both', labelsize=8)

            st.pyplot(fig2)
            plt.close(fig2)

        # --- REMOVED THE REDUNDANT CALL HERE ---
        # The call to display_actual_vs_low_savings has been moved upward.