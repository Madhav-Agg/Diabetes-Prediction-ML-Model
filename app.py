import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.graph_objects as go


# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Diabetes AI Healthcare Dashboard",
    page_icon="🩺",
    layout="wide"
)


# =====================================================
# CUSTOM CSS
# =====================================================

st.markdown("""
<style>

body {
    background-color:#f5f9ff;
}


.main {
    background-color:#f5f9ff;
}


/* Main title */

h1 {
    font-size:44px !important;
    color:#083b66;
}


h2 {
    color:#0b5cab;
}


h3 {
    color:#083b66;
}


/* Cards */

.card {

    background:white;
    padding:25px;
    border-radius:20px;
    box-shadow:0px 8px 25px rgba(0,0,0,0.08);
    margin-bottom:20px;

}


/* Metrics */

.metric-card{

background:linear-gradient(
135deg,#083b66,#00a6c8);

padding:25px;
border-radius:20px;
color:white;
text-align:center;

}


.metric-title{

font-size:18px;

}


.metric-value{

font-size:35px;
font-weight:bold;

}



/* Button */

.stButton button {

width:100%;
height:60px;

background:
linear-gradient(
90deg,
#0077b6,
#00b4d8
);

color:white;

font-size:22px;

font-weight:bold;

border-radius:15px;

border:none;

}


.stButton button:hover{

transform:scale(1.03);

}



/* Sidebar */

[data-testid="stSidebar"]{

background:#062b4f;

}


[data-testid="stSidebar"] *{

color:white;

}



/* Input labels */

label{

font-size:18px !important;

font-weight:600;

}



</style>

""", unsafe_allow_html=True)



# =====================================================
# LOAD MODEL
# =====================================================

model = joblib.load(
    "model/diabetes_model.pkl"
)

scaler = joblib.load(
    "model/scaler.pkl"
)



# =====================================================
# SIDEBAR
# =====================================================

with st.sidebar:


    st.title("🩺 Diabetes AI")


    st.markdown(
    """
    ### Models Used

    ✔ Logistic Regression

    ✔ Decision Tree

    ✔ Random Forest ⭐

    ✔ XGBoost


    ### Dataset

    100,000 Patients


    ### Features

    8 Medical Parameters


    ### Best Accuracy

    96.96%

    """
    )



    st.info(
        "AI powered healthcare prediction system"
    )



# =====================================================
# HEADER
# =====================================================


st.markdown(
"""
<h1>
🩺 Diabetes Prediction Dashboard
</h1>

<h3>
AI Powered Healthcare System
</h3>

""",
unsafe_allow_html=True
)



st.divider()



# =====================================================
# INPUT SECTION
# =====================================================


left,right = st.columns(
2,
gap="large"
)



with left:


    st.markdown(
    '<div class="card">',
    unsafe_allow_html=True
    )


    st.subheader(
        "👤 Patient Information"
    )


    gender = st.selectbox(
        "Gender",
        [
            "Female",
            "Male",
            "Other"
        ]
    )


    age = st.number_input(
        "Age",
        1,
        100,
        35
    )


    bmi = st.number_input(
        "BMI",
        10.0,
        60.0,
        25.0
    )


    smoking = st.selectbox(
        "Smoking History",
        [
            "No Info",
            "never",
            "former",
            "current",
            "ever",
            "not current"
        ]
    )


    st.markdown(
    '</div>',
    unsafe_allow_html=True
    )




with right:


    st.markdown(
    '<div class="card">',
    unsafe_allow_html=True
    )


    st.subheader(
        "❤️ Medical Conditions"
    )


    hypertension = st.selectbox(
        "Hypertension",
        [
            "No",
            "Yes"
        ]
    )


    heart = st.selectbox(
        "Heart Disease",
        [
            "No",
            "Yes"
        ]
    )


    hba1c = st.slider(
        "HbA1c Level",
        3.0,
        10.0,
        5.5
    )


    glucose = st.slider(
        "Blood Glucose Level",
        70,
        300,
        120
    )


    st.markdown(
    '</div>',
    unsafe_allow_html=True
    )



# =====================================================
# ENCODING
# =====================================================


gender_map={

"Female":0,
"Male":1,
"Other":2

}


smoking_map={

"No Info":0,
"current":1,
"ever":2,
"former":3,
"never":4,
"not current":5

}



yes_no={

"No":0,
"Yes":1

}



# =====================================================
# PREDICT BUTTON
# =====================================================


st.write("")


predict = st.button(
    "🩺 Predict Diabetes Risk"
)



if predict:


    input_data = pd.DataFrame({

        "gender":[
            gender_map[gender]
        ],

        "age":[age],

        "hypertension":[
            yes_no[hypertension]
        ],

        "heart_disease":[
            yes_no[heart]
        ],

        "smoking_history":[
            smoking_map[smoking]
        ],

        "bmi":[bmi],

        "HbA1c_level":[
            hba1c
        ],

        "blood_glucose_level":[
            glucose
        ]

    })



    scaled = scaler.transform(
        input_data
    )



    prediction = model.predict(
        scaled
    )[0]


    probability = model.predict_proba(
        scaled
    )[0][1]



    risk = probability*100



    st.divider()



    # =================================================
    # RESULT DASHBOARD
    # =================================================


    col1,col2 = st.columns(
        [1,1]
    )



    with col1:


        st.markdown(
        '<div class="card">',
        unsafe_allow_html=True
        )


        st.subheader(
            "Prediction Result"
        )


        if prediction==1:

            st.error(
                "⚠ HIGH RISK"
            )

        else:

            st.success(
                "✅ LOW RISK"
            )


        st.metric(
            "Probability",
            f"{risk:.2f}%"
        )


        if risk<30:

            st.success(
                "Excellent Condition"
            )

        elif risk<70:

            st.warning(
                "Moderate Risk"
            )

        else:

            st.error(
                "Needs Medical Attention"
            )


        st.markdown(
        '</div>',
        unsafe_allow_html=True
        )





    with col2:


        st.markdown(
        '<div class="card">',
        unsafe_allow_html=True
        )


        st.subheader(
            "Risk Gauge"
        )


        fig = go.Figure(
            go.Indicator(

                mode="gauge+number",

                value=risk,

                gauge={

                    "axis":{
                        "range":[0,100]
                    }

                }

            )
        )


        fig.update_layout(
            height=300
        )


        st.plotly_chart(
            fig,
            use_container_width=True
        )


        st.markdown(
        '</div>',
        unsafe_allow_html=True
        )



    # =================================================
    # PATIENT SUMMARY
    # =================================================


    st.subheader(
        "📋 Patient Summary"
    )


    summary = pd.DataFrame({

        "Parameter":[
            "Gender",
            "Age",
            "BMI",
            "HbA1c",
            "Glucose",
            "Smoking"
        ],

        "Value":[
            gender,
            age,
            bmi,
            hba1c,
            glucose,
            smoking
        ]

    })


    st.dataframe(
        summary,
        use_container_width=True,
        hide_index=True
    )



    # =================================================
    # MODEL COMPARISON
    # =================================================


    st.subheader(
        "🤖 Model Performance"
    )


    models={

        "Random Forest ⭐":96.96,

        "XGBoost":96.72,

        "Logistic Regression":95.94,

        "Decision Tree":94.80

    }


    for name,score in models.items():

        st.write(
            f"**{name}** {score}%"
        )

        st.progress(
            score/100
        )