# ===============================
# app.py (FULL – Bilingual TH/EN)
# ===============================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
import os

# ======================================================
# Thai tokenizer (MUST exist – same name as training)
# ======================================================
def thai_tokenizer(text):
    if text is None:
        return []
    text = str(text).lower()
    return re.findall(r"[A-Za-z]+|[ก-๙]+|\d+", text)

# ======================================================
# PAGE CONFIG
# ======================================================
st.set_page_config(
    page_title="Job Description Classifier",
    layout="wide"
)

# ======================================================
# LANGUAGE SWITCH
# ======================================================
LANG_KEY = st.sidebar.selectbox("🌐 Language / ภาษา", ["TH", "EN"])

# ======================================================
# TEXT TRANSLATION
# ======================================================
TEXT = {
    "TH": {
        "title": "ระบบทำนายสายงานจาก Job Description",
        "subtitle": "ใช้ข้อมูลเชิงโครงสร้าง + ข้อความ (TF-IDF ภาษาไทย)",
        "jd": "รายละเอียดงาน (Job Description)",
        "predict": "🚀 ทำนายผล",
        "compare": "🔁 เปรียบเทียบทุกโมเดล",
        "best": "โมเดลที่ดีที่สุด",
        "confidence": "ความเชื่อมั่น",
        "menu": "เมนู",
        "result": "ผลลัพธ์การทำนาย",
        "region": "ภูมิภาคที่ทำงาน",
        "table": "ตารางเปรียบเทียบโมเดล",
    },
    "EN": {
        "title": "Job Description Classifier",
        "subtitle": "Structured features + text (TF-IDF)",
        "jd": "Job Description",
        "predict": "🚀 Predict",
        "compare": "🔁 Compare all models",
        "best": "Best Model",
        "confidence": "Confidence",
        "menu": "Menu",
        "result": "Prediction Result",
        "region": "Work Region",
        "table": "Model Comparison Table",
    }
}
T = TEXT[LANG_KEY]

# ======================================================
# LABELS (DISPLAY ONLY – DO NOT CHANGE ORDER)
# ======================================================
SENIORITY_LABELS = {
    "TH": ["ฝึกงาน", "ระดับต้น", "ระดับกลาง", "อาวุโส", "หัวหน้าทีม"],
    "EN": ["Intern", "Junior", "Mid", "Senior", "Lead"],
}

CONTRACT_LABELS = {
    "TH": ["ประจำ", "สัญญาจ้าง", "ฝึกงาน"],
    "EN": ["Full-time", "Contract", "Internship"],
}

EDU_LABELS = {
    "TH": ["ไม่จำกัด", "ปริญญาตรี", "ปริญญาโท", "ปริญญาเอก"],
    "EN": ["Any", "Bachelor", "Master", "PhD"],
}

LANG_REQ_LABELS = {
    "TH": ["ภาษาไทย", "ภาษาอังกฤษ", "สองภาษา"],
    "EN": ["Local", "English", "Bilingual"],
}

REGION_LABELS = {
    "TH": [
        "กรุงเทพ / เมืองใหญ่",
        "ภาคกลาง",
        "ภาคเหนือ",
        "ภาคอีสาน",
        "ภาคใต้",
        "ต่างประเทศ / รีโมต",
    ],
    "EN": [
        "Metro / Capital",
        "Central",
        "North",
        "Northeast",
        "South",
        "International / Remote",
    ],
}

JOB_LABELS = {
    "TH": {
        0: "ซอฟต์แวร์",
        1: "สายข้อมูล",
        2: "ดีไซน์",
        3: "การขาย",
        4: "การตลาด",
        5: "ปฏิบัติการ",
    },
    "EN": {
        0: "Software",
        1: "Data",
        2: "Design",
        3: "Sales",
        4: "Marketing",
        5: "Operations",
    },
}

# ======================================================
# LOAD MODELS
# ======================================================
@st.cache_resource
def load_models():
    models = {}
    if os.path.exists("Logistic.joblib"):
        models["Logistic Regression"] = joblib.load("Logistic.joblib")
    if os.path.exists("SVM.joblib"):
        models["SVM"] = joblib.load("SVM.joblib")
    if os.path.exists("Random_Forest.joblib"):
        models["Random Forest"] = joblib.load("Random_Forest.joblib")
    return models

models = load_models()
if not models:
    st.error("❌ Model files not found")
    st.stop()

# ======================================================
# SIDEBAR
# ======================================================
st.sidebar.header(f"⚙️ {T['menu']}")
compare_mode = st.sidebar.checkbox(T["compare"], value=True)
selected_model = st.sidebar.selectbox("Model", list(models.keys()))

# ======================================================
# HEADER
# ======================================================
st.markdown(f"<h1 style='text-align:center'>{T['title']}</h1>", unsafe_allow_html=True)
st.markdown(f"<p style='text-align:center;color:gray'>{T['subtitle']}</p>", unsafe_allow_html=True)
st.markdown("---")

# ======================================================
# JD TEXT
# ======================================================
jd_text = st.text_area(T["jd"], height=140)

# ======================================================
# INPUT FORM
# ======================================================
with st.form("input_form"):
    c1, c2, c3 = st.columns(3)

    with c1:
        tech = st.slider("Tech Skill", 0, 100, 60)
        data = st.slider("Data Skill", 0, 100, 40)
        design = st.slider("Design Skill", 0, 100, 20)
        sales = st.slider("Sales Skill", 0, 100, 10)
        marketing = st.slider("Marketing Skill", 0, 100, 10)
        ops = st.slider("Ops Skill", 0, 100, 10)

    with c2:
        seniority_label = st.selectbox(
            "ระดับตำแหน่ง" if LANG_KEY == "TH" else "Seniority",
            SENIORITY_LABELS[LANG_KEY],
        )
        contract_label = st.selectbox(
            "ประเภทสัญญา" if LANG_KEY == "TH" else "Contract Type",
            CONTRACT_LABELS[LANG_KEY],
        )
        edu_label = st.selectbox(
            "วุฒิการศึกษา" if LANG_KEY == "TH" else "Education",
            EDU_LABELS[LANG_KEY],
        )
        lang_req_label = st.selectbox(
            "ภาษาที่ต้องการ" if LANG_KEY == "TH" else "Language Requirement",
            LANG_REQ_LABELS[LANG_KEY],
        )
        min_exp = st.number_input(
            "ประสบการณ์ขั้นต่ำ (ปี)" if LANG_KEY == "TH" else "Min Years Experience",
            0,
            20,
            2,
        )
        remote_flag = st.selectbox(
            "ทำงานระยะไกล" if LANG_KEY == "TH" else "Remote",
            ["ไม่ใช่", "ใช่"] if LANG_KEY == "TH" else ["No", "Yes"],
        )

    with c3:
        region_label = st.selectbox(T["region"], REGION_LABELS[LANG_KEY])
        resp = st.number_input("จำนวน Responsibilities", 1, 30, 6)
        req = st.number_input("จำนวน Requirements", 1, 30, 8)
        tools = st.number_input("จำนวน Tools", 0, 20, 5)
        salary_min = st.number_input("เงินเดือนต่ำสุด", 0, 200000, 30000)
        salary_max = st.number_input("เงินเดือนสูงสุด", 0, 300000, 50000)

    submit = st.form_submit_button(T["predict"], use_container_width=True)

# ======================================================
# PREDICTION
# ======================================================
if submit:
    input_df = pd.DataFrame([
        {
            "jd_text": jd_text,
            "seniority": SENIORITY_LABELS[LANG_KEY].index(seniority_label),
            "contract_type": CONTRACT_LABELS[LANG_KEY].index(contract_label),
            "region_code": REGION_LABELS[LANG_KEY].index(region_label),
            "remote_flag": 1 if remote_flag in ["Yes", "ใช่"] else 0,
            "min_years_exp": min_exp,
            "edu_min": EDU_LABELS[LANG_KEY].index(edu_label),
            "responsibilities_count": resp,
            "requirements_count": req,
            "tools_mentioned": tools,
            "lang_req": LANG_REQ_LABELS[LANG_KEY].index(lang_req_label),
            "tech_skill": tech,
            "data_skill": data,
            "design_skill": design,
            "sales_skill": sales,
            "marketing_skill": marketing,
            "ops_skill": ops,
            "salary_min": salary_min,
            "salary_max": salary_max,
            "salary_per_year_exp": ((salary_min + salary_max) / 2) / (min_exp + 1),
            "req_to_resp_ratio": req / resp,
            "skill_density": (tech + data + design + sales + marketing + ops)
            / (req + resp),
            "complexity": SENIORITY_LABELS[LANG_KEY].index(seniority_label) * 10
            + tools * 2
            + req
            + resp,
        }
    ])

    st.markdown("---")
    st.subheader(f"📊 {T['result']}")

    results = []
    run_models = models if compare_mode else {selected_model: models[selected_model]}

    for name, model in run_models.items():
        pred = model.predict(input_df)[0]
        probs = model.predict_proba(input_df)[0]
        conf = float(np.max(probs))

        results.append(
            {
                "Model": name,
                "Prediction": JOB_LABELS[LANG_KEY][pred],
                "Confidence": conf,
            }
        )

        with st.expander(name):
            st.success(JOB_LABELS[LANG_KEY][pred])
            st.write(f"{T['confidence']}: {conf:.4f}")
            prob_df = pd.DataFrame(
                {
                    "Job": list(JOB_LABELS[LANG_KEY].values()),
                    "Probability": probs,
                }
            )
            st.bar_chart(prob_df.set_index("Job"))

    res_df = pd.DataFrame(results).sort_values("Confidence", ascending=False)
    st.subheader(T["table"])
    st.dataframe(res_df, use_container_width=True)

    best = res_df.iloc[0]
    st.success(
        f"{T['best']}: {best['Model']} ({T['confidence']} = {best['Confidence']:.4f})"
    )