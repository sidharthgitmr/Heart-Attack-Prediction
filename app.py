import streamlit as st
import pandas as pd
import joblib 

st.title("Heart Attack Prediction")
st.write("This App Helps To Predict Chance Of Heart Attack")
st.divider()

model=joblib.load("model.pkl")
scaler=joblib.load('scaler.pkl')
age=joblib.load('age_encoder.pkl')
sex=joblib.load('sex_encode.pkl')
genhealth=joblib.load('genhealth_encode.pkl')
phyacti=joblib.load('phyacti_encode.pkl')
smokestat=joblib.load('smokestat_encode.pkl')
alcodrink=joblib.load('alcodrink_encode.pkl')
sleephrs=joblib.load('sleephrs_encode.pkl')
hadstrk=joblib.load('hadstrk_encode.pkl')
hadcopd=joblib.load('hadcopd_encode.pkl')
hadangina=joblib.load('hadangina_encode.pkl')
haddia=joblib.load('haddia_encode.pkl')
bmi=joblib.load('bmi_encode.pkl')
dfwalk=joblib.load('dfwalk_encode.pkl')



sexb=st.selectbox("Gender",['Male','Female'])
ageb=st.number_input("Age",min_value=0,max_value=100,step=1)
genhealthb=st.selectbox("General Health",['Very good','Fair','Good','Excellent','Poor'])
st.divider()
st.write("Lifestyle Factor")
st.divider()
phyactib=st.selectbox("Physical Activities",['Yes','No'])
smokestatb=st.selectbox("Smoke Status",['Former smoker','Never smoked','Current smoker - now smokes every day','Current smoker - now smokes some days'])
alcodrinkb=st.selectbox("Alcohol Drinking status",['No','Yes'])
sleephrsb=st.number_input("Sleep Hours",min_value=1,max_value=22,step=1)
st.divider()
st.write("Pre-existing Conditions")
st.divider()
hadstrkb=st.selectbox("Had Stroke Before",['No','Yes'])
hadcopdb=st.selectbox("Had COPD Before",['No','Yes'])
hadanginab=st.selectbox("Had Angina Before",['No','Yes'])
haddiab=st.selectbox("Had Diabetes",['No','Yes','Yes, but only during pregnancy (female)','No, pre-diabetes or borderline diabetes'])
st.divider()
st.write("Health Indicator")
st.divider()
bmib=st.number_input("BMI",min_value=1,max_value=100,step=1)
dfwalkb=st.selectbox("Difficulty In Walking",['No','Yes'])
st.divider()

st.write(model)
sex_encode=sex.transform([sexb])
genhealth_encode=genhealth.transform([genhealthb])
phyacti_encoded=phyacti.transform([phyactib])
smokestat_encoded=smokestat.transform([smokestatb])
alcodrink_encoded=alcodrink.transform([alcodrinkb])
hadstrk_encoded=hadstrk.transform([hadstrkb])
hadcopd_encoded=hadcopd.transform([hadcopdb])
hadangina_encoded=hadangina.transform([hadanginab])
haddia_encoded=haddia.transform([haddiab])
dfwalk_encoded=dfwalk.transform([dfwalkb])

data=pd.DataFrame({'Sex':sex_encode,
                   'AgeCategory':ageb,
                   'GeneralHealth':genhealth_encode,
                   'PhysicalActivities':phyacti_encoded,
                   'SmokerStatus':smokestat_encoded,
                   'AlcoholDrinkers':alcodrink_encoded,
                   'SleepHours':sleephrsb,
                   'HadStroke':hadstrk_encoded,
                   'HadCOPD':hadcopd_encoded,
                   'HadAngina':hadangina_encoded,
                   'HadDiabetes':haddia_encoded,
                   'BMI':bmib,
                   "DifficultyWalking":dfwalk_encoded
                   },index=[0])



data_scaled=scaler.transform(data)

pred=model.predict(data_scaled)[0]

##One appraoch
# output = "No Chance of Heart Attack" if pred==0 else "High Chance of Heart Attack"

#Second approach
# output=st.success("No Chance of Heart Attack Stay Healthy!") if pred==0 else st.error("Potential Heart Attack risk detected Urgent Medical Attention recommended!")


if st.button("Predict"):
    if pred==0:
        st.success("No Chance of Heart Attack - Stay Healthy!")
    else:
        st.error("Potential Heart Attack risk detected Urgent Medical Attention recommended!")


