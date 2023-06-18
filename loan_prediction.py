import pickle
from pathlib import Path
import pandas as pd
import numpy as np
import pandas as pd  # pip install pandas openpyxl
import sklearn
import streamlit as st  # pip install streamlit
import streamlit_authenticator as stauth  # pip install streamlit-authenticator
from streamlit_echarts import st_pyecharts
from pyecharts.charts import Gauge
import pyecharts.options as opts

from PIL import Image
print(pd.__version__,np.__version__)
# emojis: https://www.webfx.com/tools/emoji-cheat-sheet/

st.set_page_config(page_title="Loan Default Prediction Dashboard", page_icon=":bank:", layout="wide")
st.markdown("""
<style>
   
</style>
""",unsafe_allow_html=True)
st.markdown(f"""<style>
.css-16idsys p{{
font-size:20px;}}
.css-uf99v8 {{
margin-top: -100px;}}
.stTabs css-0 e15kodlz0{{margin-top: -50px;}}
.stImage{{
border: black;
    border-style: solid;
    border-width: 2px;
}}

 </style>""",unsafe_allow_html=True)
 
with open("style.css") as f:
    st.markdown(f'<style> {f.read()}</style>', unsafe_allow_html=True)
st.title('Loan Default Prediction')

hide_table_row_index = """
            <style>
            thead tr th:first-child {display:none}
            tbody th {display:none}
            </style>
            """

# Inject CSS with Markdown
st.markdown(hide_table_row_index, unsafe_allow_html=True)

# --- USER AUTHENTICATION ---
names = ["test_user"]
usernames = ["user@123"]
file_path = Path(__file__).parent / "hashed_pw.pkl"
with file_path.open("rb") as file:
    hashed_passwords = pickle.load(file)

print(hashed_passwords)
credentials = {
        "usernames":{
            usernames[0]:{
                "name":names[0],
                "password":hashed_passwords[0]
                }
        
            }
        }
# load hashed passwords
#file_path = Path(__file__).parent / "hashed_pw.pkl"


authenticator = stauth.Authenticate(credentials,"Loan Default Prediction", "abcdef", cookie_expiry_days=30)

name, authentication_status, username = authenticator.login("Login", "main")
if 'key' not in st.session_state:
    st.session_state['key'] = 'value'
print(authentication_status ,"dd")
if authentication_status == False:
    st.error("Username/password is incorrect")

if authentication_status == None:
    st.warning("Please enter your username and password")

if authentication_status:
    print("done")
    tab1, tab2 ,tab3= st.tabs(["📈 Data Summary ","📈 Data Insights ", "⚙️ Prediction"])
    data = np.random.randn(10, 1)

    
    with tab1:
        st.write("Variable Descriptive Summary")
        data_numerical ={
    'Variable': ['VerificationType', 'Age', 'Gender', 'AppliedAmount', 'Amount', 'Interest', 'LoanDuration',
                 'MonthlyPayment', 'UseOfLoan', 'Education', 'MaritalStatus', 'EmploymentStatus', 'OccupationArea',
                 'HomeOwnershipType', 'IncomeTotal', 'ExistingLiabilities', 'LiabilitiesTotal', 'RefinanceLiabilities',
                 'DebtToIncome', 'FreeCash', 'PlannedInterestTillDate', 'PrincipalOverdueBySchedule', 'RecoveryStage',
                 'CreditScoreEeMini', 'PrincipalPaymentsMade', 'InterestAndPenaltyPaymentsMade', 'PrincipalBalance',
                 'InterestAndPenaltyBalance', 'NoOfPreviousLoansBeforeLoan', 'AmountOfPreviousLoansBeforeLoan',
                 'PreviousRepaymentsBeforeLoan', 'PreviousEarlyRepaymentsCountBeforeLoan', 'Country',
                 'EmploymentDurationCurrentEmployer', 'Rating', 'NewCreditCustomer', 'Restructured'],
    'Average': ['0', '40', '-', '2,699', '2,544', '34', '1', '113', '-1', '-1', '-1', '-1', '-1', '-1', '1763.924509', '0', '490.75779',
'0', '5.829296865', '91.75591447', '983.6418726', '330.9863395', '1', '947.2257905', '972.2413931', '599.1607942',
'1544.728145', '939.7743378', '0', '3130.293908', '979.2854385', '0', '', '', '', '', ''],
    'Min': ['0', '18', '-', '32', '6', '3', '1', '-', '8', '5', '5', '6', '19', '10', '0', '40', '0', '23', '0', '-211', '0', '0', '2', '500', '0', '0', '0', '-2.66', '0', '0','0', '0', '', '', '', '', ''],
    'Max': ['4', '77', '2', '10,632', '10,632', '264', '60', '2,369', '-', '-', '-', '-', '-', '-', '1012019', '40', '12400000', '23',
'83.33', '158748.64', '14948.91', '10630', '2', '1000', '10632', '18393.46', '10632', '78982.07', '27', '72778', '34077.42', '11', '', '', '', '', '']}
        print(len(data_numerical["Variable"]),len(data_numerical["Average"]),len(data_numerical["Min"]),len(data_numerical["Max"]))
        
        df_numerical = pd.DataFrame(data_numerical)
        st.table(df_numerical)
        st.subheader("Feature Description")
        
        col111,col112,col123,col124= st.columns(4)
        with col111:
            st.write("Education")
            data_education={
    'Variable': ['-1', '0', '1', '2', '3', '4', '5'],
    'Value': ['Not present', 'Not present', 'Primary', 'Basic', 'Vocational', 'Secondary', 'Higher']
}       
            df_education = pd.DataFrame(data_education)
            st.table(df_education)
        
        with col112:
            st.write("Employment")
            data_employment={
    'Variable': ['-1', '0', '1', '2', '3', '4', '5', '6'],
    'Value': ['Unknown', 'Unknown', 'Unemployed', 'Partially unemployed', 'Fully unemployed', 'Self-employed',
              'Entrepreneur', 'Retiree']
}  
            df_employment = pd.DataFrame(data_employment)
            st.table(df_employment)
        with col123:
            st.write("Gender")
            data_gender={
    'Variable': [ '0', '1', '2'],
    'Value': ['Male','Female','Unknown', ]
}  
            df_gender = pd.DataFrame(data_gender)
            st.table(df_gender)
        
        with col124:
            st.write("Recovery Stage")
            data_stage={
    'Variable': [ '0', '1'],
    'Value': ['Recovered','Collection' ]
}  
            df_stage = pd.DataFrame(data_stage)
            st.table(df_stage)
        
        col211,col212,col213,col214= st.columns(4)
        with col211:
            st.write("Verification Type")
            data_verification={
    'Variable': ['0', '1', '2', '3', '4'],
    'Value': ['Not set', 'Unverified', 'Unverified on phone', 'Verified', 'Verified on phone']
}  
            df_verification = pd.DataFrame(data_verification)
            st.table(df_verification)
            
        with col212:
            st.write("Marital Status")
            data_Marital={
    'Variable': ['-1', '0', '1', '2', '3', '4', '5'],
    'Value': ['Not Specified', 'Not Specified', 'Married', 'Cohabitant', 'Single', 'Divorce', 'Widow']
}  
            df_Marital = pd.DataFrame(data_Marital)
            st.table(df_Marital)
            
        with col213:
            st.write("Home Ownership Type")
            data_Home={
    
    'Variable': ['-1', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10'],
    'Value': ['Unknown', 'Homeless', 'Owner', 'Living with parents', 'Unfurnished rent', 'Furnished rent',
              'Council house', 'Joint tenant', 'Joint ownership', 'Mortgage', 'Owner with encumbrance', 'Other']

}  
            df_Home = pd.DataFrame(data_Home)
            st.table(df_Home)
            
        with col214:
            st.write("Occupation Area")
            data_Occupation={
    'Variable': ['-1', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19'],
    'Value': ['Not specified', 'Not specified', 'Other', 'Mining', 'Processing', 'Energy', 'Utilities', 'Construction',
              'Retail', 'Transport', 'Hospitality', 'Telecom', 'Finance', 'Real estate', 'Research', 'Administrative',
              'Civil services', 'Education', 'Healthcare', 'Art', 'Agriculture']
}  
            df_Occupation = pd.DataFrame(data_Occupation)
            st.table(df_Occupation)
        
        col311,col312= st.columns(2)
        with col311:
        
            st.write("Country")
            st.text("EE,ES,FI,SK")
        with col312:
        
            st.write("EmploymentDurationCurrentEmployer")
            st.text("MoreThan5Years,UpTo3Years,UpTo5Years,UpTo1Year,TrialPeriod,Retiree Other")
        
        col313,col314,col315=st.columns(3) 
        with col313:
            st.write("Rating")
            st.text("C,B,A,F,D,AA")    
        with col314:
            st.write("NewCreditCustomer")
            st.text("EE,ES,FI,SK")
        with col315:
            st.write("Restructured")
            st.text("TRUE,FALSE")
        

        
        
        
        
    with tab2:
        col201,col202,col203= st.columns(3)
        with col201:
                image_21 = Image.open('image003.png')

                st.image(image_21, width =400)
        with col202:
                image_22 = Image.open('image005.png')

                st.image(image_22, width =400)
        with col203:
                image_23 = Image.open('image007.png')

                st.image(image_23, width =400)
        col301,col302,col303= st.columns(3)
        with col301:
                image_31 = Image.open('image009.png')

                st.image(image_31, width =400)
        with col302:
                image_32 = Image.open('image011.png')

                st.image(image_32, width =400)
        with col303:
                image_33 = Image.open('image013.png')

                st.image(image_33, width =400)
            
    with tab3:
        
        
        uploaded_file = st.file_uploader("Upload file for prediction")
        df_sample = pd.read_excel(r"Input (1).xlsx")
        with st.expander("See Sample Input File"):
            st.table(df_sample)
        if st.button('Submit'):
            if uploaded_file is None:
               st.warning("Please upload file")
            else:
               
               data = pd.read_excel(uploaded_file)
               bool_data= data.select_dtypes('bool')
               data = data.drop(bool_data.columns, axis=1)
               features_bool_data= list(bool_data.columns)
   
   
               filename = "bool_encoder.pkl"
               with open(filename, 'rb') as file:
                   label_encoder = pickle.load(file)
                   
   
               bool_data['NewCreditCustomer'] = label_encoder.transform(bool_data['NewCreditCustomer'])
               bool_data['Restructured'] = label_encoder.transform(bool_data['Restructured'])
               cat_data= data.select_dtypes('object')
               data = data.drop(cat_data.columns, axis=1)
               features_cat_data=list(cat_data.columns)
   
               filename = "cat1_encoder.pkl"
               with open(filename, 'rb') as file:
                   label_encoder_ = pickle.load(file)
   
               transformed_data= label_encoder_.transform(cat_data)
               tranformed_data=pd.DataFrame(transformed_data)
               tranformed_data.columns=features_cat_data
               object_data = pd.concat([tranformed_data,bool_data], axis=1)
               object_data.shape
   
               final_data=pd.concat([data,object_data], axis=1)
               
               filename=r"normalize_function.pkl"
               with open(filename, 'rb') as file:
                   normalization_function = pickle.load(file)
   
               data = pd.DataFrame(normalization_function.transform(final_data))
   
               data.columns=final_data.columns
               data=data.fillna(0)
               
               filename=r"model_final.pkl"
               with open(filename, 'rb') as file:
                   prediction = pickle.load(file)
               probability=prediction.predict(data)
               print(round(probability[0],2),"hjhjh")
               ans=round(probability[0],2)
               print(type(int(ans)),ans)
               c1=(Gauge()
                                   .add("", [("Probability", int(ans*100))], radius="80%")
                                   .set_global_opts(title_opts=opts.TitleOpts(title="Probability"))
                                   .set_series_opts(
                                       axisline_opts=opts.AxisLineOpts(
                                       linestyle_opts=opts.LineStyleOpts(
                                       color=[[0.2, "#B9ABAA"],[0.8, "#F13626"], [1, "#F81D0B"]], width=20
                                       ))))
               st_pyecharts(c1)


    
    
        if st.button('Use Sample File for prediction '):
            
            data = df_sample
            bool_data= data.select_dtypes('bool')
            data = data.drop(bool_data.columns, axis=1)
            features_bool_data= list(bool_data.columns)


            filename = "bool_encoder.pkl"
            with open(filename, 'rb') as file:
                label_encoder = pickle.load(file)
                

            bool_data['NewCreditCustomer'] = label_encoder.transform(bool_data['NewCreditCustomer'])
            bool_data['Restructured'] = label_encoder.transform(bool_data['Restructured'])
            cat_data= data.select_dtypes('object')
            data = data.drop(cat_data.columns, axis=1)
            features_cat_data=list(cat_data.columns)

            filename = "cat1_encoder.pkl"
            with open(filename, 'rb') as file:
                label_encoder_ = pickle.load(file)

            transformed_data= label_encoder_.transform(cat_data)
            tranformed_data=pd.DataFrame(transformed_data)
            tranformed_data.columns=features_cat_data
            object_data = pd.concat([tranformed_data,bool_data], axis=1)
            object_data.shape

            final_data=pd.concat([data,object_data], axis=1)
            
            filename=r"normalize_function.pkl"
            with open(filename, 'rb') as file:
                normalization_function = pickle.load(file)

            data = pd.DataFrame(normalization_function.transform(final_data))

            data.columns=final_data.columns
            data=data.fillna(0)
            
            filename=r"model_final.pkl"
            with open(filename, 'rb') as file:
                prediction = pickle.load(file)
            probability=prediction.predict(data)
            
            ans=round(probability[0],2)
            print(type(int(ans)),ans)
            c1=(Gauge()
                                .add("", [("Probability", int(ans*100))], radius="80%")
                                .set_global_opts(title_opts=opts.TitleOpts(title="Probability"))
                                .set_series_opts(
                                    axisline_opts=opts.AxisLineOpts(font_size=40, color="blue",
                                    linestyle_opts=opts.LineStyleOpts(
                                    color=[[0.2, "#B9ABAA"],[0.8, "#F13626"], [1, "#F81D0B"]], width=20
                                    ))))
            st_pyecharts(c1)
    
            
            
