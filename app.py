import streamlit as st
import pickle
import pandas as pd
import sklearn

with open('model/elastic_math_model.pkl','rb') as file:
    elastic_math_model = pickle.load(file)

with open('model/elastic_read_model.pkl','rb') as file:
    elastic_read_model = pickle.load(file)

with open('model/elastic_write_model.pkl','rb') as file:
    elastic_write_model = pickle.load(file)

with open('model/one_hot_columns.pkl','rb') as file:
    one_hot_columns = pickle.load(file)

with open('model/scaler.pkl','rb') as file:
    scaler = pickle.load(file)

def main():
    design = """<div style='padding:15px;'>
                    <h1 style='color:#fff'>Exam Score Prediction</h1>
                </div>"""
    st.markdown(design, unsafe_allow_html=True)
    left, right = st.columns((2, 2))
    gen = left.selectbox('Gender', ('Female', 'Male'))
    Eth = right.selectbox('Ethnic Group', ('group A', 'group B', 'group C', 'group D', 'group E'))
    ParEdu = left.selectbox('Parent Education', ('High School','College', 'Associates Degree', 'Bachelors Degree', 'Masters Degree'))
    LunTyp = right.selectbox('Lunch Type', ('Free/Reduced', 'Standard'))
    TesPre = left.selectbox('Test Preparation', ('None', 'Completed'))
    ParMaritStat = right.selectbox('Parent Marital Status', ('Divorced', 'Married', 'Single', 'Widowed'))
    PracSpo = left.selectbox('Practice Sport', ('Never', 'Sometimes', 'Regularly'))
    Is1stkid = right.selectbox('First Child', ('Yes', 'No'))
    NrSib = left.number_input('Number of Sibling', step=1, value=0, format="%d")
    Trans = right.selectbox('Mode of Transportation', ('Private', 'School Bus'))
    Weekly = st.selectbox('Weekly Study Hours', ('Less than 5 hours', 'Between 5-10 hours', 'More than 10 hours'))
    button = st.button('Predict')

    #if button is clicked (ketika button dipencet)
    if button:
        #make prediction
        math_scores, read_scores, write_scores = predict(gen, Eth, ParEdu, LunTyp, TesPre, ParMaritStat, PracSpo, Is1stkid, NrSib, Trans, Weekly)
        st.success("Math Scores: {:.2f}".format(math_scores[0]))
        st.success("Reading Scores: {:.2f}".format(read_scores[0]))
        st.success("Write Scores: {:.2f}".format(write_scores[0]))

def predict(gen, Eth, ParEdu, LunTyp, TesPre, ParMaritStat, PracSpo, Is1stkid, NrSib, Trans, Weekly):
    #processing user input
    data_baru = {'Gender': gen,
                'EthnicGroup': Eth,
                'ParentEduc': ParEdu,
                'LunchType': LunTyp,
                'TestPrep': TesPre,
                'ParentMaritalStatus': ParMaritStat,
                'PracticeSport': PracSpo,
                'IsFirstChild': Is1stkid,
                'NrSiblings': NrSib,
                'TransportMeans': Trans,
                'WklyStudyHours': Weekly}
    
    tmp=pd.DataFrame.from_dict(data_baru,orient='index').transpose()

    # Mapping the Gender
    gender_mapping = {
        'Female': 0,
        'Male': 1
    }

    # Mapping the LunchType
    lunch_mapping = {
        'Free/Reduced': 0,
        'Standard': 1
    }

    # Mapping the IsFirstChild
    value_mapping = {
        'No': 0,
        'Yes': 1
    }

    # Mapping the TestPrep
    test_mapping = {
        'None': 0,
        'Completed': 1
    }

    # Mapping the Schoolbus
    bus_mapping = {
        'Private': 0,
        'School Bus': 1
    }

    # Mapping the Sport Activity
    sport_mapping = {
        'Never': 0,
        'Sometimes': 1,
        'Regularly': 2
    }

    # Mapping the Parent Education
    pedu_mapping = {
        'High School': 0,
        'College': 1,
        'Associates Degree': 2,
        'Bachelors Degree':3,
        'Masters Degree':4
    }

    # Mapping the Weekly Study Hours
    weekly_mapping = {
        'Less than 5 hours': 0,
        'Between 5-10 hours': 1,
        'More than 10 hours': 2
    }

    # Mapping the Ethnic Group
    ethnic_mapping = {
        'group A': 0,
        'group B': 1,
        'group C': 2,
        'group D': 3,
        'group E': 4
    }

    # Fixing the values in the column
    tmp['Gender'] = tmp['Gender'].map(gender_mapping)
    tmp['LunchType'] = tmp['LunchType'].map(lunch_mapping)
    tmp['IsFirstChild'] = tmp['IsFirstChild'].map(value_mapping)
    tmp['TestPrep'] = tmp['TestPrep'].map(test_mapping)
    tmp['TransportMeans'] = tmp['TransportMeans'].map(bus_mapping)
    tmp['PracticeSport'] = tmp['PracticeSport'].map(sport_mapping)
    tmp['ParentEduc'] = tmp['ParentEduc'].map(pedu_mapping)
    tmp['WklyStudyHours'] = tmp['WklyStudyHours'].map(weekly_mapping)
    tmp['EthnicGroup'] = tmp['EthnicGroup'].map(ethnic_mapping)

    #One Hot Encoding
    categorical_cols = ['ParentMaritalStatus']
    for col in categorical_cols:
        tmp = pd.get_dummies(tmp, columns=[col], prefix = [col], drop_first=False)

    #add missing column
    for kolom in one_hot_columns:
        if kolom not in tmp.columns:
            tmp[kolom] = 0

    tmp = tmp[one_hot_columns]

    scaled_tmp = pd.DataFrame(scaler.transform(tmp))
    scaled_tmp.columns = tmp.columns.values
    scaled_tmp.index = tmp.index.values

    #Making prediction
    math_scores = elastic_math_model.predict(scaled_tmp)
    read_scores = elastic_read_model.predict(scaled_tmp)
    write_scores = elastic_write_model.predict(scaled_tmp)
    return math_scores, read_scores, write_scores

if __name__ == "__main__":
    main()

#How to run
# type in terminal python -m streamlit run app.py
