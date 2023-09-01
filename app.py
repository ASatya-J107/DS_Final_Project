import streamlit as st
import pickle
import pandas as pd
import scikit-learn as sklearn

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
    gen = left.selectbox('Gender', ('female', 'male'))
    Eth = right.selectbox('Ethnic Group', ('group A', 'group B', 'group C', 'group D', 'group E'))
    ParEdu = left.selectbox('Parent Education', ('associates degree', 'bachelors degree', 'high school', 'masters degree','some college','some high school'))
    LunTyp = right.selectbox('Lunch Type', ('free/reduced', 'standard'))
    TesPre = left.selectbox('Test Preparation', ('none', 'completed'))
    ParMaritStat = right.selectbox('Parent Marital Status', ('divorced', 'married', 'single', 'widowed'))
    PracSpo = left.selectbox('Practice Sport', ('never', 'regularly', 'sometimes'))
    Is1stkid = right.selectbox('First Child', ('yes', 'no'))
    NrSib = left.number_input('Number of Sibling', step=1, value=0, format="%d")
    Trans = right.selectbox('Mode of Transportation', ('private', 'school_bus'))
    Weekly = st.selectbox('Weekly Study Hours', ('Less than 5 hours', 'Between 5-10 hours', 'More than 10 hours'))
    button = st.button('Predict')

    #if button is clicked (ketika button dipencet)
    if button:
        #make prediction
        math_scores, read_scores, write_scores = predict(gen, Eth, ParEdu, LunTyp, TesPre, ParMaritStat, PracSpo, Is1stkid, NrSib, Trans, Weekly)
        if math_scores >= 75:
            st.success("Math Scores: {:.2f}".format(math_scores[0]))
        else:
            if math_scores >= 70:
                st.warning("Math Scores: {:.2f}".format(math_scores[0]))
            else:
                st.error("Math Scores: {:.2f}".format(math_scores[0]))

        if read_scores >= 75:
            st.success("Reading Scores: {:.2f}".format(read_scores[0]))
        else:
            if read_scores >= 70:
                st.warning("Reading Scores: {:.2f}".format(math_scores[0]))
            else:
                st.error("Reading Scores: {:.2f}".format(math_scores[0]))

        if write_scores >= 75:
            st.success("Write Scores: {:.2f}".format(write_scores[0]))
        else:
            if write_scores >= 70:
                st.warning("Write Scores: {:.2f}".format(math_scores[0]))
            else:
                st.error("Write Scores: {:.2f}".format(math_scores[0]))


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
        'female': 0,
        'male': 1
    }

    # Mapping the LunchType
    lunch_mapping = {
        'free/reduced': 0,
        'standard': 1
    }

    # Converting IsFirstChild to object type
    tmp['IsFirstChild'] = tmp['IsFirstChild'].astype(object)

    # Mapping the IsFirstChild
    value_mapping = {
        'no': 0,
        'yes': 1
    }

    # Mapping the TestPrep
    test_mapping = {
        'none': 0,
        'completed': 1
    }

    # Mapping the Schoolbus
    bus_mapping = {
        'private': 0,
        'school_bus': 1
    }

    # Fixing the values in the column
    tmp['Gender'] = tmp['Gender'].map(gender_mapping)
    tmp['LunchType'] = tmp['LunchType'].map(lunch_mapping)
    tmp['IsFirstChild'] = tmp['IsFirstChild'].map(value_mapping)
    tmp['TestPrep'] = tmp['TestPrep'].map(test_mapping)
    tmp['TransportMeans'] = tmp['TransportMeans'].map(bus_mapping)

    #One Hot Encoding
    categorical_cols = ['EthnicGroup', 'ParentEduc', 'ParentMaritalStatus', 'PracticeSport', 'WklyStudyHours']
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
