import streamlit as st
from  sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, roc_auc_score, classification_report, accuracy_score, confusion_matrix
import joblib
import urllib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def load_weights_buildModel():
    st.text("Loading Random Forest Weights ....... ")
    loaded_rf = joblib.load("random_forest.joblib")
    st.text("Random Forest Model loaded ....... ")


def main():
    # Render the readme as markdown using st.markdown.
    readme_text = st.markdown(get_file_content_as_string("instructions.md"))

    # Download external dependencies.
    #for filename in EXTERNAL_DEPENDENCIES.keys():
     #   download_file(filename)
    #load_weights_buildModel()
    st.text("Loading Random Forest Weights ....... ")
    loaded_rf = joblib.load("random_forest.joblib")
    st.text("Random Forest Model loaded ....... ")

    # Once we have the dependencies, add a selector for the app mode on the sidebar.
    st.sidebar.title("What to do")
    app_mode = st.sidebar.selectbox("Choose the app mode",
        ["Show instructions", "Run the app"]) #, "Show the source code"
    if app_mode == "Show instructions":
        st.sidebar.success('To continue select "Run the app".')
   # elif app_mode == "Show the source code":
   #    readme_text.empty()
   #    st.code(get_file_content_as_string("streamlit_app.py"))
    elif app_mode == "Run the app":
        readme_text.empty()
        run_the_app(loaded_rf)

# Download a single file and make its content available as a string.
@st.experimental_singleton(show_spinner=False)
def get_file_content_as_string(path):
    file = open(path, "r")
    return file.read()
    #url = 'https://raw.githubusercontent.com/streamlit/demo-self-driving/master/' + path
    # response = urllib.request.urlopen(url)
    return  #response.read().decode("utf-8")


# This is the main app app itself, which appears when the user selects "Run the app".
def run_the_app(loaded_rf):
    st.title("Enter the Prediction details")
    selectoption = st.selectbox("Select Option",["Single Location","Multiple Location"])
    uploadfile = None
    locationId = None
    id = None
    if selectoption == "Single Location":
        locationId = st.text_input("Enter location details: ", key="Locationid")
        id = st.text_input("Enter Id: ", key="id")

    elif selectoption == "Multiple Location":
        uploadfile = st.file_uploader("Upload CSV")
    #st.button("Submit")

    if st.button("Submit"):
        processTestData(uploadfile,loaded_rf,locationId, id)
    return

def processTestData(uploadfile, loaded_rf, locationId, id):


    if uploadfile is not None:
        testdataframe = pd.read_csv(uploadfile)
        #st.write(dataframe)

        eventTypedf = pd.read_csv("event_type.csv")
        severityTypedf = pd.read_csv("severity_type.csv")
        featuresTypedf = pd.read_csv("log_feature.csv")
        eventTypedf = pd.read_csv("event_type.csv")
        resourcedf = pd.read_csv("resource_type.csv")

        traineventdf = pd.merge(testdataframe, eventTypedf.drop_duplicates(subset=['id']), on='id')
        traineventseveritydf = pd.merge(traineventdf, severityTypedf.drop_duplicates(subset=['id']), how='left',
                                        on='id')
        traineventseverityFeaturesdf = pd.merge(traineventseveritydf, featuresTypedf.drop_duplicates(subset=['id']),
                                                how='left', on='id')
        traineventseverityFeaturesResourcesdf = pd.merge(traineventseverityFeaturesdf,
                                                         resourcedf.drop_duplicates(subset=['id']), how='left', on='id')
        lb = LabelEncoder()

        traineventseverityFeaturesResourcesdf['location_labelled'] = lb.fit_transform(
            traineventseverityFeaturesResourcesdf['location'])
        traineventseverityFeaturesResourcesdf['severity_type_labelled'] = lb.fit_transform(
            traineventseverityFeaturesResourcesdf['severity_type'])
        traineventseverityFeaturesResourcesdf['resource_type_labelled'] = lb.fit_transform(
            traineventseverityFeaturesResourcesdf['resource_type'])
        traineventseverityFeaturesResourcesdf['log_feature_labelled'] = lb.fit_transform(
            traineventseverityFeaturesResourcesdf['log_feature'])
        traineventseverityFeaturesResourcesdf['event_type_labelled'] = lb.fit_transform(
            traineventseverityFeaturesResourcesdf['event_type'])

        #target = traineventseverityFeaturesResourcesdf['fault_severity']
        train_X = traineventseverityFeaturesResourcesdf.drop(
            ['location', 'severity_type', 'resource_type', 'log_feature', 'event_type'], axis=1)
        #st.write(train_X)
        test_X = train_X.set_index(train_X.id).drop('id', axis=1)
        dfProb = pd.DataFrame(loaded_rf.predict(test_X))
        #st.dataframe(dfProb)
        st.table(dfProb)

    elif locationId is not None:
        data = {'id': [id], 'location': [locationId]}
        testdataframe = pd.DataFrame(data)

        testdataframe['id'] = testdataframe['id'].astype(int)
        eventTypedf = pd.read_csv("event_type.csv")
        severityTypedf = pd.read_csv("severity_type.csv")
        featuresTypedf = pd.read_csv("log_feature.csv")
        eventTypedf = pd.read_csv("event_type.csv")
        resourcedf = pd.read_csv("resource_type.csv")

        #testdataframe.drop(testdataframe.columns[0], axis=1)

        st.dataframe(testdataframe)
        st.dataframe(eventTypedf)

        traineventdf = pd.merge(testdataframe, eventTypedf.drop_duplicates(subset=['id']), on='id')
        traineventseveritydf = pd.merge(traineventdf, severityTypedf.drop_duplicates(subset=['id']), how='left',
                                        on='id')
        traineventseverityFeaturesdf = pd.merge(traineventseveritydf, featuresTypedf.drop_duplicates(subset=['id']),
                                                how='left', on='id')
        traineventseverityFeaturesResourcesdf = pd.merge(traineventseverityFeaturesdf,
                                                         resourcedf.drop_duplicates(subset=['id']), how='left', on='id')
        lb = LabelEncoder()

        traineventseverityFeaturesResourcesdf['location_labelled'] = lb.fit_transform(
            traineventseverityFeaturesResourcesdf['location'])
        traineventseverityFeaturesResourcesdf['severity_type_labelled'] = lb.fit_transform(
            traineventseverityFeaturesResourcesdf['severity_type'])
        traineventseverityFeaturesResourcesdf['resource_type_labelled'] = lb.fit_transform(
            traineventseverityFeaturesResourcesdf['resource_type'])
        traineventseverityFeaturesResourcesdf['log_feature_labelled'] = lb.fit_transform(
            traineventseverityFeaturesResourcesdf['log_feature'])
        traineventseverityFeaturesResourcesdf['event_type_labelled'] = lb.fit_transform(
            traineventseverityFeaturesResourcesdf['event_type'])

        #target = traineventseverityFeaturesResourcesdf['fault_severity']
        train_X = traineventseverityFeaturesResourcesdf.drop(
            ['location', 'severity_type', 'resource_type', 'log_feature', 'event_type'], axis=1)
        #st.write(train_X)
        test_X = train_X.set_index(train_X.id).drop('id', axis=1)
        dfProb = pd.DataFrame(loaded_rf.predict(test_X))
        #st.dataframe(dfProb)
        st.table(dfProb)

if __name__ == "__main__":
    main()
