import streamlit as st
from  sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, roc_auc_score, classification_report, accuracy_score, confusion_matrix
import joblib
import urllib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
#from PIL import Image

def load_weights_buildModel():
    st.progress(1)
    st.text("Loading Random Forest Weights ....... ")
    #image = Image.open('loading.jpg')
    #st.image(image, "Loading")
    loaded_rf = joblib.load("random_forest.joblib")
    st.text("Random Forest Model loaded ....... ")
    st.progress(100)


def main():
    # Render the readme as markdown using st.markdown.
    readme_text = st.markdown(get_file_content_as_string("instructions.md"))

    # Download external dependencies.
    #for filename in EXTERNAL_DEPENDENCIES.keys():
     #   download_file(filename)
    #load_weights_buildModel()
    st.text("Loading Random Forest Weights ....... ")
    st.progress(1)
    #image = Image.open('loading.jpg')
    #st.image(image, "Loading")
    loaded_rf = joblib.load("random_forest.joblib")
    st.progress(100)
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
    resuourceType = None
    severityType = None
    featureType = None
    volume = None
    eventType = None
    provideAll = False

    if selectoption == "Single Location":
        provideAll = st.checkbox("Provide all details")
        locationId = st.text_input("Enter location details: ", key="Locationid")
        id = st.text_input("Enter Id: ", key="id")
        if provideAll:
            eventType = st.text_input("Enter Event Type details: ", key="eventId")
            resuourceType = st.text_input("Enter Resource Type details: ", key="resourceid")
            severityType = st.text_input("Enter Severity Type details: ", key="severityid")
            featureType = st.text_input("Enter Log Feature Type details: ", key="featureid")
            volume = st.text_input("Enter Log Volume details: ", key="volumeid")


    elif selectoption == "Multiple Location":
        uploadfile = st.file_uploader("Upload CSV")
    #st.button("Submit")

    if st.button("Submit"):
        processTestData(uploadfile,loaded_rf,locationId, id, resuourceType, severityType, featureType, volume, eventType, provideAll)
    return

def converttoFaultSeverity(x):
    stringarr = []
    for i in x:
        if i == 0:
            stringarr.append("No fault")
        elif i == 1:
            stringarr.append("Few Faults")
        elif i == 2:
            stringarr.append("Many Faults")
    return stringarr

def color_survived(val):
    color = 'red' if val=="Many Faults" else 'yellow' if val=="Few Faults" else 'green'
    return f'background-color: {color}'



def processTestData(uploadfile, loaded_rf, locationId, id, resuourceType, severityType, featureType, volume, eventType, processTestData):


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
        #dfProb = pd.DataFrame(loaded_rf.predict(test_X))
        #st.dataframe(dfProb)
        dfProbArray = (loaded_rf.predict(test_X))
        #st.dataframe(dfProb)
        #st.table(dfProb)

        dfProb = pd.DataFrame()
        dfProb['Id'] = testdataframe['id']
        dfProb['Location'] = testdataframe['location']
        dfProb['Fault Severity'] = converttoFaultSeverity(dfProbArray)
        st.table(dfProb.style.applymap(color_survived, subset=['Fault Severity']))

    elif locationId is not None:
        data = {'id': [id], 'location': [locationId]}
        testdataframe = pd.DataFrame(data)
        testdataframe['id'] = testdataframe['id'].astype(int)
        if (processTestData and len(resuourceType) >0 and len(severityType)>0 and len(featureType)>0 and len(volume)>0 and len(eventType) >0) :
            st.text("All details provided, cosiderin the provided inputs")
            #, , featureType, volume,
            data = {'id': [id], 'location': [locationId], 'resource_type':[resuourceType], 'severity_type': [severityType], 'log_feature': [featureType], 'volume':[volume], 'event_type':[eventType]}
            traineventseverityFeaturesResourcesdf = pd.DataFrame(data)
            traineventseverityFeaturesResourcesdf['id'] = traineventseverityFeaturesResourcesdf['id'].astype(int)
            traineventseverityFeaturesResourcesdf['volume'] = traineventseverityFeaturesResourcesdf['volume'].astype(int)

        else:
            st.text("Details are missing and hence considering the train data for inference")
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
        dfProbArray = (loaded_rf.predict(test_X))
        #st.dataframe(dfProb)
        #st.table(dfProb)
        dfProb = pd.DataFrame()
        dfProb['Id'] = data['id']
        dfProb['Location'] = data['location']
        dfProb['Fault Severity'] = converttoFaultSeverity(dfProbArray)
        st.table(dfProb.style.applymap(color_survived, subset=['Fault Severity']))

        #dfProb = pd.DataFrame((converttoFaultSeverity(dfProbArray)))

        #dfProb['id'] = data['id']
        #dfProb['location'] = data['location']

        #st.table(dfProb)

if __name__ == "__main__":
    main()
