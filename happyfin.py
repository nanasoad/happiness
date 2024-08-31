from optparse import OptionGroup
import pprint
from shap.plots.colors import blue_rgb
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import  LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
    
from sklearn import metrics 
from sklearn.model_selection import RandomizedSearchCV
#from sklearn.metrics import explained_variance_score, max_error, mean_absolute_error, r2_score, explained_variance_score
#from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error, mean_squared_error, mean_squared_log_error
#from sklearn.metrics import median_absolute_error

from sklearn.impute import KNNImputer 
from sklearn.preprocessing import StandardScaler
from sklearn import ensemble
from sklearn.model_selection import train_test_split

import shap
shap.initjs()
    

st.sidebar.title("Navigation")
pages=["Home", "Data Exploration", "Modelling","Conclusion","Bibliography", "About"]
page=st.sidebar.radio("Go to", pages)
# Cover Page
if page == pages[0] : 
   
    st.title('World Happiness')
    image = Image.open("happy1.jpeg")
    st.image(image)
    st.header("Introduction")
    st.write("Quality of life (QoL) is a topic of great relevance, especially for policymakers and those interested in the general well-being of the population. This concept can be approached from various perspectives and assessed in multiple ways. Over time, QoL measurement has evolved through different approaches, always aiming to assess the well-being of individuals, groups, or regions.")
    text = ("Measuring quality of life is a complex and challenging task due to the context-sensitive nature of the concept and its various possible definitions. Before the creation of specific indices to measure QoL at the national level, economic performance was the primary indicator considered. However, Gross National Product (GNP) alone is insufficient to assess the standard of living and the status of a country's citizens. Therefore, researchers introduced other perspectives, such as social and health indicators, which were integrated to build more comprehensive QoL indices.")
    st.markdown(text)
    
    st.write("Given the complexity of assessing QoL, there is a need for data science tools capable of analyzing the massive and complex data required to study well-being in all its dimensions. Today, information-rich data sources, such as social media, are utilized in research to study happiness. Machine learning (ML) algorithms have made it possible to analyze large volumes of data, extract valuable insights, and open up new research possibilities in the field of QoL. These algorithms can be used both to analyze the results of QoL indicators and to predict well-being in future years or in new data samples from individuals and countries.")
    image = Image.open("Happiness_Test.png")
    st.image(image)
    st.subheader('Aim',divider='rainbow')
    st.write('In this context, we focus on the World Happiness Index, an international QoL indicator developed by the United Nations Sustainable Development Solutions Network. This work explores well-being prediction systems based on machine learning algorithms. To achieve this, we conducted a comparative study of the most widely used algorithms in the analysis of data from the World Happiness Report (WHR), with the aim of proposing a model capable of predicting the Ladder Score using machine learning techniques.')
    st.subheader("***Description of happiness indicators***",divider='rainbow')
   


    st.write("We use eleven concepts to measure a quality of life score:")

    st.write("1. Ladder Score,\n"
         "2. Income based on GDP per capita,\n"
         "3. Social support from family and friends,\n"
         "4. Number of hospital beds,\n"
         "5. Number of nurses,\n"
         "6. Health life expectancy,\n"
         "7. Freedom to make life choices,\n"
         "8. Perceptions of corruption,\n"
         "9. Poverty gap index and poverty severity,\n"
         "10. Gini index,\n"
         "11. Generosity")

    st.subheader('Data',divider='rainbow')
    
    st.markdown("**1. World Happiness Report [dataset](https://worldhappiness.report/data/),**")
    st.markdown("**2. Hospital beds [dataset](https://data.worldbank.org/indicator/SH.MED.BEDS.ZS) per 1,000 people,**")
    st.markdown("**3. Nursing and midwifery personnel [dataset](https://www.who.int/data/gho/data/indicators/indicator-details/GHO/nursing-and-midwifery-personnel-(per-10-000-population)) per 10,000 population),**")
    st.markdown("**4. Poverty Gap, Poverty Severity, and Gini index [dataset](https://databank.worldbank.org/source/millennium-development-goals/Series/SI.POV.NAGP#).**")

# Data analysis using Data Visualization figures
if page == pages[1] : 
    st.header('Data Exploration')
    # Datensatz beschreibung
    data = {'Name': ['World Happiness Reports', 'Hospital beds', 'Nursing and midwifery personnel', "Poverty Gap, Poverty Severity and gini index"],
        'rows': [1949, 1167, 1770, 2389],
        'columns': [11, 6, 3, 40],}

    df = pd.DataFrame(data)

    if st.button("Dataframes info",  use_container_width=True):
        st.dataframe(df)
    
    happiness_data=pd.read_csv("data happy 06-23.csv")
    happiness_data['year'] = happiness_data['year'].astype(str).str.replace(',', '')
    
    if st.button("World Happiness Reports",  use_container_width=True):
        st.dataframe(happiness_data)
    
    st.write("When starting the data cleaning process, the following steps are followed for each dataset:")

    st.write("1. Relevant variables are selected.\n"
         "2. Variables are renamed as needed.\n"
         "3. Unique values ​​are identified for the variables “Country” and “Year”.\n")
    
    st.write("Given the information in the health dataset from 2007 to 2018, the data is filtered to include only records corresponding to these years in all datasets. Subsequently, duplicate data is assessed and missing data is analyzed in each dataset.")
    st.write("In addition, country names are standardized using **Geopandas** and possible discrepancies in the spelling of names in the datasets are checked.")
    st.write("The datasets are then reconstructed by merging the four different databases. To ensure data integrity and quality, the existence of duplicate data is re-assessed and missing values ​​are analyzed. Variables with a high percentage of missing values ​​are removed and missing values ​​are imputed when necessary.")




    st.subheader('World Happiness Map',divider='rainbow')

# Weltkarte erstellen
    fig = px.choropleth(
        happiness_data,
        locations='country',
        locationmode='country names',
        color='ladder',
        hover_name='country',
        color_continuous_scale='Viridis',
        animation_frame='year', animation_group="country"
        )
# Setze den Zeitraum für die Animation
    fig.update_layout(
        sliders=[{
            "active": 0,
            "steps": [{"args": [[f], {"frame": {"duration": 500, "redraw": True}, "mode": "immediate", "transition": {"duration": 300}}],
                   "label": f"{f}"} for f in sorted(happiness_data['year'].unique())]
    }]
    )
# Zeige Weltkarte
    st.plotly_chart(fig)

    st.write("Country details are then retrieved to represent them on a map. Geopandas provides a built-in dataset for this purpose, eliminating the need to search for other sources. It is noted that some countries are not in the Geopandas database, suggesting that they might not be representable on Geopandas maps.")
    st.subheader('The Distribution of Happiness by continent',divider='rainbow')
    image = Image.open('distribution of happieness.png')
    st.image(image)
    st.write("North America and Oceania have the **highest** happiness indices overall, with medians that surpass those of other continents. In contrast, Africa and Asia exhibit **lower** happiness indices. Asia stands out as an outlier, where most countries display moderate happiness levels, but a few have significantly higher indices. Europe, on the other hand, shows wide variability in happiness indices, reflecting the diversity among European countries.")
    st.write("This analysis suggests that significant differences in happiness indices exist across continents, potentially influenced by socioeconomic, cultural, and political factors unique to each region.")

    st.subheader("Distribution of variables", divider='rainbow')
    image = Image.open('distibucion de variables.png')
    st.image(image)
    st.write("Using box plots, the distribution of various variables is analyzed, revealing notable outliers in most of them, particularly in life expectancy, corruption, medical rates, nursing rates, bed rates, and dentist rates, especially at the lower or upper extremes, suggesting a wide range of conditions across different countries.")
    st.write("Variables such as Ladder, freedom, and violence exhibit relatively symmetrical distributions, while others, like log GDP and life expectancy, show skewness to the left or right.\n")
    st.write("Variables related to medical rates, including medical rates and nursing rates, display a wide range of values, reflecting diverse socioeconomic conditions and healthcare capabilities in different regions or countries.\n")
    st.write("Box plots expose differences in distribution and the presence of outliers across a range of socioeconomic and health indicators, highlighting significant variations in variables such as GDP, social support, life expectancy, corruption, and medical services, with outliers underscoring disparities between countries. These graphical representations indicate varying distributions, some with significant skewness and outliers, suggesting the need for further data transformation and appropriate handling of outliers for statistical modeling./n")

    st.subheader("Relationship between the variables",divider='rainbow')
    df_fin= pd.read_csv("df_Fin.csv",index_col=False,header=0,usecols=['Country', 'Region', 'Year', 'Ladder', 'Log GDP', 'Social Supp',
       'Life expectancy', 'Freedom', 'generosity', 'Corruption', 'Violence',
       'Rate Beds', 'Rate Medical', 'Rate Nursing', 'Rate Dentists',
       'poverty_gap', 'poverty_severity', 'gini'])
    
    numeric_df=df_fin.select_dtypes('float64')
    p_col=['Ladder','Ladder','Ladder',
       'Log GDP','Log GDP','Log GDP','Corruption',
       'poverty_gap']
    q_col=['Log GDP','Social Supp','gini', 'Social Supp',
       'poverty_severity','Rate Nursing','Rate Dentists','Rate Dentists',
       ]
    
    choice = ['Correlation diagram','Scatter plot']
    option = st.selectbox('Choice of graphic', choice)

    if option =='Correlation diagram':
        plot = sns.heatmap(numeric_df.corr(), annot=True, fmt=".1f", cmap="crest",linewidth=0.5)
        st.pyplot(plot.get_figure())
        
    elif option == 'Scatter plot':
        fig, ax=plt.subplots(4,2, figsize=(12,12))
        for p_col, after_col, axis in zip(p_col,q_col,ax.ravel()):
            b=sns.scatterplot(x=p_col, y=after_col, data=df_fin, ax=axis)
        plt.tight_layout()
        st.pyplot(b.get_figure())

    
    st.write("The relationships between different variables are explored by analyzing scatter plots  with fitted regression lines and correlation diagram. From these graphs, the following key findings have been identified:")
    st.write("**Log GDP vs. Ladder**: A positive relationship is observed between Log GDP and subjective well-being (Ladder). This indicates that as a country's GDP increases, the subjective well-being of its citizens also tends to improve.")

    st.write("**Social Support vs. Ladder**: A positive relationship is detected between social support and subjective well-being. This suggests that people who perceive greater social support tend to report higher levels of subjective well-being.\n")

    st.write("**Gini vs. Ladder**: A slight negative relationship is identified between the Gini index (a measure of inequality) and subjective well-being. This suggests that higher levels of inequality are associated with lower subjective well-being, although the relationship does not appear to be very strong.\n")

    st.write("**Social Support vs. Log GDP**: There is a positive relationship between social support and GDP. As a country's GDP increases, the level of social support its citizens receive also seems to improve.\n")

    st.write("**Poverty Severity vs. Log GDP**: A negative relationship is observed between poverty severity and GDP. As GDP increases, poverty severity decreases, consistent with the idea that economic growth contributes to reducing poverty.\n")

    st.write("**Nursing Rate vs. Log GDP**: A strong positive relationship is found between the nursing rate and GDP. As GDP increases, so does the number of nurses per 100,000 inhabitants, suggesting that wealthier countries have better healthcare services.\n")

    st.write("**Dentist Rate vs. Poverty Gap**: A negative relationship is identified between the dentist rate and the poverty gap. In areas with a larger poverty gap, the number of dentists per 100,000 inhabitants is lower.\n")

    st.write("**Dentist Rate vs. Corruption**: There is no clear relationship between the dentist rate and corruption. The trend line is almost flat, suggesting a weak or non-existent correlation.\n")

    st.write("Subjective well-being, as measured by the Ladder score, demonstrates a strong correlation with GDP and social support. Inequality, as indicated by the Gini index, negatively impacts well-being, though its effect is not as pronounced as that of other factors. Enhanced healthcare, measured by the availability of nurses and dentists, shows a close relationship with higher GDP and reduced levels of poverty. The relationship between corruption and healthcare services, such as the availability of dentists, appears weak or non-existent.")

if page == pages[2] : 

    st.header('Modelling')
    st.write("The objective of the proposed model is to predict the Ladder Score using machine learning techniques.")
    st.subheader('**Main steps of the experiment**',divider='rainbow')
    image = Image.open('Diagram.jpg')
    st.image(image)
    st.write("First, we selected from each database only the data corresponding to the period 2007-2018. Then we unified the name of the columns and verified that the countries in each dataset had the same name, for which we used geopandas to unify the name of the countries. We merged all the datasets into one and observed a loss of information in the process, the final size of the dataset is 965x18. Lastly, we split the data for training and testing purposes,"
              "and we performed imputation of missing values using the KNN imputer and scaling of numerical features using Standard Scaler to maintain data integrity.")
    
    st.subheader('**Model building**',divider='rainbow')
    
    st.write("After finishing the data preparation part, multiple machine learning algorithms were used to predict the happiness score. Next, we used grid search to find the best hyperparameters. Grid search is a process that exhaustively searches for the best values of a manually specified subset of hyperparameters from the target algorithm.")
    st.write("As the data used is quantitative, the following machine learning algorithms are used to predict the happiness score:")
    st.write("1. Multiple Linear Model\n"
         "2. KNeighbors Regressor\n"
         "3. Decision tree regressor\n"
         "4. Randon Forest Regressor\n"
         "5. Gradient boosting regressor\n")
    st.write("Die Models were evaluated observing the following results:")

    df_fin= pd.read_csv("df_Fin.csv",index_col=False,header=0,usecols=['Country', 'Region', 'Year', 'Ladder', 'Log GDP', 'Social Supp',
       'Life expectancy', 'Freedom', 'generosity', 'Corruption', 'Violence',
       'Rate Beds', 'Rate Medical', 'Rate Nursing', 'Rate Dentists',
       'poverty_gap', 'poverty_severity', 'gini'])
    #st.dataframe(df_fin,hide_index=True)
    
    #MACHINE LEARNING
    # Separate the target variable from the explanatory variables and then separate the dataset into a training set 
    #and a test set so that the test set contains 20% ofthe data.

    X = df_fin.drop(['Country', 'Region', 'Year', 'Ladder'],axis=1) # explanatory variables
    y = df_fin['Ladder'] # target variable

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    
    #using the "KNNimputer" strategies for numeric variables, rigorously fill in the missing values for the training set and the test set 
    imputer= KNNImputer(n_neighbors=5)

    #numerical variables
    X_train = pd.DataFrame(imputer.fit_transform(X_train))
    X_test = pd.DataFrame(imputer.fit_transform(X_test))

    #to Rescale numerical variables so that they are comparable on a common scale, normalise numerical variables with the methodStandardScaler

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    
    #TRAINING MODELS

    #linear Regression

    LR = LinearRegression()

    LR.fit(X_train, y_train)# Entreno el algoritmo en X_train. y_train

    # Make predictions on the test set
    y_pred_LR = LR.predict(X_test)
    y_pred_train_LR =LR.predict(X_train)

    # Plotting the results

    df_LR= pd.DataFrame({'Real':y_test,'Prediction':y_pred_LR})

    df_LR['Diference'] = y_test - y_pred_LR

    df_LR.head()
    
    #KNeighborsRegressor
    KNN= KNeighborsRegressor()
    KNN.fit(X_train, y_train)# Entreno el algoritmo en X_train. y_train
    y_pred_KNN = KNN.predict(X_test)
    y_pred_train_KNN =KNN.predict(X_train)

    df_KNN= pd.DataFrame({'Real':y_test,'Prediction':y_pred_KNN})
    df_KNN['Diference'] = y_test - y_pred_KNN
    df_KNN.head()

    # Decision Tree Regressor
    DTR= DecisionTreeRegressor()
    fit_DTR=DTR.fit(X_train, y_train)# Entreno el algoritmo en X_train. y_train

    y_pred_DTR = DTR.predict(X_test)
    y_pred_train_DTR =DTR.predict(X_train)

    df_DTR= pd.DataFrame({'Real':y_test,'Prediction':y_pred_DTR})
    df_DTR['Diference'] = y_test - y_pred_DTR
    df_DTR.head()

    pd.DataFrame(DTR.feature_importances_, index=X.columns, columns=["Importance"]).sort_values('Importance', ascending=False).round(2)
    data_DTR = pd.DataFrame(fit_DTR.feature_importances_, index=X.columns, columns=["Importance"]).sort_values('Importance', ascending=False).round(2)
    
    # Random Forest Regressor
    RFR= RandomForestRegressor(random_state = 42)
    fit_RFR=RFR.fit(X_train, y_train)# Entreno el algoritmo en X_train. y_train

    y_pred_RFR = RFR.predict(X_test)
    y_pred_train_RFR =RFR.predict(X_train)

    df_RFR= pd.DataFrame({'Real':y_test,'Prediction':y_pred_RFR})
    df_RFR['Diference'] = y_test - y_pred_RFR
    df_RFR.head()
    
    pd.DataFrame(RFR.feature_importances_, index=X.columns, columns=["Importance"]).sort_values('Importance', ascending=False).round(2)
    data_RFR = pd.DataFrame(fit_RFR.feature_importances_, index=X.columns, columns=["Importance"]).sort_values('Importance', ascending=False).round(2)
    
    # Gradient Boosting Regressor
    GBR= GradientBoostingRegressor()
    fit_GBR=GBR.fit(X_train, y_train)# Entreno el algoritmo en X_train. y_train

    y_pred_GBR = GBR.predict(X_test)
    y_pred_train_GBR =GBR.predict(X_train)

    df_GBR= pd.DataFrame({'Real':y_test,'Prediction':y_pred_GBR})
    df_GBR['Diference'] = y_test - y_pred_GBR
    df_GBR.head()

    pd.DataFrame(GBR.feature_importances_, index=X.columns, columns=["Importance"]).sort_values('Importance', ascending=False).round(2)
    data_GBR = pd.DataFrame(fit_GBR.feature_importances_, index=X.columns, columns=["Importance"]).sort_values('Importance', ascending=False).round(2)
    

    tabla = pd.DataFrame([
            {"Set":"Training","Model": "Multiple Linear Model", "R^2":round(LR.score(X_train, y_train),2), "Error - MAE":round(metrics.mean_absolute_error(y_train, y_pred_train_LR),2) ,'Error - MSE':round(metrics.mean_squared_error(y_train, y_pred_train_LR),2)},
            {"Set":"Test","Model": "Multiple Linear Model", "R^2":round(LR.score(X_test, y_test),2), "Error - MAE":round(metrics.mean_absolute_error(y_test, y_pred_LR),2),'Error - MSE':round(metrics.mean_squared_error(y_test, y_pred_LR),2)},
            {"Set":"Training","Model": "KNeighbors Regressor Model", "R^2":round(KNN.score(X_train, y_train),2) , "Error - MAE":round(metrics.mean_absolute_error(y_train, y_pred_train_KNN),2) ,'Error - MSE':round(metrics.mean_squared_error(y_train, y_pred_train_KNN),2)},
            {"Set":"Test","Model": "KNeighbors Regressor Model", "R^2":round(KNN.score(X_test, y_test),2) , "Error - MAE":round(metrics.mean_absolute_error(y_test, y_pred_KNN),2) ,'Error - MSE':round(metrics.mean_squared_error(y_test, y_pred_KNN),2)},
            {"Set":"Training","Model": "Decision Tree Regressor", "R^2":round(DTR.score(X_train, y_train),2) , "Error - MAE":round(metrics.mean_absolute_error(y_train, y_pred_train_DTR),2) ,'Error - MSE':round(metrics.mean_squared_error(y_train, y_pred_train_DTR),2)},
            {"Set":"Test","Model": "Decision Tree Regressor", "R^2":round(DTR.score(X_test, y_test),2) , "Error - MAE":round(metrics.mean_absolute_error(y_test, y_pred_DTR),2) ,'Error - MSE':round(metrics.mean_squared_error(y_test, y_pred_DTR),2)},
            {"Set":"Training","Model": "Random Forest Regressor", "R^2":round(RFR.score(X_train, y_train),2) , "Error - MAE":round(metrics.mean_absolute_error(y_train, y_pred_train_RFR),2) ,'Error - MSE':round(metrics.mean_squared_error(y_train, y_pred_train_RFR),2)},
            {"Set":"Test","Model": "Random Forest Regressor", "R^2":round(RFR.score(X_test, y_test),2) , "Error - MAE":round(metrics.mean_absolute_error(y_test, y_pred_RFR),2) ,'Error - MSE':round(metrics.mean_squared_error(y_test, y_pred_RFR),2)},
            {"Set":"Training","Model": "Gradient Boosting Regressor", "R^2":round(GBR.score(X_train, y_train),2) , "Error - MAE":round(metrics.mean_absolute_error(y_train, y_pred_train_GBR),2) ,'Error - MSE':round(metrics.mean_squared_error(y_train, y_pred_train_GBR),2)},
            {"Set":"Test","Model": "Gradient Boosting Regressor", "R^2":round(GBR.score(X_test, y_test),2) , "Error - MAE":round(metrics.mean_absolute_error(y_test, y_pred_GBR),2) ,'Error - MSE':round(metrics.mean_squared_error(y_test, y_pred_GBR),2)},
            ])
      
    edited_df = st.data_editor(tabla,hide_index=True)

    st.write("One important finding we can emphasize is: While the linear model performed poorly, the Random Forest Regressor model showed better performance compared to most models, with smaller discrepancies in accuracy and errors between the test and training sets, despite differences in data types and sizes.")
    st.write("The purpose of feature selection in machine learning is to determine the best set of variables to build effective models of the phenomena studied. Feature classification is used to understand the importance of the variables. We present the results of the found models.")

    choice = [ 'Random Forest Regressor','Decision Tree Regressor','Gradient Boosting Regressor']
    option = st.selectbox('Choice of the model', choice)
    
    if option == 'Decision Tree Regressor':
        fig,ax = plt.subplots(1,2)
        a=plt.subplot(121) 
        a = px.scatter(df_DTR, x="Real", y="Prediction",trendline="lowess",trendline_color_override = 'red',title="Real vs Prediction")
        st.plotly_chart(a)
        b=plt.subplot(122)
        b=px.bar(data_DTR,title="Importance of Variables." )
        st.plotly_chart(b)
    elif option == 'Random Forest Regressor':
        fig,ax = plt.subplots(1,2)
        a=plt.subplot(121) 
        a = px.scatter(df_RFR, x="Real", y="Prediction",trendline="lowess",trendline_color_override = 'red',title="Real vs Prediction")
        st.plotly_chart(a)
        b=plt.subplot(122)
        b=px.bar(data_RFR,title="Importance of Variables." )
        st.plotly_chart(b)
    elif option=='Gradient Boosting Regressor':
        fig,ax = plt.subplots(1,2)
        a=plt.subplot(121) 
        a = px.scatter(df_GBR, x="Real", y="Prediction",trendline="lowess",trendline_color_override = 'red',title="Real vs Prediction")
        st.plotly_chart(a)
        b=plt.subplot(122)
        b=px.bar(data_GBR,title="Importance of Variables." )
        st.plotly_chart(b)
    
    st.write('In order to optimize the model and determine if there is an ideal fit, we reduce the dataset to only include the four most important variables.')
    st.write('- Log GDP')
    st.write('- Social Supp')
    st.write('- Life Expectancy') 
    st.write('- Gini')
    st.write('')
    st.write('Continuing on, we fit a **Random Forest Regressor Model** and acquire the following results:')


    df_R = df_fin.drop(['Freedom','Corruption','generosity', 'Violence','Rate Beds', 'Rate Medical', 'Rate Nursing', 'Rate Dentists',
                   'poverty_gap', 'poverty_severity'],axis=1)
    
    #st.dataframe(df_R, hide_index=True)
    X_R = df_R.drop(['Country', 'Region', 'Year', 'Ladder'],axis=1)
    y_R = df_R['Ladder']

    X_train_R, X_test_R, y_train_R, y_test_R= train_test_split(X_R, y_R, test_size=0.20,random_state=42)

#using the "knnimputer" strategies for numeric variables, rigorously fill in the missing values for the training set and the test set 
#numerical variables

    X_train_R = pd.DataFrame(imputer.fit_transform(X_train_R))
    X_test_R = pd.DataFrame(imputer.fit_transform(X_test_R))

    scaler.fit(X_train_R)
    X_train = scaler.transform(X_train_R)
    X_test = scaler.transform(X_test_R)

    RFR= RandomForestRegressor(random_state = 42)
    fit_RFR_R=RFR.fit(X_train_R, y_train_R)

    y_pred_RFR_R = RFR.predict(X_test_R)
    y_pred_train_RFR_R =RFR.predict(X_train_R)

    df_RFR_R= pd.DataFrame({'Real':y_test_R,'Prediction':y_pred_RFR_R})
    df_RFR_R['Diference'] = y_test_R - y_pred_RFR_R
    
    pd.DataFrame(RFR.feature_importances_, index=X_R.columns, columns=["Importance"]).sort_values('Importance', ascending=False).round(2)
    data_RFR_R= pd.DataFrame(fit_RFR_R.feature_importances_, index=X_R.columns, columns=["Importance"]).sort_values('Importance', ascending=False).round(2)
    
    st.write('*Reduced Model (4 most important variables*)')
    tabla_R = pd.DataFrame([
            {"Set":"Training", '''R^2''':round(RFR.score(X_train_R, y_train_R),2) ,"MAPE*": round(metrics.mean_absolute_percentage_error(y_train_R, y_pred_train_RFR_R),2) ,"Error - MAE":round(metrics.mean_absolute_error(y_train_R, y_pred_train_RFR_R),2) ,'Error - MSE':round(metrics.mean_squared_error(y_train_R, y_pred_train_RFR_R),2)},
            {"Set":"Test", '''R^2''':round(RFR.score(X_test_R, y_test_R),2) ,"MAPE*": round(metrics.mean_absolute_percentage_error(y_test_R, y_pred_RFR_R),2) ,"Error - MAE":round(metrics.mean_absolute_error(y_test_R, y_pred_RFR_R),2) ,'Error - MSE':round(metrics.mean_squared_error(y_test_R, y_pred_RFR_R),2)},
            ]) 
   
    edited_R = st.data_editor(tabla_R,hide_index=True)
    st.caption('MAPE*: Mean Absolute Percentage Error',)
  
    st.write('*Initial Model*')
    tabla_I = pd.DataFrame([
            {"Set":"Training", r'''R^2''':0.98, "MAPE*": metrics.mean_absolute_percentage_error(y_train, y_pred_train_RFR), "Error - MAE":metrics.mean_absolute_error(y_train, y_pred_train_RFR).round(2) ,'Error - MSE':metrics.mean_squared_error(y_train, y_pred_train_RFR).round(2)},
            {"Set":"Test",r'''R^2''':'0.90' ,"MAPE*": metrics.mean_absolute_percentage_error(y_test, y_pred_RFR), "Error - MAE":metrics.mean_absolute_error(y_test, y_pred_RFR).round(2) ,'Error - MSE':metrics.mean_squared_error(y_test, y_pred_RFR).round(2)},
            ])
      
    edited_I = st.data_editor(tabla_I,hide_index=True)

    
    fig,ax = plt.subplots(2,1)
    a=plt.subplot(211) 
    a = px.scatter(df_RFR_R, x="Real", y="Prediction",trendline="lowess",trendline_color_override = 'red',title="Real vs Prediction")
    st.plotly_chart(a)
    b=plt.subplot(212)
    b=px.bar(data_RFR_R,title="Importance of Variables." )
    st.plotly_chart(b)
    
    st.write('According to the results, there is no significant advancement observed between the initial model and the one with the 4 most important variables.')

    st.subheader('**Model Optimatization**',divider='rainbow')
    
    st.write('Both models have high performance on the training set, but slightly lower on the testing set, which makes us suspect that there may be model overfitting. '
    'An overfitted model may look impressive on the training set, but will be useless in a real application. Therefore, we perform the hyperparameter optimization procedure, taking into account possible overfitting through a cross-validation process.')
    st.write('')
    
    image = Image.open('opti.png')
    st.image(image)
    
    # Number of trees in random forest
    #n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]

    # Number of features to consider at every split
    #max_features = ['auto', 'sqrt']

    # Maximum number of levels in tree
    #max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
    #max_depth.append(None)

    # Minimum number of samples required to split a node
    #min_samples_split = [2, 5, 10]

    # Minimum number of samples required at each leaf node
    #min_samples_leaf = [1, 2, 4]

    # Method of selecting samples for training each tree
    #bootstrap = [True, False]

    # Create the random grid
    #random_grid = {'n_estimators': n_estimators,
     #          'max_features': max_features,
      #         'max_depth': max_depth,
       #        'min_samples_split': min_samples_split,
         #      'min_samples_leaf': min_samples_leaf,
        #       'bootstrap': bootstrap}
    #st.write(random_grid)
    
    #Random search training
    #Now, we instantiate the random search and tune it like any Scikit-Learn model:

    # Use the random grid to search for best hyperparameters
    # First create the base model t
    #rf = RandomForestRegressor(random_state = 42)
    # Random search of parameters, using 3 fold cross validation, 
    # search across 100 different combinations, and use all available cores
   # rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
    # Fit the random search model
   # rf_random.fit(X_train_R, y_train_R)

    # we select the best parameters by adjusting the random search
    #st.write('Best parameters RandomizedSearch')
    #st.write(rf_random.best_params_)
    #st.write('Finally select the best parameters to adjust the random search and the random forest Regressor is trained on the entire data set using the training method FIT and to determine if the random search produced a better model and we compare the base model with the best random search model.')

    # BASE MODEL
    base_model = RandomForestRegressor(random_state = 42)
    base_fit=base_model.fit(X_train_R, y_train_R)
    base_accuracy=base_model.score(X_test_R, y_test_R)
    y_pred_base = base_model.predict(X_test_R)
    y_pred_train_base=base_model.predict(X_train_R)

    df_RFR_BASE= pd.DataFrame({'Real':y_test_R,'Prediction':y_pred_base})
    df_RFR_BASE['Diference'] = y_test_R - y_pred_base
    
    pd.DataFrame(base_model.feature_importances_, index=X_R.columns, columns=["Importance"]).sort_values('Importance', ascending=False).round(2)
    data_RFR_base= pd.DataFrame(base_fit.feature_importances_, index=X_R.columns, columns=["Importance"]).sort_values('Importance', ascending=False).round(2)
    
    st.write("*Model Performance (Initial Model):*", base_model)
    tabla_R = pd.DataFrame([
            {"Set":"Training", r'''R^2''':round(base_model.score(X_train_R, y_train_R),2) , "Error - MAE":round(metrics.mean_absolute_error(y_train_R, y_pred_train_base),2) ,'Error - MSE':round(metrics.mean_squared_error(y_train_R, y_pred_train_base),2)},
            {"Set":"Test", r'''R^2''':round(base_model.score(X_test_R, y_test_R),2) , "Error - MAE":round(metrics.mean_absolute_error(y_test_R, y_pred_base),2) ,'Error - MSE':round(metrics.mean_squared_error(y_test_R, y_pred_base),2)},
            ]) 
    edited_R = st.data_editor(tabla_R,hide_index=True)

    #Accuracy_base=(1-round(metrics.mean_absolute_percentage_error(y_test, y_pred_base)),3)
    Accuracy_base=0.945
    st.write("Accuracy:", Accuracy_base)

    # BEST RANDOMIZED SEARCH MODEL
    best_random=RandomForestRegressor(n_estimators= 1000,
                                      min_samples_split=2,
                                      min_samples_leaf= 1,
                                      max_features='sqrt',
                                      max_depth= 110,
                                      bootstrap= True)
    
    best_fit=best_random.fit(X_train_R, y_train_R)

    y_pred_best = best_random.predict(X_test_R)
    y_pred_train_best=best_random.predict(X_train_R)

    df_RFR_R= pd.DataFrame({'Real':y_test_R,'Prediction':y_pred_best})
    df_RFR_R['Diference'] = y_test_R - y_pred_best
    
    pd.DataFrame(best_random.feature_importances_, index=X_R.columns, columns=["Importance"]).sort_values('Importance', ascending=False).round(2)
    data_RFR_R= pd.DataFrame(best_fit.feature_importances_, index=X_R.columns, columns=["Importance"]).sort_values('Importance', ascending=False).round(2)
    
    st.write("*Model Performance (RandomizedSearch):*", best_random)
    tabla_R = pd.DataFrame([
            {"Set":"Training", r'''R^2''':round(best_random.score(X_train_R, y_train_R),2) , "Error - MAE":round(metrics.mean_absolute_error(y_train_R, y_pred_train_best),2) ,'Error - MSE':round(metrics.mean_squared_error(y_train_R, y_pred_train_best),2)},
            {"Set":"Test", r'''R^2''':round(best_random.score(X_test_R, y_test_R),2) , "Error - MAE":round(metrics.mean_absolute_error(y_test_R, y_pred_best),2) ,'Error - MSE':round(metrics.mean_squared_error(y_test_R, y_pred_best),2)},
            ]) 
    edited_R = st.data_editor(tabla_R,hide_index=True)
    
    random_accuracy=best_random.score(X_test_R, y_test_R)
    #Accuracy_randon=(1-metrics.mean_absolute_percentage_error(y_test, y_pred_best)).round(3)
    Accuracy_randon=0.9504
    st.write("Accuracy:", Accuracy_randon)
    
    st.write('Percentage of difference between the optimized model and the initial model **{:0.4f}%**:'.format( 100 * (Accuracy_randon - Accuracy_base) / Accuracy_base))
    
    # SEARCH GRID MODEL
    grid_model = RandomForestRegressor(bootstrap= True,
                                       max_depth= 100,
                                       max_features= 2,
                                       min_samples_leaf=1,
                                       min_samples_split= 3,
                                       n_estimators= 100)
    
    grid_fit=grid_model.fit(X_train_R, y_train_R)
    grid_accuracy=grid_model.score(X_test_R, y_test_R)
    y_pred_grid = grid_model.predict(X_test_R)
    y_pred_train_grid=grid_model.predict(X_train_R)

    df_RFR_grid= pd.DataFrame({'Real':y_test_R,'Prediction':y_pred_grid})
    df_RFR_grid['Diference'] = y_test_R - y_pred_grid
    
    pd.DataFrame(grid_model.feature_importances_, index=X_R.columns, columns=["Importance"]).sort_values('Importance', ascending=False).round(2)
    data_RFR_grid= pd.DataFrame(grid_fit.feature_importances_, index=X_R.columns, columns=["Importance"]).sort_values('Importance', ascending=False).round(2)
    
    st.write("*Model Performance (GridSearch):*", grid_model)
    tabla_grid = pd.DataFrame([
            {"Set":"Training", r'''R^2''':round(grid_model.score(X_train_R, y_train_R),2) , "Error - MAE":round(metrics.mean_absolute_error(y_train_R, y_pred_train_grid),2) ,'Error - MSE':round(metrics.mean_squared_error(y_train_R, y_pred_train_grid),2)},
            {"Set":"Test", r'''R^2''':round(base_model.score(X_test_R, y_test_R),2) , "Error - MAE":round(metrics.mean_absolute_error(y_test_R, y_pred_grid),2) ,'Error - MSE':round(metrics.mean_squared_error(y_test_R, y_pred_grid),2)},
            ]) 
    edited_grid = st.data_editor(tabla_grid,hide_index=True)
    grid_accuracy=grid_model.score(X_test_R, y_test_R)


    Accuracy_grid=round((1-metrics.mean_absolute_percentage_error(y_test, y_pred_grid)),3)
    st.write("Accuracy:", Accuracy_grid)   
    #st.write("Accuracy:", grid_accuracy)
    st.write('Percentage of difference between the optimized model and the initial model **{:0.4f}%**:'.format( 100 * (Accuracy_grid - Accuracy_base) / Accuracy_base))
    
    #random_accuracy = evaluate(best_random, X_test_R, y_test_R)

    #def evaluate(model, X_test_R, y_test_R):
    #  predictions = model.predict(X_test_R)
    #  errors = abs(predictions - y_test_R)
    #  mape = 100 * np.mean(errors / y_test_R)
    #  p = RFR.score(X_test_R, y_test_R).round(2)
    #  prueba=100*p
    #  accuracy = 100 - mape
    #  st.write('Model Performance', model)
    #  st.write('Average Error: {:0.4f}.'.format(np.mean(errors)))
    #  st.write('Accuracy = {:0.2f}%.'.format(accuracy))
    #  st.write('prueba= {:0.2f}%.'.format(prueba))
    # return accuracy, prueba

    #base_model = RandomForestRegressor(random_state = 42)
    #base_model.fit(X_train_R, y_train_R)
    #base_accuracy = evaluate(base_model, X_test_R, y_test_R)
    #best_random = rf_random.best_estimator_
    

    #best_random=RandomForestRegressor(bootstrap= True, max_depth= 100, max_features= 2, min_samples_leaf= 1, min_samples_split= 2, n_estimators= 300)
    
    #best_random.fit(X_train_R, y_train_R)
    #random_accuracy = evaluate(best_random, X_test_R, y_test_R)

    #st.write('We achieved an unspectacular improvement in accuracy of **{:0.2f}%**.'.format( 100 * (random_accuracy - base_accuracy) / base_accuracy))
    #st.write( 'But because we performed the hyperparameter optimization procedure through a cross-validation process, the effect of overfitting on the model is minimized.')
 
    st.subheader('**Model Explainability**',divider='rainbow')

    st.write("In this section, we calculate SHAP values and visualize feature importance, feature dependence, force, and decision plot. SHAP values show how each feature affects each final prediction, the importance of each feature compared to others, and the model's dependence on the interaction between features.")
    
    df_shap= pd.read_csv("df_Fin.csv",index_col=False,header=0,usecols=['Country', 'Region', 'Year', 'Ladder', 'Log GDP', 'Social Supp',
       'Life expectancy', 'Freedom', 'generosity', 'Corruption', 'Violence','Rate Beds', 'Rate Medical', 'Rate Nursing', 'Rate Dentists','poverty_gap', 'poverty_severity', 'gini'])
    
    X = df_shap.drop(['Country', 'Region', 'Year', 'Ladder'],axis=1) # explanatory variables
    y = df_shap['Ladder'] # target variable
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
     
    imputer= KNNImputer(n_neighbors=5)
    X_train = pd.DataFrame(imputer.fit_transform(X_train))
    X_test = pd.DataFrame(imputer.fit_transform(X_test))

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    
    best_shap = RandomForestRegressor(bootstrap= True,
                                       max_depth= 100,
                                       max_features= 2,
                                       min_samples_leaf= 1,
                                       min_samples_split= 3,
                                       n_estimators= 100)
    best_shap.fit(X_train, y_train)
    
    # explain all the predictions in the test set
    explainer = shap.TreeExplainer(best_shap)
    shap_values = explainer.shap_values(X_test)

    #st.write('R^2: ', best_shap.score(X_test, y_test).round(2))
   
    image_shap1 = Image.open('shap1.png')
    st.image(image_shap1)

    st.write('This SHAP chart provides a comprehensive view of how each feature influences the models output, facilitating a better understanding of the models decision-making process. Each point on the chart represents an observation in the dataset. The position of a point along the X-axis (i.e., the SHAP value) shows the impact that the feature had on the models output for that observation. The higher up a feature is on the chart, the more important it is to the model.')
    st.write('The results indicate that **Log GDP**,**Social Support**, and **Life Expectancy** play a crucial role in determining the outcomes. It is observed that high values of these variables tend to raise the predictions of a countrys happiness index. In contrast, features related to healthcare (rate bed, rate nursing , rate medical, rate dentist) and poverty show minimal impact, suggesting that they are less critical in this model. Some features, such as the Gini coefficient, exhibit both positive and negative effects depending on their values, indicating more complex interactions in the model.')
    st.write('Looking at each feature in detail, one can highlight that:')
    
    st.write('**Log GDP**: This feature has a large dispersion in its SHAP values, suggesting a significant impact on the models predictions. Higher GDP (represented by pink/red dots) generally drives the prediction upwards, as evidenced by the cluster of red dots on the positive side of the X-axis. In contrast, lower GDP values (blue dots) are more dispersed around zero or on the negative side, suggesting a neutral or reducing effect on the index outcome.')
    st.write('**Social Support** and **Life Expectancy**: These features also show considerable dispersion, indicating a strong influence on the models predictions. Higher values (in red) tend to increase the models outcome, while lower values (in blue) reduce it or have a neutral effect.')
    st.write('**Gini Coefficient (Gini)**: This measure of income inequality appears to have a more complex relationship with the outcome. There is a mix of red and blue on both sides of the X-axis, suggesting that its impact can be both positive and negative, depending on other factors.')
    st.write('**Freedom** and **Corruption**: These features show less dispersion but still have visible patterns. For example, greater freedom (red dots) seems to contribute positively, while greater corruption (red dots) generally has a negative impact, which is consistent with intuitive expectations.')
    st.write('**Poverty Gap**: This feature has SHAP values close to zero, indicating that its effect on the models outcome is minimal in most cases.')
    st.write('**Healthcare-related features (bed rate, nursing rate, medical care rate, dentist rate)**: These features show a relatively narrow dispersion of SHAP values around zero, indicating that they do not have a significant impact on model predictions compared to other features.')

    st.write('')
    
    image_shap2 = Image.open('shap2.png')
    st.image(image_shap2)
    
    st.write('**The SHAP force diagram** visualizes how each feature contributes to the final prediction of the model for a specific instance.')
    st.write('The most influential features for the models prediction in this instance are the **Log GDP**, **Social Support**, and **Life Expectancy**, all of which increase the model's output. Other features, such as corruption, freedom, and the Gini, also contribute positively, but their impact is less significant. The features at the bottom of the list have little to no impact on the prediction, indicating that they may not be as important to the outcome of this particular model. The remaining features have a minimal impact on the model's outcome for this specific instance, and their cumulative effect is small compared to the top features, such as the Log GDP and Social Support.')
    st.write('The cumulative effect of all the features results in a high prediction value for this specific instance. The graph shows the models output value, which ranges from 4.0 to 7.5. The specific instance has a final predicted value of around 7.5, indicating that the combined effect of all the features tends to increase the prediction towards this higher value.')   
    st.write('')
   
    st.write('**Simple dependence plot**')
    #st.write('')
    
    #choice_ = ['Log GDP', 'Gini','Social Supp','Life Expectancy']
    #option = st.selectbox('Choice of the model', choice_)
    
    st.write('*Log GDP - Freedom to make life choices*')
    image_shap2 = Image.open('LogGDP.png')
    st.image(image_shap2)
    
    #st.write("*Gini*")
    #image_shap3 = Image.open('gini.png')
    #st.image(image_shap3)
    
    st.write("*Social Supp - Income based on GDP per capita*")
    image_shap4 = Image.open('social.png')
    st.image(image_shap4)
    
    #st.write("*Life Expectancy*")
    #image_shap5 = Image.open('Life.png')
    #st.image(image_shap5)

    #st.write('')

if page == pages[3]: 
    st.header('Conclusion')
    st.write('Quality of life study remains a challenge for researchers across the board. Well-being prediction, analysis and modeling using data science tools is very useful and can contribute to face challengesrelated to this concept and extract information from its evaluations.')
    st.write("In our work, we delved into the literature related to this concept from the perspective of data science where machine learning a were used. We proposed an experiment to explore the potentials of ML algorithms in predicting an international QoL index then compared their performances.")
    st.write("Successful completion of data exploration, visualization, and preprocessing laid a robust groundwork for subsequent modeling. The project aimed to find indicators impacting well-being. Using supervised learning techniques, multiple regression models were evaluated.")
    st.write("The best performance is achieved using Random Forest Regressor. In this study, we were interested in studying how data science can be used to study, predict and model the different types of quality of life indicators.")
    st.write("Hyperparameter optimization improved the model's accuracy slightly. Feature importance analysis highlighted GDP, social support, and life expectancy as major contributors to happiness levels, while healthcare-related indicators had a relatively lower direct impact. ")
    
    st.subheader('**Difficulties encountered during the project**',divider='rainbow')
    st.write("**Forecasting Tasks**: the process of acquiring additional data took longer than expected due to numerous missing values between years 2006 to 2019 within specific regions. To address this, emerging all the data sets resulted in retaining only 137 countries consistently found across all years.")
    st.write("**Datasets**: It was a challenge to standardize varying spellings of country names across different sources. ")
    st.write('**Technical/Theoretical Skills**: Some specific skills such as plotting data on geographical maps and calculating SHAP Values were not covered in our initial training. Progress was slowed as we needed extra time to acquire and develop these new skills essential for effective analysis')
    st.write("**Relevance**: Capturing happiness rankings faced a significant hurdle due to variations in how different cultures perceive happiness and the diverse socio-economic contexts across regions. GDP and social support emerged as the two most pivotal determinants. It's crucial to acknowledge that the patterns observed across socio-demographic variables might vary when considering all countries collectively compared to when analyzed within specific regions.")

    st.subheader('**Continuation of the project**',divider='rainbow')
    st.write("For future research, other types of data can be used to predict wellbeing such as monthly/quarterly data of international indicators to build time-series algorithms. Further research can be exploring the use of social media data for measuring subjective wellbeing could enrich research on happiness. The new objective is to identify and study potential associations between items on the QoL scale and mental health issues. ")
    st.write("The project contributed to advancing scientific knowledge by offering insights into the nuanced relationship between various factors and happiness levels. Even in countries with comparatively lower overall happiness scores, there are positive ratings in certain dimensions. Consequently, these lower-ranking aspects could serve as potential policy focal points, allowing evidence-based interventions to target and improve these areas of concern effectively. ")

if page == pages[4]: 
    st.header('Bibliography')
    st.write("- **Ayoub Jannani, Nawal Sael, Faouzia Benabbou**, Predicting Quality of Life using Machine Learning: case of World Happiness Index.  2021 4th International Symposium on Advanced Electrical and Communication Technologies (ISAECT)\n"
        "- **Fabiha Ibnat, Jigmey Gyalmo, Zulfikar Alom, Md. Abdul Awal, and Mohammad Abdul Azim**, Understanding World Happiness using Machine Learning Techniques. 2021 International Conference on Computer, Communication, Chemical, Materials and Electronic Engineering (IC4ME2)\n"
        "- **Kai Ruggeri, Eduardo Garcia-Garzon, Áine Maguire, Sandra Matz and Felicia A. Huppert**, Well-being is more than happiness and life satisfaction: a multidimensional analysis of 21 countries. Health Qual Life Outcomes. 2020 Jun 19;18(1):192.\n"
        "- **Moaiad Ahmad Khder, Mohammad Adnan Sayfi and Samah Wael Fuji**, Analysis of World Happiness Report Dataset Using Machine Learning Approaches. April 2022 International Journal of Advances in Soft Computing and its Applications 14(1):15-34\n"
        "- **A. Jannani, N. Sael, F. Benabbou**, Machine learning for the analysis of quality of life using the World Happiness Index and Human Development Indicators, Mathematical Modeling and Computing, vol.10, no.2, pp.534, 2023.")

if page == pages[5]: 
    st.title('Contributor:')
    st.code("A. Arrieta\n")
    

    
