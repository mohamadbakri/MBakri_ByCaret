import streamlit as st
import pandas as pd
import io


@st.cache_data  # 👈 Add the caching decorator
def load_data(url):
    df = pd.read_csv(url)
    return df


st.set_page_config(layout="wide", page_icon='./logo.png', page_title='EDA')
all_vizuals = ['Detect columns types', 'Detect null values']


def write_to_st(string, font_size='100'):
    st.write('<p style="font-size:{}%">&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp{}</p>'.format(
        font_size, string), unsafe_allow_html=True)


def write_to_st_sidebar(string, font_size='100'):
    st.sidebar.write('<p style="font-size:{}%">{}</p>'.format(
        font_size, string), unsafe_allow_html=True)


def df_info(df):
    # Replace spaces in columns name with _
    df.columns = df.columns.str.replace(' ', '_')
    # The StringIO module is an in-memory file-like object
    # buffer is where to return the outpu
    buffer = io.StringIO()
    df.info(buf=buffer)
    s = buffer.getvalue()

    df_info = s.split('\n')

    counts = []
    names = []
    nn_count = []
    dtype = []
    for i in range(5, len(df_info)-3):
        line = df_info[i].split()
        counts.append(line[0])
        names.append(line[1])
        nn_count.append(line[2])
        dtype.append(line[4])

    df_info_dataframe = pd.DataFrame(
        data={'#': counts, 'Column': names, 'Non-Null Count': nn_count, 'Data Type': dtype})
    return df_info_dataframe.drop('#', axis=1)


def df_isnull(df):
    res = pd.DataFrame(df.isnull().sum()).reset_index()
    res['Percentage'] = round(res[0] / df.shape[0] * 100, 2)
    res['Percentage'] = res['Percentage'].astype(str) + '%'
    return res.rename(columns={'index': 'Column', 0: 'Number of null values'})


def Convert_string_to_categorical(df):
    # Convert all columns with string values into category values
    for label, content in df.items():
        if pd.api.types.is_string_dtype(content) or pd.api.types.is_object_dtype(content):
            df[label] = content.astype('category').cat.as_ordered()
    return df

# 3- after detecting null values and features types  you can ask user what techniques he want to apply in the columns , ask him like  what do you want to do with categorical ( most frequent or just put additional class for missing value ) and ask him again for  continuous   ( mean or median or mode )


def Check_for_null_values_and_fill(df, fill_with):
    # Check for which numeric columns have null values.
    # Fill numeric null columns with mean or median or mode.
    for label, content in df.items():
        if pd.api.types.is_numeric_dtype(content):
            if pd.isnull(content).sum():
                if fill_with == "Mean":
                    # Fill missing numeric values with mean.
                    df[label] = content.fillna(content.mean())
                if fill_with == "Median":
                    # Fill missing numeric values with median.
                    df[label] = content.fillna(content.median())
                if fill_with == "Mode":
                    # Fill missing numeric values with mode.
                    df[label] = content.fillna(content.mode()[0])


# def prepare_date_for_fitting(df):
def turn_categorical_to_numbers_and_fill_missing(df):
   # Turn categorical variables into numbers and fill missing
    for label, content in df.items():
        if not pd.api.types.is_numeric_dtype(content):
            # Turn categories into numbers and add +1
            df[label] = pd.Categorical(content).codes+1
    return df


def predict(df, chosen_target):
    if df[chosen_target].dtype == "bool" or df[chosen_target].dtype == 'category':
        from pycaret.classification import setup, compare_models, pull
        st.write("Classification Case:")
        setup(df, target=chosen_target)
        # setup(df, target=chosen_target, silent=True)
        setup_df = pull()
        st.dataframe(setup_df, width=2000)
        best_model = compare_models()
        compare_df = pull()
        st.dataframe(compare_df, width=2000)

    if df[chosen_target].dtype != "bool" or df[chosen_target].dtype != 'category':
        from pycaret.regression import setup, compare_models, pull
        st.write("Regression Case:")
        setup(df, target=chosen_target)
        # setup(df, target=chosen_target, silent=True)
        setup_df = pull()
        st.dataframe(setup_df, width=2000)
        best_model = compare_models()
        compare_df = pull()
        st.dataframe(compare_df, width=2000)


# if os.path.exists('./dataset.csv'):
#     df = pd.read_csv('dataset.csv', index_col=None)
with st.sidebar:
    # st.image("https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png")
    st.image(
        "https://t3.ftcdn.net/jpg/03/34/91/12/360_F_334911218_UijkFoVixOVSYlPLVMdsrBDTNvDkgEj1.jpg")
    # st.image("https://electropi.ai/logo_dark.webp")
    st.title("AutoML")
    # choice = st.radio(
    #     "Navigation", ["Upload", "Detect columns types , null values", "Modelling", "Download"])
    st.info("This application helps you explore and build your data.")


st.title("Upload Your Dataset")
file = st.file_uploader("Upload Your Dataset")
if file:

    vizuals = st.sidebar.multiselect(
        "Detect columns types , null values 👇", all_vizuals)
    df = load_data(file)
    # df = pd.read_csv(file, index_col=None)
    # df.to_csv('dataset.csv', index=None)
    st.dataframe(df, width=1500)

    if 'Detect columns types' in vizuals:
        st.subheader('Detect columns types:')
        st.dataframe(df_info(df), width=1500)

    if 'Detect null values' in vizuals:
        st.subheader('NA Value Information:')
        if df.isnull().sum().sum() == 0:
            st.write('There is not any NA value in your dataset.')
        else:
            st.dataframe(df_isnull(df), width=1500)
            # functions.space(2)

    # ask user to decide what columns to drop.
    cols_to_drop = st.sidebar.multiselect(
        "Select columns to drop:?", df.columns)
    drop = st.sidebar.button('Drop Columns')
    if "drop_state" not in st.session_state:
        st.session_state.drop_state = False

    if drop or st.session_state.drop_state:
        st.session_state.drop_state = True
        df.drop(cols_to_drop, axis=1, inplace=True)
        if drop and cols_to_drop:
            st.subheader('Dataframe after dropping [{}] columns:'.format(
                ",".join(cols_to_drop)))
            st.dataframe(df, width=1500)

    # Replace continuous( mean or median or mode )

    # mean_median_mode = st.sidebar.selectbox(
    #     "Replace missing values with:", ("Select Replacement Type", "Mean", "Median", "Mode"))

    # replace = st.sidebar.button('Replace with:')
    # if "replace_state" not in st.session_state:
    #     st.session_state.replace_state = False
    # if replace or st.session_state.replace_state:
    #     st.session_state.replace_state = True
    #     Check_for_null_values_and_fill(df,  mean_median_mode)
    #     if replace and (mean_median_mode != "Select Replacement Type"):
    #         write_to_st(
    #             f"Dataframe after fill missing values with {mean_median_mode}", "140")
    #         st.dataframe(df, width=1500)

    if "convert_string_categoricals_state" not in st.session_state:
        st.session_state.convert_string_categoricals_state = False
    convert_string_categoricals = st.sidebar.checkbox(
        'Convert Strings to Categorical')
    if convert_string_categoricals or st.session_state.convert_string_categoricals_state:
        st.session_state.convert_string_categoricals_state = True
        Convert_string_to_categorical(df)
        if convert_string_categoricals:
            write_to_st(
                "Dataframe after convert string values to categorical.", "140")
            st.dataframe(df, width=1500)
            # st.dataframe(df_info(df), width=1500)

    if "turn_categorical_numbers_and_fill_missing_state" not in st.session_state:
        st.session_state.turn_categorical_numbers_and_fill_missing_state = False

    turn_categorical_numbers_and_fill_missing = st.sidebar.checkbox(
        'Turn categorical to numbers and fill missing values with additional class.')
    if turn_categorical_numbers_and_fill_missing or st.session_state.turn_categorical_numbers_and_fill_missing_state:
        st.session_state.turn_categorical_numbers_and_fill_missing_state = True
        turn_categorical_to_numbers_and_fill_missing(df)
        if turn_categorical_numbers_and_fill_missing:
            write_to_st(
                "Dataframe after turnning categorical to numbers.", "140")
            st.dataframe(df, width=1500)
            # st.dataframe(df_info(df), width=1500)

    mean_median_mode = st.sidebar.selectbox(
        "Replace missing values with:", ("Select Replacement Type", "Mean", "Median", "Mode"))

    replace = st.sidebar.button('Replace with:')
    if "replace_state" not in st.session_state:
        st.session_state.replace_state = False
    if replace or st.session_state.replace_state:
        st.session_state.replace_state = True
        Check_for_null_values_and_fill(df,  mean_median_mode)
        if replace and (mean_median_mode != "Select Replacement Type"):
            write_to_st(
                f"Dataframe after fill missing values with {mean_median_mode}", "140")
            st.dataframe(df, width=1500)

    chosen_target = st.sidebar.selectbox(
        "Select column to Predict:?", df.columns)

    predct = st.sidebar.button('Predict')
    if "predct_state" not in st.session_state:
        st.session_state.predct_state = False

    if predict or st.session_state.predct_state:
        st.session_state.predct_state = True
        predict(df, chosen_target)
