import pandas as pd
import json
import streamlit as st

def render_upload():
    st.header("📂 Upload Dataset")

    uploaded_file = st.file_uploader(
        "Upload your dataset",
        type=["csv", "json"],
        help="Supported formats: CSV and JSON"
    )

    if uploaded_file is None:
        return None

    extension = uploaded_file.name.split(".")[-1].lower()

    try:
        if extension == "csv":
            df = pd.read_csv(uploaded_file)
        elif extension == "json":

            data = json.load(uploaded_file)

            # Case 1:
            # [
            #   {...},
            #   {...}
            # ]
            if isinstance(data, list):
                df = pd.DataFrame(data)

            # Case 2:
            # {"data":[...]}
            elif isinstance(data, dict) and "data" in data:
                df = pd.DataFrame(data["data"])

            # Case 3:
            # {"records":[...]}
            elif isinstance(data, dict) and "records" in data:

                df = pd.DataFrame(data["records"])

            # Case 4:
            # Nested JSON
            else:

                df = pd.json_normalize(data)

        else:

            st.error("Unsupported file format.")
            st.stop()


        # -----------------------------
        # Empty Dataset Check
        # -----------------------------
        if df.empty:

            st.error("Uploaded dataset is empty.")
            st.stop()

        # -----------------------------
        # Save in Session
        # -----------------------------
        st.session_state.df = df

        st.success(
            f"Dataset loaded successfully! "
            f"({len(df)} rows × {len(df.columns)} columns)"
        )

    except pd.errors.EmptyDataError:

        st.error("The uploaded file is empty.")

        st.stop()

    except json.JSONDecodeError:

        st.error("Invalid JSON file.")

        st.stop()

    except UnicodeDecodeError:

        st.error(
            "Unable to decode the uploaded file. "
            "Please upload a UTF-8 encoded file."
        )

        st.stop()

    except Exception as e:

        st.error(f"Error while loading dataset:\n{e}")

        st.stop()

    return df