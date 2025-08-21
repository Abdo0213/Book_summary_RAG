import streamlit as st
import requests
import os

BACKEND_URL = "http://localhost:8000"  # adjust to your backend
UPLOAD_DIR = r"D:\Courses\NLP_NTI\Final project\uploads"

def load_file(save_path):
    payload = {
        "path": save_path.replace("\\", "/"),
        "flag": False
    }
    response = requests.post(f"{BACKEND_URL}/books/build", json=payload)
    return response

def summary_en():
    summary = requests.get(f"{BACKEND_URL}/books/summarize")
    if summary.status_code == 200:
        summary = summary.json()   # <-- convert Response to dict
        st.session_state["summary_en"] = summary.get("summary", "")
        st.success("Summary generated!")
        return summary.get("summary", "")
    else:
        st.error("Error from backend")

def summary_ar(content):
    translate_payload = {
        "content": content
    }
    ar_summary_obj = requests.post(f"{BACKEND_URL}/books/translate", json=translate_payload)
    if ar_summary_obj.status_code == 200:
        ar_summary_obj = ar_summary_obj.json()
        st.session_state["summary_ar"] = ar_summary_obj.get("translated_text", "")
        return ar_summary_obj.get("translated_text", "")
    else:
        st.error("Error from backend")


if "page" not in st.session_state:
    st.session_state.page = "home"

if st.session_state.page == "home":
    st.title("📚 Book Summarization")
    if st.button("Upload now"):
        st.session_state.page = "upload"
        st.rerun()

elif st.session_state.page == "upload":
    st.title("📂 Upload your Document")

    uploaded_file = st.file_uploader("Choose a file", type=["pdf", "docx", "txt"])

    if uploaded_file:
        st.success("File uploaded successfully!")

        # Save uploaded file locally so backend can read it
        save_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        if st.button("Summarize"):
            response = load_file(save_path)
            if response.status_code == 200:
                english_summary = summary_en()
            else:
                st.error("Error from backend")
        if st.button("Translate"):
            arabic_summary = summary_ar(st.session_state["summary_en"])
