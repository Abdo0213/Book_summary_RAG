import streamlit as st
import requests

BASE_URL = "http://127.0.0.1:8000"
if "user_id" not in st.session_state:
    st.session_state.user_id = None

def login_page():
    st.title("🔐 Login / Register")

    choice = st.radio("Select option", ["Login", "Register"])
    email = st.text_input("Email")
    username = st.text_input("Username") if choice == "Register" else ""
    password = st.text_input("Password", type="password")

    if choice == "Register":
        if st.button("Register"):
            res = requests.post(f"{BASE_URL}/register", json={
                "username": username,
                "email": email,
                "password": password
            })
            if res.status_code == 200:
                st.success("Registered! Now login.")
            else:
                st.error(res.json()["detail"])
    else:
        if st.button("Login"):
            res = requests.post(f"{BASE_URL}/login", json={
                "username": "x",  # dummy (FastAPI ignores username on login)
                "email": email,
                "password": password
            })
            if res.status_code == 200:
                st.session_state.user_id = res.json()["user_id"]
                st.rerun()
            else:
                st.error(res.json()["detail"])

def home_page():
    st.sidebar.title("📚 Menu")
    if st.sidebar.button("🚪 Logout"):
        st.session_state.user_id = None
        st.rerun()

    page = st.sidebar.radio("Go to", ["Upload Book", "Chatbot", "Favorite Books"])

    if page == "Upload Book":
        st.header("📤 Upload Book")
        uploaded = st.file_uploader("Choose a book", type=["pdf", "epub", "txt"])
        if uploaded and st.button("Summarize"):
            with st.spinner("⏳ Processing book and generating summaries..."):
                res = requests.post(
                    f"{BASE_URL}/upload/",
                    files={"file": (uploaded.name, uploaded.getvalue())}
                )
                if res.status_code == 200:
                    st.success("✅ Summaries are ready!")

                    # روابط التحميل من الـ backend
                    st.download_button(
                        "⬇️ Download English Summary",
                        data=requests.get(f"{BASE_URL}/download/english").content,
                        file_name="summary_en.txt"
                    )
                    st.download_button(
                        "⬇️ Download Arabic Summary",
                        data=requests.get(f"{BASE_URL}/download/arabic").content,
                        file_name="summary_ar.txt"
                    )
                else:
                    st.error("❌ Failed to process file.")

    elif page == "Chatbot":
        st.header("📖 Chatbot")
        st.write("Chatbot coming soon...")

    elif page == "Favorite Books":
        st.header("⭐ Favorite Books")
        st.write("List of user’s favorite books will appear here...")

if st.session_state.user_id is None:
    login_page()
else:
    home_page()
