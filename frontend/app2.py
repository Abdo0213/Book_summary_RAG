import streamlit as st
import requests
import os
# ======================
# Config
# ======================
BACKEND_URL = "http://localhost:8000"  # adjust to your backend
UPLOAD_DIR = r"D:\Courses\NLP_NTI\Final project\uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ======================
# Backend Helpers
# ======================
def load_file(save_path):
    payload = {"path": save_path.replace("\\", "/"), "flag": False}
    return requests.post(f"{BACKEND_URL}/books/build", json=payload)

def summary_en():
    response = requests.get(f"{BACKEND_URL}/books/summarize")
    if response.status_code == 200:
        data = response.json()
        st.session_state["summary_en"] = data.get("summary", "")
        st.success("✅ English Summary Generated!")
        return st.session_state["summary_en"]
    else:
        st.error("❌ Error from backend (summarize)")

def summary_ar(content):
    payload = {"content": content}
    response = requests.post(f"{BACKEND_URL}/books/translate", json=payload)
    if response.status_code == 200:
        data = response.json()
        st.session_state["summary_ar"] = data.get("translated_text", "")
        st.success("✅ Arabic Summary Generated!")
        return st.session_state["summary_ar"]
    else:
        st.error("❌ Error from backend (translate)")

def download_pdf(lang="en"):
    filename = "summary_en.pdf" if lang == "en" else "summary_ar.pdf"
    response = requests.get(f"{BACKEND_URL}/books/download?filename={filename}")
    if response.status_code == 200:
        return response.content
    else:
        st.error("❌ Error downloading PDF")
        return None

def chat_with_bot(query: str):
    payload = {"question": query}
    response = requests.post(f"{BACKEND_URL}/qa/ask", json=payload)
    if response.status_code == 200:
        return response.json().get("answer", "⚠️ No answer found")
    else:
        return "❌ Error contacting chatbot backend"

# ======================
# Authentication Pages
# ======================

def login_page():
    col1, col2, col3 = st.columns([1,2,1])

    with col2:  # Middle column
        st.markdown("<h2 style='text-align: center;'>🔐 Login</h2>", unsafe_allow_html=True)

        username = st.text_input("📧 Username")
        password = st.text_input("🔑 Password", type="password")

        if st.button("Login", use_container_width=True):
            res = requests.post(f"{BACKEND_URL}/auth/login", json={
                "username": username,
                "password": password
            })
            if res.status_code == 200:
                st.session_state.user_id = res.json()["user_id"]
                st.session_state.categories = res.json()["categories"]
                st.success("✅ Login successful!")
                st.session_state.page = "upload"
                st.rerun()
            else:
                st.error(res.json().get("detail", "Login failed"))

        st.markdown("---")

        if st.button("🏠 Back to Home", use_container_width=True):
            st.session_state.page = "home"
            st.rerun()

def register_page():
    col1, col2, col3 = st.columns([1,2,1])

    with col2:
        st.markdown("<h2 style='text-align: center;'>📝 Register</h2>", unsafe_allow_html=True)

        username = st.text_input("👤 Username")
        email = st.text_input("📧 Email")
        password = st.text_input("🔑 Password", type="password")
        categories = st.text_input("Categories")

        if st.button("Register", use_container_width=True):
            res = requests.post(f"{BACKEND_URL}/auth/register", json={
                "username": username,
                "email": email,
                "password": password,
                "categories": categories
            })
            if res.status_code == 200:
                st.success("🎉 Registered successfully! Please login.")
                st.session_state.page = "login"
                st.rerun()
            else:
                st.error(res.json().get("detail", "Registration failed"))

        st.markdown("---")

        if st.button("🏠 Back to Home", use_container_width=True):
            st.session_state.page = "home"
            st.rerun()
# ======================
# Layout
# ======================
st.set_page_config(
    page_title="Book Summarization",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown(""" 
<style>
    .main-container {
        max-width: 1200px;
        margin: 0 auto;
        padding: 2rem;
    }
    .title {
        color: #2a5db0;
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    .description {
        color: #444;
        font-size: 1.3rem;
        line-height: 1.6;
        margin-bottom: 2rem;
    }
    .custom-button {
        background: linear-gradient(135deg, #4a90e2 0%, #2a5db0 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 12px 24px;
        margin: 8px 0;
        font-size: 16px;
        font-weight: 600;
        cursor: pointer;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
        width: 100%;
        text-align: center;
        display: block;
        text-decoration: none;
    }
    .custom-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }
    .custom-button.register {
        background: linear-gradient(135deg, #ff9800 0%, #f57c00 100%);
    }
    .upload-button {
        background: linear-gradient(135deg, #6a11cb 0%, #2575fc 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 14px 28px;
        font-size: 18px;
        font-weight: 600;
        cursor: pointer;
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
        transition: all 0.3s ease;
        margin-top: 2rem;
        width: 100%;
    }
    .upload-button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 16px rgba(0,0,0,0.2);
    }
    .divider {
        height: 1px;
        background: linear-gradient(to right, transparent, #ddd, transparent);
        margin: 1.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ======================
# Page Navigation
# ======================
if "page" not in st.session_state:
    st.session_state.page = "home"

if st.session_state.page == "home":
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown(
            """
            <div style="text-align: left; padding: 20px;">
                <h1 class="title">📚 Book Summarization</h1>
                <p class="description" style="color: white; font-size: 18px">
                    Upload any book or document and get a concise, AI-powered summary instantly.  
                    Save time, focus on what matters most, and never miss key insights again.  
                    <ul style="list-style: none;">
                        <li>✅ Summarize long textbooks into clear, digestible notes. </li>
                        <li>✅ Highlight key insights from research papers and articles.  </li>
                        <li>✅ Boost productivity by reducing hours of reading into minutes. </li>
                        <li>✅ Access your summaries anytime, anywhere.  </li>
                    </ul>
                    Turn information overload into clarity — let AI do the heavy lifting so you can learn smarter, not harder.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown('<div class="right-column">', unsafe_allow_html=True)
        st.markdown('<h2 style="color: #2a5db0; font-weight:bold ;margin-bottom: 1.5rem;">Get Started</h2>', unsafe_allow_html=True)

        col_log = st.columns([1])[0]  # single full-width column
        with col_log:
            if st.button("🔐 Login", key="login_btn", use_container_width=True):
                st.session_state.page = "login"
                st.rerun()

        col_reg = st.columns([1])[0]  # single full-width column
        with col_reg:
            if st.button("📝 Register", key="register_btn", use_container_width=True):
                st.session_state.page = "register"
                st.rerun()


        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown('<p style="color: #666; text-align: center;">Or continue as a guest</p>', unsafe_allow_html=True)

        col_reg = st.columns([1])[0]  # single full-width column
        with col_reg:
            if st.button("🚀 Upload as Guest", key="guest_upload", use_container_width=True):
                st.session_state.page = "upload"
                st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

elif st.session_state.page == "login":
    login_page()

elif st.session_state.page == "register":
    register_page()

elif st.session_state.page == "upload":
    # Back to Home button
    if st.button("🏠 Back to Home"):
        st.session_state.page = "home"
        st.rerun()
    st.set_page_config(layout="wide")
    col1, col2 = st.columns([1.5, 1]) 
    # ---------------- LEFT: Summarization ----------------
    with col1:
        st.title("📚 Book Summarization")

        uploaded_file = st.file_uploader("Choose a file", type=["pdf", "docx", "txt"])

        if uploaded_file:
            st.success("✅ File uploaded successfully!")

            save_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            if st.button("Summarize"):
                response = load_file(save_path)
                if response.status_code == 200:
                    english_summary = summary_en()
                    
            if "summary_en" in st.session_state:
                st.markdown("### 📄 English Summary")
                st.write(st.session_state["summary_en"])

                if st.button("Translate"):
                    arabic_summary = summary_ar(st.session_state["summary_en"])

            if "summary_ar" in st.session_state:
                st.markdown("### 📄 Arabic Summary")
                st.write(st.session_state["summary_ar"])

            st.markdown("### 💾 Download Summaries")
            col1d, col2d = st.columns(2)
            if "summary_en" in st.session_state:
                with col1d:
                    if st.button("Generate English PDF"):
                        st.session_state["pdf_en"] = download_pdf(lang="en")

                    if "pdf_en" in st.session_state:
                        st.download_button(
                            "⬇️ Download English PDF",
                            st.session_state["pdf_en"],
                            "summary_en.pdf",
                            mime="application/pdf"
                        )

            # Arabic PDF
            if "summary_ar" in st.session_state:
                with col2d:
                    if st.button("Generate Arabic PDF"):
                        st.session_state["pdf_ar"] = download_pdf(lang="ar")

                    if "pdf_ar" in st.session_state:
                        st.download_button(
                            "⬇️ Download Arabic PDF",
                            st.session_state["pdf_ar"],
                            "summary_ar.pdf",
                            mime="application/pdf"
                        )

    # ---------------- RIGHT: Chatbot ----------------
    
    with col2:
        if st.session_state.user_id :
            col13, col23, col33 = st.columns([1,2,1])
            with col33:
                if st.button("🚪 Logout", use_container_width=True):
                    st.session_state.user_id = None
                    st.session_state.categories = None
                    st.session_state.page = "home"
                    st.rerun()
        if uploaded_file:
            if "messages" not in st.session_state:
                st.session_state.messages = []
            if "processing" not in st.session_state:
                st.session_state.processing = False
            
            st.markdown(""" 
                <style>
                    body {
                        background: linear-gradient(135deg, #121212 0%, #1a1a2e 100%);
                        color: white;
                    }
                    
                    .message {
                        display: flex;  
                        gap: 10px;  
                        align-items: center; 
                        margin-bottom: 15px;
                    }  
                    
                    .user-message-container {
                        display: flex; 
                        justify-content: flex-end; 
                        width: 100%; 
                    }
                    
                    .user-bubble {  
                        display: flex;
                        background: linear-gradient(135deg, #4a90e2 0%, #3a7bd5 100%); 
                        color: white;  
                        padding: 12px 16px;  
                        border-radius: 18px 18px 0 18px;  
                        max-width: 75%;  
                        margin: 5px 0;  
                        order: 1;
                        box-shadow: 0 2px 8px rgba(74, 144, 226, 0.3);
                        transition: all 0.3s ease;
                    }  
                    
                    .bot-bubble {
                        display: block;
                        background: linear-gradient(135deg, #f5f5dc 0%, #e8e8d8 100%);
                        color: #333;
                        padding: 12px 16px;
                        border-radius: 18px 18px 18px 0;
                        max-width: 75%;
                        margin: 8px 0;
                        box-shadow: 0 2px 8px rgba(245, 245, 220, 0.3);
                        align-self: flex-start;
                        word-wrap: break-word;
                    }
                    
                    .user-bubble:hover {
                        transform: translateY(-2px);
                        box-shadow: 0 4px 12px rgba(74, 144, 226, 0.4);
                    }
                    
                    .bot-bubble:hover {
                        transform: translateY(-2px);
                        box-shadow: 0 4px 12px rgba(245, 245, 220, 0.4);
                    }
                    
                    /* Chat container */
                    .chat-container {  
                        display: flex;  
                        flex-direction: column;  
                        gap: 5px;
                    }  
                    
                    .botIcon {  
                        background: linear-gradient(135deg, #ffab40 0%, #ff8f00 100%);
                        color: #121212; 
                        border-radius: 50%;  
                        width: 50px; 
                        height: 50px; 
                        font-size: 28px; 
                        text-align: center;  
                        line-height: 50px; 
                        flex-shrink: 0;
                        box-shadow: 0 2px 6px rgba(255, 171, 64, 0.4);
                    } 
                    
                    .userIcon {  
                        background: linear-gradient(135deg, #3d7dd8 0%, #2a5db0 100%);
                        color: white; 
                        border-radius: 50%;  
                        width: 50px; 
                        height: 50px; 
                        font-size: 28px; 
                        text-align: center;  
                        line-height: 50px; 
                        flex-shrink: 0; 
                        order: 2;
                        box-shadow: 0 2px 6px rgba(61, 125, 216, 0.4);
                    }
                    
                    /* Spinner animation */
                    .spinner {
                        display: inline-block;
                        animation: spin 1s linear infinite;
                    }
                    
                    @keyframes spin {
                        0% { transform: rotate(0deg); }
                        100% { transform: rotate(360deg); }
                    }
                </style>  
            """, unsafe_allow_html=True) 

            st.title("🤖 Chat with Bot") 

            # Display all messages including any in-progress ones
            for i, msg in enumerate(st.session_state.messages):
                if msg["role"] == "user": 
                    st.markdown(f""" 
                    <div class='chat-container'> 
                        <div class='user-message-container'> 
                            <div class='message'> 
                                <div class='user-bubble'>{msg['content']}</div> 
                                <div class='userIcon'>👤</div> 
                            </div> 
                        </div> 
                    </div> 
                    """, unsafe_allow_html=True) 
                else: 
                    st.markdown(f""" 
                    <div class='chat-container'> 
                        <div class='message'> 
                            <div class='botIcon'>🤖</div> 
                            <div class='bot-bubble'>{msg['content']}</div> 
                        </div> 
                    </div> 
                    """, unsafe_allow_html=True)

            # Show thinking animation if processing
            if st.session_state.processing:
                st.markdown(f""" 
                <div class='chat-container'> 
                    <div class='message'> 
                        <div class='botIcon'>🤖</div> 
                        <div class='bot-bubble'>
                            <div style="display: flex; align-items: center; gap: 10px;">
                                <div class="spinner">🤔</div>
                                <span>Thinking...</span>
                            </div>
                        </div> 
                    </div> 
                </div> 
                """, unsafe_allow_html=True)

            user_input = st.chat_input("Type your message here...") 
            if user_input and not st.session_state.processing:
                # Add user message immediately
                st.session_state.messages.append({"role": "user", "content": user_input})
                if len(st.session_state.messages) > 10:
                    st.session_state.messages = st.session_state.messages[-10:]
                st.session_state.processing = True
                st.rerun()

            # This runs after the rerun when processing is True
            if st.session_state.processing and st.session_state.messages[-1]["role"] == "user":
                # Get the last user message
                user_message = st.session_state.messages[-1]["content"]
                bot_response = chat_with_bot(user_message)
                st.session_state.messages.append({"role": "bot", "content": bot_response})
                if len(st.session_state.messages) > 10:
                    st.session_state.messages = st.session_state.messages[-10:]
                st.session_state.processing = False
                st.rerun() 

