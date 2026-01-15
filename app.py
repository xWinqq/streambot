import streamlit as st
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
import fitz
import os

# --- 1. Pagina Configuratie ---
st.set_page_config(page_title="OERbot - Dulon College", page_icon="📚", layout="centered")

# --- 2. Geavanceerde CSS ---
def apply_custom_css():
    st.markdown(f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Nunito:wght@300;400;700&display=swap');
        
        /* Sidebar styling */
        [data-testid="stSidebar"] {{ background-color: #f0f2f6 !important; }}
        
        [data-testid="stAppViewContainer"], .main, [data-testid="stHeader"] {{
            background-color: white !important;
            color: #1f1f1f !important;
        }}
        p, h1, h2, h3, h4, label, .stMarkdown {{
            color: #1f1f1f !important;
            font-family: 'Nunito', sans-serif !important;
        }}
        .stButton>button {{
            background-color: white !important;
            color: #e5241d !important;
            border-radius: 10px !important;
            border: 1.5px solid #e5241d !important;
            font-weight: 600 !important;
            width: 100% !important;
        }}
        .stButton>button:hover {{ background-color: #e5241d !important; color: white !important; }}
        </style>
    """, unsafe_allow_html=True)

apply_custom_css()

# --- 3. API & AI Model ---
api_key = st.secrets.get("openai_api_key")
admin_user = st.secrets.get("admin_username")
admin_pass = st.secrets.get("admin_password")
# Note: Using gpt-4o-mini as gpt-5-nano does not exist yet (as of early 2024/2025)
llm = ChatOpenAI(model="gpt-4o-mini", api_key=api_key, temperature=0.5)

# --- 4. Persistent Storage Setup ---
UPLOAD_DIR = "uploads"
CHROMA_DIR = "chroma_db"
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

@st.cache_resource
def initialize_vector_store(pdf_paths):
    try:
        if not pdf_paths:
            return None
        embeddings = OpenAIEmbeddings(api_key=api_key)
        all_documents = []
        
        for path in pdf_paths:
            doc = fitz.open(path)
            file_name = os.path.basename(path)
            for page in doc:
                text = page.get_text().strip()
                if text:
                    all_documents.append(Document(
                        page_content=text, 
                        metadata={"page": page.number + 1, "source": file_name}
                    ))
        
        return Chroma.from_documents(
            all_documents, 
            embeddings, 
            persist_directory=os.path.join(os.getcwd(), CHROMA_DIR, "shared_pdf")
        )
    except Exception as e:
        st.error(f"Fout bij verwerken documenten: {e}")
        return None

# --- 5. Session State beheer ---
if 'messages' not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hoi! Ik ben OERbot 😊. Ik heb alle reglementen doorgelezen. Waar kan ik je vandaag mee helpen?"}]
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'show_disclaimer' not in st.session_state:
    st.session_state.show_disclaimer = False

# --- 6. Document Loading (Ensures files are kept) ---
existing_pdfs = [os.path.join(UPLOAD_DIR, f) for f in os.listdir(UPLOAD_DIR) if f.endswith(".pdf")]
if existing_pdfs:
    st.session_state.vector_store = initialize_vector_store(existing_pdfs)
else:
    st.session_state.vector_store = None

# --- 7. Branding ---
col1, col2, col3 = st.columns([1,3,1])
with col2:
    if os.path.exists("logo.png"): st.image("logo.png", use_container_width=True)
    else: st.title("🤖 OERbot")
st.markdown("<p style='text-align: center; opacity: 0.8; font-size: 0.9em;'>Jouw hulp voor vragen over de OER op het Dulon College.</p>", unsafe_allow_html=True)

# --- 8. Chat Logica ---
def handle_query(query):
    st.session_state.messages.append({"role": "user", "content": query})
    if st.session_state.vector_store is None:
        st.session_state.messages.append({"role": "assistant", "content": "Ik heb nog geen documenten om te lezen. Vraag de beheerder om PDF's te uploaden! 👍"})
    else:
        results = st.session_state.vector_store.similarity_search_with_score(query, k=4)
        docs = [r[0] for r in results if r[1] < 0.6]

        if not docs:
            response = "Ik kan je hier helaas alleen helpen met informatie uit de OER. Deze vraag staat niet in de OER, dus kan ik je hier niets over zeggen. 😊"
        else:
            context_text = "\n\n".join([f"Bron [{d.metadata['source']}]: {d.page_content}" for d in docs])
            system_prompt = f"Jij bent OERbot... [STRICT OER RULES] ... CONTEXT: {context_text}"
            chat_template = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{question}")])
            formatted = chat_template.format_messages(question=query)
            response = llm.invoke(formatted).content
        
        st.session_state.messages.append({"role": "assistant", "content": response})

# --- 9. UI Components ---
st.markdown("#### Waar wil je meer over weten?")
q_col1, q_col2 = st.columns(2, gap="small")
with q_col1:
    if st.button("🔄 Herkansingen", use_container_width=True): handle_query("Hoe werkt een herkansing?"); st.rerun()
    if st.button("🤒 Ziek bij examen", use_container_width=True): handle_query("Wat als ik ziek ben voor een examen?"); st.rerun()
with q_col2:
    if st.button("🚫 Fraude regels", use_container_width=True): handle_query("Wat gebeurt er bij fraude?"); st.rerun()
    if st.button("👨‍🏫 Persoonlijke hulp", use_container_width=True): handle_query("Wie helpt bij persoonlijke omstandigheden?"); st.rerun()

st.divider()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if chat_input := st.chat_input("Stel je vraag aan OERbot..."):
    handle_query(chat_input); st.rerun()

# --- 10. Sidebar Admin (The "X" functionality) ---
with st.sidebar:
    if not st.session_state.logged_in:
        st.title("Admin")
        u = st.text_input("User")
        p = st.text_input("Pass", type="password")
        if st.button("Login"):
            if u == admin_user and p == admin_pass:
                st.session_state.logged_in = True
                st.rerun()
    else:
        # Hier is de "X" knop om het beheer menu te sluiten
        col_title, col_close = st.columns([4,1])
        with col_title:
            st.title("Beheer")
        with col_close:
            if st.button("✖️"): 
                st.session_state.logged_in = False
                st.rerun()
        
        uploaded_files = st.file_uploader("Upload OER PDF's", type="pdf", accept_multiple_files=True)
        if uploaded_files:
            for uploaded_file in uploaded_files:
                pdf_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
                with open(pdf_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
            st.success("Bestanden opgeslagen!")
            st.rerun() # Refresh to trigger vector store initialization

    st.sidebar.markdown("---")
    if st.sidebar.button("📜 Algemene Voorwaarden"): 
        st.session_state.show_disclaimer = not st.session_state.show_disclaimer
    if st.session_state.show_disclaimer: 
        st.sidebar.info("Disclaimer: Aan antwoorden kunnen geen rechten worden ontleend.")