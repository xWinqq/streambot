import streamlit as st
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
import fitz
import os
import shutil

# 1. Pagina Configuratie & Styling
st.set_page_config(page_title="OERbot - Dulon College", page_icon="📚", layout="centered")

def apply_custom_css():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Nunito:wght@300;400;700&display=swap');
        [data-testid="collapsedControl"] { display: none; }
        .stButton>button {
            background-color: white !important; color: #e5241d !important;
            border-radius: 10px !important; border: 1.5px solid #e5241d !important;
            font-weight: 600 !important; width: 100%;
        }
        .stButton>button:hover { background-color: #e5241d !important; color: white !important; }
        [data-testid="stChatMessage"] { border-radius: 15px !important; }
        </style>
    """, unsafe_allow_html=True)

apply_custom_css()

# 2. API & Admin Setup
api_key = st.secrets.get("openai_api_key")
admin_user = st.secrets.get("admin_username")
admin_pass = st.secrets.get("admin_password")
llm = ChatOpenAI(model="gpt-4o-mini", api_key=api_key, temperature=0)

# 3. Vector Store Functies (Met opslag op schijf)
DB_PATH = os.path.join(os.getcwd(), "chroma_db_v2")
UPLOAD_DIR = "uploads"

@st.cache_resource
def load_vector_store():
    if os.path.exists(DB_PATH) and os.listdir(DB_PATH):
        return Chroma(persist_directory=DB_PATH, embedding_function=OpenAIEmbeddings(api_key=api_key))
    return None

def process_and_save_pdfs(uploaded_files):
    if not os.path.exists(UPLOAD_DIR):
        os.makedirs(UPLOAD_DIR)
    
    documents = []
    for uploaded_file in uploaded_files:
        file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        doc = fitz.open(file_path)
        for page in doc:
            text = page.get_text().strip()
            if text:
                documents.append(Document(
                    page_content=text, 
                    metadata={"source": uploaded_file.name, "page": page.number + 1}
                ))
    
    if documents:
        if os.path.exists(DB_PATH):
            shutil.rmtree(DB_PATH)
        vector_store = Chroma.from_documents(
            documents, 
            OpenAIEmbeddings(api_key=api_key), 
            persist_directory=DB_PATH
        )
        return vector_store
    return None

# 4. Session State
if 'messages' not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hoi! Ik ben OERbot 😊. Stel je vraag over het examenreglement."}]
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = load_vector_store()

# 5. Centrale Chat Logica
def handle_query(query):
    st.session_state.messages.append({"role": "user", "content": query})
    
    if st.session_state.vector_store is None:
        response = "Ik heb nog geen documenten ingeladen. Vraag de beheerder om de PDF's te uploaden."
    else:
        # Zoek relevante tekst
        docs = st.session_state.vector_store.similarity_search(query, k=5)
        context_text = "\n\n".join([d.page_content for d in docs])

        # DE STRENGE PROMPT
        system_prompt = f"""
        Jij bent OERbot, een assistent voor het Dulon College. 
        Je mag uitsluitend antwoorden op basis van de verstrekte CONTEXT.

        STRICTE REGELS:
        1. Als het antwoord NIET letterlijk of indirect in de context staat, zeg dan: "Ik kan je hier helaas alleen helpen met informatie uit de OER. Deze vraag staat niet in de OER, dus kan ik je hier niets over zeggen. 😊"
        2. Beantwoord GEEN algemene vragen (geen recepten, geen computertips, geen algemene kennis).
        3. Gebruik B1-taal en spreek de student aan met 'je'.
        4. BRON: Als de context een artikelnummer noemt (bijv. Artikel 5), zet dit dan onderaan op een nieuwe regel met 📖. Noem GEEN bron als er geen artikelnummer in de tekst staat.

        CONTEXT:
        {context_text}
        """

        chat_template = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{question}")])
        ai_response = llm.invoke(chat_template.format_messages(question=query))
        response = ai_response.content
            
    st.session_state.messages.append({"role": "assistant", "content": response})

# 6. UI
st.title("🤖 OERbot")

# Chat weergave
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input
if prompt := st.chat_input("Stel je vraag..."):
    handle_query(prompt)
    st.rerun()

# 7. Sidebar Beheer
with st.sidebar:
    if not st.session_state.logged_in:
        st.subheader("Admin Login")
        u = st.text_input("Gebruiker")
        p = st.text_input("Wachtwoord", type="password")
        if st.button("Log in"):
            if u == admin_user and p == admin_pass:
                st.session_state.logged_in = True
                st.rerun()
    else:
        st.success("Beheerder modus")
        files = st.file_uploader("Upload OER PDF's", type="pdf", accept_multiple_files=True)
        if st.button("Verwerk PDF's"):
            if files:
                with st.spinner("Verwerken..."):
                    st.session_state.vector_store = process_and_save_pdfs(files)
                    st.success("Klaar! De OER is bijgewerkt.")
                    st.rerun()
        
        if st.button("Verwijder alles"):
            if os.path.exists(DB_PATH): shutil.rmtree(DB_PATH)
            if os.path.exists(UPLOAD_DIR): shutil.rmtree(UPLOAD_DIR)
            st.cache_resource.clear()
            st.session_state.vector_store = None
            st.rerun()

        if st.button("Log uit"):
            st.session_state.logged_in = False
            st.rerun()