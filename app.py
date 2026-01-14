import streamlit as st
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
import fitz  # PyMuPDF
import os
import shutil

# --- 1. CONFIGURATIE & STYLING ---
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

# --- 2. SETUP VARIABELEN ---
api_key = st.secrets.get("openai_api_key")
admin_user = st.secrets.get("admin_username")
admin_pass = st.secrets.get("admin_password")

# Gebruik 'gpt-4o-mini' voor de beste balans tussen snelheid en kosten
llm = ChatOpenAI(model="gpt-4o-mini", api_key=api_key, temperature=0)

# --- 3. FUNCTIES VOOR PDF VERWERKING ---
def process_pdfs(uploaded_files):
    """Verwerkt geüploade bestanden en zet ze in de vector store."""
    documents = []
    for uploaded_file in uploaded_files:
        # Lees PDF direct uit geheugen
        doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        for page in doc:
            text = page.get_text().strip()
            if text:
                documents.append(Document(
                    page_content=text, 
                    metadata={"source": uploaded_file.name, "page": page.number + 1}
                ))
    
    if documents:
        embeddings = OpenAIEmbeddings(api_key=api_key)
        # We maken een nieuwe store aan in het geheugen
        return Chroma.from_documents(documents, embeddings)
    return None

# --- 4. SESSION STATE ---
if 'messages' not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hoi! Ik ben OERbot 😊. Stel je vraag over de OER of het examenreglement."}]
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None

# --- 5. CHAT LOGICA ---
def handle_query(query):
    st.session_state.messages.append({"role": "user", "content": query})
    
    if st.session_state.vector_store is None:
        response = "Ik kan de OER nog niet inzien. Vraag de beheerder om de PDF's te uploaden! 📂"
    else:
        # Zoek relevante stukken tekst
        docs = st.session_state.vector_store.similarity_search(query, k=4)
        context_text = "\n\n".join([f"Bron: {d.metadata['source']}\n{d.page_content}" for d in docs])

        system_prompt = f"""
        Jij bent OERbot, een behulpzame assistent van het Dulon College.
        Beantwoord vragen van studenten uitsluitend op basis van de onderstaande tekst.

        STIJLREGELS:
        - Gebruik B1-taal (eenvoudig Nederlands).
        - Spreek de student aan met 'je' en 'jij'.
        - Wees kort en krachtig.

        BRONVERMELDING (CRITISCH):
        - Als de tekst expliciet een artikelnummer of lid noemt (bijv. Artikel 4.2), schrijf dan op een NIEUWE REGEL onderaan: 📖 [Artikelnaam/Nummer].
        - Als er GEEN artikelnummer in de tekst staat, laat de regel met de 📖 dan volledig weg. Fantaseer NOOIT een artikelnummer.

        CONTEXT:
        {context_text}
        """

        chat_template = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{question}")])
        ai_response = llm.invoke(chat_template.format_messages(question=query))
        response = ai_response.content
            
    st.session_state.messages.append({"role": "assistant", "content": response})

# --- 6. UI LAYOUT ---
st.title("🤖 OERbot")

# Quick Actions
cols = st.columns(2)
with cols[0]:
    if st.button("🔄 Herkansingen"): handle_query("Hoe werken herkansingen?"); st.rerun()
with cols[1]:
    if st.button("🤒 Ziekte"): handle_query("Wat als ik ziek ben tijdens een examen?"); st.rerun()

st.divider()

# Chat display
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Stel je vraag..."):
    handle_query(prompt)
    st.rerun()

# --- 7. SIDEBAR (BEHEER) ---
with st.sidebar:
    if not st.session_state.logged_in:
        st.subheader("Beheerder Login")
        u = st.text_input("Gebruiker")
        p = st.text_input("Wachtwoord", type="password")
        if st.button("Log in"):
            if u == admin_user and p == admin_pass:
                st.session_state.logged_in = True
                st.rerun()
    else:
        st.success("Ingelogd als beheerder")
        uploaded_files = st.file_uploader("Upload OER PDF's", type="pdf", accept_multiple_files=True)
        if st.button("Verwerk bestanden"):
            if uploaded_files:
                with st.spinner("Bezig met verwerken..."):
                    st.session_state.vector_store = process_pdfs(uploaded_files)
                    st.success(f"{len(uploaded_files)} bestanden geladen!")
            else:
                st.warning("Upload eerst PDF's.")
        
        if st.button("Log uit"):
            st.session_state.logged_in = False
            st.rerun()