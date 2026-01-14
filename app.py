import streamlit as st
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
import fitz
import os
import shutil

# 1. Pagina Configuratie
st.set_page_config(page_title="OERbot - Dulon College", page_icon="📚", layout="centered")

# 2. CSS (Onveranderd)
def apply_custom_css():
    st.markdown(f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Nunito:wght@300;400;700&display=swap');
        [data-testid="collapsedControl"] {{ display: none; }}
        [data-testid="stAppViewContainer"], .main, [data-testid="stHeader"] {{
            background-color: white !important; color: #1f1f1f !important;
        }}
        [data-testid="stSidebar"] {{ background-color: #f0f2f6 !important; }}
        p, h1, h2, h3, h4, span, label, .stMarkdown {{
            color: #1f1f1f !important; font-family: 'Nunito', sans-serif !important;
        }}
        .stButton>button {{
            background-color: white !important; color: #e5241d !important;
            border-radius: 10px !important; border: 1.5px solid #e5241d !important;
            font-weight: 600 !important; transition: all 0.2s ease;
        }}
        .stButton>button:hover {{ background-color: #e5241d !important; color: white !important; }}
        [data-testid="stChatMessage"] {{
            background-color: #f8f9fa !important; border: 1px solid #eee !important; border-radius: 15px !important;
        }}
        </style>
    """, unsafe_allow_html=True)

apply_custom_css()

# 3. API & Configuratie
api_key = st.secrets.get("openai_api_key")
admin_user = st.secrets.get("admin_username")
admin_pass = st.secrets.get("admin_password")
# Let op: gpt-5-nano bestaat nog niet officieel, ik laat het staan als je een custom endpoint gebruikt, 
# anders is 'gpt-4o' of 'gpt-3.5-turbo' de standaard.
llm = ChatOpenAI(model="gpt-4o", api_key=api_key, temperature=0.5)

# 4. Vector Store Initialisatie voor Meerdere PDF's
@st.cache_resource
def process_all_pdfs():
    upload_dir = "uploads"
    if not os.path.exists(upload_dir):
        return None
    
    pdf_files = [f for f in os.listdir(upload_dir) if f.endswith(".pdf")]
    if not pdf_files:
        return None

    all_documents = []
    embeddings = OpenAIEmbeddings(api_key=api_key)
    
    for pdf_file in pdf_files:
        path = os.path.join(upload_dir, pdf_file)
        try:
            doc = fitz.open(path)
            for page in doc:
                text = page.get_text().strip()
                if text:
                    all_documents.append(Document(
                        page_content=text, 
                        metadata={"source": pdf_file, "page": page.number + 1}
                    ))
        except Exception as e:
            st.error(f"Fout bij lezen van {pdf_file}: {e}")
    
    if not all_documents:
        return None
        
    # We bouwen de database opnieuw op met alle documenten uit de map
    return Chroma.from_documents(
        all_documents, 
        embeddings, 
        persist_directory=os.path.join(os.getcwd(), "chroma_db", "shared_pdf")
    )

# 5. Session State
if 'messages' not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hoi! Ik ben OERbot 😊. Heb je een vraag over het examenreglement of de OER?"}]
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = process_all_pdfs() # Probeer bestaande pdfs te laden bij start
if 'show_disclaimer' not in st.session_state:
    st.session_state.show_disclaimer = False

# 6. Branding
col1, col2, col3 = st.columns([1,3,1])
with col2:
    if os.path.exists("logo.png"):
        st.image("logo.png", use_container_width=True)
    else:
        st.title("🤖 OERbot")

# 7. Centrale Chat Logica
def handle_query(query):
    st.session_state.messages.append({"role": "user", "content": query})
    
    if st.session_state.vector_store is None:
        st.session_state.messages.append({"role": "assistant", "content": "Ik kan de OER-regels nog niet inzien. Vraag de beheerder om PDF's te uploaden! 👍"})
    else:
        results = st.session_state.vector_store.similarity_search_with_score(query, k=4)
        # Score filter: lager is beter bij Chroma (afstand)
        docs = [r[0] for r in results if r[1] < 0.6]

        system_prompt = f"""
        Jij bent OERbot, een vriendelijke assistent van het Dulon College.
        Help studenten met vragen op basis van de verstrekte context uit de OER documenten.
        
        STIJL:
        - Wees warm en behulpzaam. Gebruik B1-taal en 'je/jij'.
        - Beantwoord GEEN vragen buiten de OER.
        
        BRONVERMELDING REGELS:
        - Vermeld de bron (bijv. 📖 Artikel X lid Y) op een NIEUWE REGEL onderaan je antwoord.
        - BELANGRIJK: Vermeld de bron ALLEEN als je de exacte artikel- of hoofdstuknaam in de context kunt vinden.
        - Als de context geen specifiek artikelnummer noemt, laat de regel met 📖 dan HELEMAAL WEG. Fantaseer geen artikelnummers.

        ANTWOORD STRUCTUUR:
        1. Korte erkenning.
        2. Helder antwoord in B1-taal.
        3. Bronvermelding (ALLEEN indien bekend in de tekst, anders overslaan).
        4. Succes-afsluiting.

        CONTEXT:
        {" ".join([d.page_content for d in docs]) if docs else "GEEN INFORMATIE BESCHIKBAAR."}
        """
        
        if not docs:
            response = "Ik kan hier in de OER helaas geen informatie over vinden. Je kunt dit het beste even navragen bij je slb'er of de examencommissie. 😊"
        else:
            chat_template = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{question}")])
            formatted = chat_template.format_messages(question=query)
            full_response = ""
            # Gebruik hier de stream van de LLM
            response_container = llm.invoke(formatted)
            response = response_container.content
            
        st.session_state.messages.append({"role": "assistant", "content": response})

# 8. Quick Actions (Onveranderd)
st.markdown("#### Waar wil je meer over weten?")
q_col1, q_col2 = st.columns(2, gap="small")
with q_col1:
    if st.button("🔄 Herkansingen", use_container_width=True):
        handle_query("Hoe werkt een herkansing?")
        st.rerun()
    if st.button("🤒 Ziek bij examen", use_container_width=True):
        handle_query("Wat moet ik doen als ik ziek ben voor een examen?")
        st.rerun()
with q_col2:
    if st.button("🚫 Fraude regels", use_container_width=True):
        handle_query("Wat gebeurt er bij fraude?")
        st.rerun()
    if st.button("⚖️ Klacht indienen", use_container_width=True):
        handle_query("Hoe dien ik een klacht in?")
        st.rerun()

st.divider()

# 9. Chat Geschiedenis
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 10. Chat Input
if chat_input := st.chat_input("Stel je eigen vraag aan OERbot..."):
    handle_query(chat_input)
    st.rerun()

# 11. Zijbalk (Beheer voor Meerdere PDF's)
with st.sidebar:
    if not st.session_state.logged_in:
        st.title("Admin")
        u = st.text_input("Gebruikersnaam")
        p = st.text_input("Wachtwoord", type="password")
        if st.button("Inloggen"):
            if u == admin_user and p == admin_pass:
                st.session_state.logged_in = True
                st.rerun()
    else:
        st.title("Beheer")
        uploaded_files = st.file_uploader("Upload OER PDF's", type="pdf", accept_multiple_files=True)
        if st.button("Bestanden Verwerken"):
            if uploaded_files:
                os.makedirs("uploads", exist_ok=True)
                for uploaded_file in uploaded_files:
                    pdf_path = os.path.join("uploads", uploaded_file.name)
                    with open(pdf_path, "wb") as f: 
                        f.write(uploaded_file.getbuffer())
                
                # Forceer herladen van de database
                st.cache_resource.clear()
                st.session_state.vector_store = process_all_pdfs()
                st.success(f"{len(uploaded_files)} PDF('s) succesvol verwerkt!")
            else:
                st.warning("Selecteer eerst bestanden.")
        
        if st.button("Verwijder alle PDF's"):
            if os.path.exists("uploads"):
                shutil.rmtree("uploads")
                st.cache_resource.clear()
                st.session_state.vector_store = None
                st.rerun()

        if st.button("Uitloggen"):
            st.session_state.logged_in = False
            st.rerun()

    st.sidebar.markdown("---")
    if st.sidebar.button("Algemene Voorwaarden"):
        st.session_state.show_disclaimer = not st.session_state.show_disclaimer
    if st.session_state.show_disclaimer:
        st.sidebar.info("Disclaimer tekst...")