import streamlit as st
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
import fitz  # PyMuPDF
import os

# --- 1. CONFIGURATIE ---
st.set_page_config(
    page_title="OERbot - Dulon College", 
    page_icon="📚", 
    layout="centered",
    initial_sidebar_state="collapsed"
)

# --- 2. CSS STYLING ---
def apply_custom_css():
    st.markdown(f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Nunito:wght@300;400;700&display=swap');
        
        [data-testid="stAppViewContainer"], .main, [data-testid="stHeader"] {{
            background-color: white !important;
            color: #1f1f1f !important;
        }}
        [data-testid="stSidebar"] {{ background-color: #f0f2f6 !important; }}
        
        p, h1, h2, h3, h4, label, .stMarkdown {{
            color: #1f1f1f !important;
            font-family: 'Nunito', sans-serif !important;
        }}
        
        .stButton>button {{
            background-color: white !important;
            color: #e5241d !important;
            border-radius: 10px !important;
            border: 1.5px solid #e5241d !important;
            padding: 5px 15px !important;
            font-weight: 600 !important;
            width: 100% !important;
        }}
        .stButton>button:hover {{ background-color: #e5241d !important; color: white !important; }}
        [data-testid="stChatMessage"] {{ background-color: #f8f9fa !important; border: 1px solid #eee !important; border-radius: 15px !important; }}
        </style>
    """, unsafe_allow_html=True)

apply_custom_css()

# --- 3. API SETUP ---
api_key = st.secrets.get("openai_api_key")
admin_user = st.secrets.get("admin_username")
admin_pass = st.secrets.get("admin_password")

if not api_key:
    st.error("Geen OpenAI API key gevonden in secrets!")
    st.stop()

llm = ChatOpenAI(model="gpt-4o-mini", api_key=api_key, temperature=0.5)

# --- 4. VECTOR STORE LOGICA ---
@st.cache_resource(show_spinner=False)
def initialize_vector_store(pdf_paths):
    """Leest PDF's in en maakt de zoekmachine aan."""
    try:
        embeddings = OpenAIEmbeddings(api_key=api_key)
        all_documents = []
        
        for path in pdf_paths:
            if not os.path.exists(path): continue
            
            doc = fitz.open(path)
            file_name = os.path.basename(path)
            file_has_text = False
            
            for page in doc:
                text = page.get_text().strip()
                if text:
                    file_has_text = True
                    all_documents.append(Document(
                        page_content=text, 
                        metadata={"page": page.number + 1, "source": file_name}
                    ))
            
            # Check of het bestand misschien een scan is (geen tekst gevonden)
            if not file_has_text:
                print(f"⚠️ WAARSCHUWING: Bestand '{file_name}' bevat geen leesbare tekst (mogelijk een scan).")

        if not all_documents:
            return None

        # Chroma DB aanmaken
        persist_dir = os.path.join(os.getcwd(), "chroma_db")
        return Chroma.from_documents(
            all_documents, 
            embeddings, 
            persist_directory=persist_dir
        )
    except Exception as e:
        st.error(f"Critical error in vector store: {e}")
        return None

# --- 5. INITIALISATIE & SESSION STATE ---
if 'messages' not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hoi! Ik ben de examenbot 😊. Ik heb de reglementen gelezen. Waar kan ik je mee helpen?"}]

if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

# Auto-load logic: Check direct de map 'uploads' bij opstarten
if 'vector_store' not in st.session_state:
    upload_dir = "uploads"
    if not os.path.exists(upload_dir):
        os.makedirs(upload_dir)
        
    existing_pdfs = [os.path.join(upload_dir, f) for f in os.listdir(upload_dir) if f.endswith(".pdf")]
    
    if existing_pdfs:
        with st.spinner("Kennisbank laden..."):
            st.session_state.vector_store = initialize_vector_store(existing_pdfs)
    else:
        st.session_state.vector_store = None

# --- 6. USER INTERFACE (MAIN) ---
col1, col2, col3 = st.columns([1,3,1])
with col2:
    if os.path.exists("logo.png"): st.image("logo.png", use_container_width=True)
    else: st.title("🤖 OERbot")
st.markdown("<p style='text-align: center; opacity: 0.8; font-size: 0.9em;'>Jouw hulp voor vragen over de onderwijs- en examenreglementen.</p>", unsafe_allow_html=True)

# --- 7. CHAT LOGICA ---
def handle_query(query):
    st.session_state.messages.append({"role": "user", "content": query})
    
    if st.session_state.vector_store is None:
        st.session_state.messages.append({"role": "assistant", "content": "⚠️ Ik heb nog geen documenten geladen. Vraag de beheerder om PDF's te uploaden."})
    else:
        with st.spinner("Zoeken in documenten... 📚"):
            # Zoek de 4 meest relevante stukjes tekst
            results = st.session_state.vector_store.similarity_search_with_score(query, k=4)
            # Filter resultaten die te weinig lijken (score > 0.6 is vaak ruis in L2 distance)
            docs = [r[0] for r in results] # We pakken ze gewoon allemaal, filtering kan te streng zijn

            if not docs:
                response = "Ik kan het antwoord helaas niet vinden in de geüploade documenten."
            else:
                context_text = "\n\n".join([f"Bron [{d.metadata['source']}, pagina {d.metadata['page']}]: {d.page_content}" for d in docs])
                
                system_prompt = f"""
                Jij bent OERbot, een vriendelijke assistent.
                
                INSTRUCTIES:
                - Gebruik ALLEEN de onderstaande context om antwoord te geven.
                - Als het antwoord niet in de tekst staat, zeg dat dan eerlijk.
                - Vermeld altijd de bron (Bestandsnaam + Pagina).
                
                CONTEXT:
                {context_text}
                """
                
                chat_template = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{question}")])
                formatted = chat_template.format_messages(question=query)
                
                full_response = "".join([chunk.content for chunk in llm.stream(formatted)])
                response = full_response
            
            st.session_state.messages.append({"role": "assistant", "content": response})

# --- 8. QUICK ACTIONS ---
st.markdown("#### Veelgestelde vragen:")
q_col1, q_col2 = st.columns(2, gap="small")
with q_col1:
    if st.button("🔄 Herkansingen", use_container_width=True): handle_query("Hoe werkt een herkansing?"); st.rerun()
    if st.button("🤒 Ziek bij examen", use_container_width=True): handle_query("Wat als ik ziek ben voor een examen?"); st.rerun()
with q_col2:
    if st.button("🚫 Fraude regels", use_container_width=True): handle_query("Wat gebeurt er bij fraude?"); st.rerun()
    if st.button("👨‍🏫 Persoonlijke hulp", use_container_width=True): handle_query("Wie helpt bij persoonlijke omstandigheden?"); st.rerun()

st.divider()

# --- 9. DISPLAY CHAT ---
for message in st.session_state.messages:
    role = message["role"]
    avatar = "🤖" if role == "assistant" else "👤"
    with st.chat_message(role, avatar=avatar):
        st.markdown(message["content"])

if chat_input := st.chat_input("Typ hier je vraag..."):
    handle_query(chat_input)
    st.rerun()

# --- 10. ADMIN SIDEBAR (VERBETERD) ---
with st.sidebar:
    if not st.session_state.logged_in:
        st.header("Admin Login")
        u = st.text_input("Gebruikersnaam")
        p = st.text_input("Wachtwoord", type="password")
        if st.button("Inloggen"):
            if u == admin_user and p == admin_pass: 
                st.session_state.logged_in = True
                st.rerun()
            else:
                st.error("Foute gegevens")
    else:
        st.header("📂 Document Beheer")
        
        # Laat zien wat er nu in de map staat
        upload_dir = "uploads"
        if not os.path.exists(upload_dir): os.makedirs(upload_dir)
        files_on_disk = os.listdir(upload_dir)
        
        st.markdown(f"**Huidige bestanden ({len(files_on_disk)}):**")
        if files_on_disk:
            for f in files_on_disk:
                st.caption(f"- {f}")
        else:
            st.caption("_Geen bestanden gevonden_")

        st.markdown("---")
        uploaded_files = st.file_uploader("Nieuwe PDF's toevoegen", type="pdf", accept_multiple_files=True)
        
        if uploaded_files:
            # 1. Bestanden opslaan
            for uploaded_file in uploaded_files:
                save_path = os.path.join(upload_dir, uploaded_file.name)
                with open(save_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
            
            # 2. ALLES opnieuw inlezen (oude + nieuwe)
            all_paths = [os.path.join(upload_dir, f) for f in os.listdir(upload_dir) if f.endswith(".pdf")]
            
            with st.spinner("Database herbouwen..."):
                # Forceer herladen door cache te clearen (optioneel, maar veilig)
                initialize_vector_store.clear()
                st.session_state.vector_store = initialize_vector_store(all_paths)
                
            st.success("✅ Database succesvol bijgewerkt!")
            st.rerun()

        if st.button("Uitloggen"):
            st.session_state.logged_in = False
            st.rerun()