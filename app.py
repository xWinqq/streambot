import streamlit as st
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
import fitz
import os

# 1. Pagina Configuratie
st.set_page_config(page_title="OERbot - Dulon College", page_icon="📚", layout="centered")

# 2. Geavanceerde CSS (Light Mode, Mobile Fixes, No Link Icons & Sleek Disclaimer)
def apply_custom_css():
    st.markdown(f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Nunito:wght@300;400;700&display=swap');
        
        [data-testid="collapsedControl"], .stMarkdown a.header-anchor {{
            display: none !important;
        }}

        [data-testid="stAppViewContainer"], .main, [data-testid="stHeader"] {{
            background-color: white !important;
            color: #1f1f1f !important;
        }}
        [data-testid="stSidebar"] {{
            background-color: #f0f2f6 !important;
        }}
        p, h1, h2, h3, h4, span, label, .stMarkdown {{
            color: #1f1f1f !important;
            font-family: 'Nunito', sans-serif !important;
        }}

        @media (max-width: 640px) {{
            [data-testid="column"] {{
                width: 100% !important;
                flex: 1 1 100% !important;
                min-width: 100% !important;
            }}
        }}

        .stButton>button {{
            background-color: white !important;
            color: #e5241d !important;
            border-radius: 10px !important;
            border: 1.5px solid #e5241d !important;
            padding: 5px 15px !important;
            height: 2.8em !important;
            font-weight: 600 !important;
            width: 100% !important;
            margin-bottom: 8px !important;
            transition: all 0.2s ease;
        }}
        .stButton>button:hover {{
            background-color: #e5241d !important;
            color: white !important;
        }}

        .disclaimer-btn>div>button {{
            background-color: transparent !important;
            color: gray !important;
            border: none !important;
            font-size: 0.8em !important;
            height: auto !important;
            padding: 0 !important;
            text-decoration: underline !important;
            font-weight: normal !important;
        }}

        [data-testid="stChatMessage"] {{
            background-color: #f8f9fa !important;
            border: 1px solid #eee !important;
            border-radius: 15px !important;
        }}
        </style>
    """, unsafe_allow_html=True)

apply_custom_css()

# 3. API & AI Model (GPT-5 Nano voor snelheid en lage kosten)
api_key = st.secrets.get("openai_api_key")
admin_user = st.secrets.get("admin_username")
admin_pass = st.secrets.get("admin_password")

# GPT-5 Nano is geoptimaliseerd voor instructies en snelheid
llm = ChatOpenAI(model="gpt-5-nano", api_key=api_key, temperature=0.5)

# 4. Vector Store Functie (Ondersteunt nu meerdere documenten)
@st.cache_resource
def initialize_vector_store(file_paths):
    try:
        embeddings = OpenAIEmbeddings(api_key=api_key)
        all_documents = []
        
        for path in file_paths:
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
            persist_directory=os.path.join(os.getcwd(), "chroma_db", "oer_store")
        )
    except:
        return None

# 5. Session States
if 'messages' not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hoi! Ik ben OERbot 😊. Ik heb de reglementen gelezen en sta klaar om je te helpen. Waar heb je een vraag over?"}]
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'show_disclaimer' not in st.session_state:
    st.session_state.show_disclaimer = False

# 6. Logo & Branding
col1, col2, col3 = st.columns([1,3,1])
with col2:
    if os.path.exists("logo.png"):
        st.image("logo.png", use_container_width=True)
    else:
        st.title("🤖 OERbot")
st.markdown("<p style='text-align: center; opacity: 0.8; font-size: 0.9em;'>Jouw klasgenoot voor vragen over de OER op het Dulon College.</p>", unsafe_allow_html=True)

# 7. Centrale Chat Logica
def handle_query(query):
    st.session_state.messages.append({"role": "user", "content": query})
    
    if st.session_state.vector_store is None:
        st.session_state.messages.append({"role": "assistant", "content": "Ik help je graag, maar ik heb nog geen documenten om te raadplegen. Vraag de beheerder om de PDF's te uploaden! 👍"})
    else:
        results = st.session_state.vector_store.similarity_search_with_score(query, k=4)
        docs = [r[0] for r in results if r[1] < 0.6]

        system_prompt = f"""
        Jij bent OERbot, een vriendelijke klasgenoot op het Dulon College.
        Help studenten met vragen uit de verstrekte OER documenten.
        
        STIJL & PERSONA:
        - Wees warm en behulpzaam. Varieer je begroetingen.
        - Gebruik B1-taal en 'je/jij'.
        - Beantwoord GEEN vragen buiten de OER.
        - Verzin niets en blijf altijd OERbot.
        
        BRONVERMELDING:
        - Vermeld ALTIJD de bron (bijv. Artikel X lid Y uit [Bestandsnaam]) op een NIEUWE REGEL onderaan je antwoord.
        - Gebruik de emoji 📖.

        ANTWOORD STRUCTUUR:
        1. Korte erkenning van de vraag.
        2. Samenvatting van de regel in begrijpelijke taal.
        3. Bronvermelding op nieuwe regel met 📖.
        4. Duidelijke Call to Action.
        5. Positieve afsluiting.

        CONTEXT: {" ".join([f"[{d.metadata['source']}]: {d.page_content}" for d in docs]) if docs else "GEEN INFO."}
        """
        
        if not docs:
            response = "Ik kan je hier helaas alleen helpen met informatie uit de officiële reglementen. Deze vraag staat er niet in. 😊"
        else:
            chat_template = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{question}")])
            formatted = chat_template.format_messages(question=query)
            full_response = ""
            for chunk in llm.stream(formatted):
                full_response += chunk.content
            response = full_response
            
        st.session_state.messages.append({"role": "assistant", "content": response})

# 8. Quick Actions
st.markdown("#### Waar wil je meer over weten?")
q_col1, q_col2 = st.columns(2, gap="small")
with q_col1:
    if st.button("🔄 Herkansingen"):
        handle_query("Hoe werkt een herkansing?"); st.rerun()
    if st.button("🤒 Ziek bij examen"):
        handle_query("Wat moet ik doen als ik ziek ben voor een examen?"); st.rerun()
    if st.button("📝 Vrijstellingen"):
        handle_query("Hoe kan ik vrijstelling aanvragen?"); st.rerun()
with q_col2:
    if st.button("🚫 Fraude regels"):
        handle_query("Wat gebeurt er bij fraude?"); st.rerun()
    if st.button("⚖️ Klacht indienen"):
        handle_query("Hoe dien ik een klacht in?"); st.rerun()
    if st.button("👨‍🏫 Persoonlijke hulp"):
        handle_query("Wie helpt mij bij persoonlijke omstandigheden?"); st.rerun()

st.divider()

# 9. Chat Geschiedenis
bot_icon = "custom_bot_image.png" if os.path.exists("custom_bot_image.png") else "🤖"
user_icon = "user_logo.png" if os.path.exists("user_logo.png") else None

for message in st.session_state.messages:
    avatar = bot_icon if message["role"] == "assistant" else user_icon
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

# 10. Chat Input & Disclaimer
if chat_input := st.chat_input("Stel je eigen vraag aan OERbot..."):
    handle_query(chat_input); st.rerun()

st.markdown('<div class="disclaimer-btn">', unsafe_allow_html=True)
if st.button("Algemene Voorwaarden & Disclaimer"):
    st.session_state.show_disclaimer = not st.session_state.show_disclaimer
st.markdown('</div>', unsafe_allow_html=True)

if st.session_state.show_disclaimer:
    st.info("**Disclaimer:** Dit hulpmiddel is informatief. Aan de antwoorden kunnen geen rechten worden ontleend. De officiële OER-tekst is leidend.")

# 11. Beheerder Sidebar (Multi-upload)
with st.sidebar:
    if not st.session_state.logged_in:
        st.title("Admin Login")
        u = st.text_input("Username"); p = st.text_input("Password", type="password")
        if st.button("Inloggen"):
            if u == admin_user and p == admin_pass: st.session_state.logged_in = True; st.rerun()
    else:
        st.title("Beheer")
        uploaded_files = st.file_uploader("Upload OER PDF's", type="pdf", accept_multiple_files=True)
        if uploaded_files:
            os.makedirs("uploads", exist_ok=True)
            saved_paths = []
            for uploaded_file in uploaded_files:
                path = os.path.join("uploads", uploaded_file.name)
                with open(path, "wb") as f: f.write(uploaded_file.getbuffer())
                saved_paths.append(path)
            
            # Verwerk alle bestanden in de uploads map
            st.session_state.vector_store = initialize_vector_store(saved_paths)
            st.success(f"{len(uploaded_files)} bestanden verwerkt!")
        
        if st.button("Uitloggen"):
            st.session_state.logged_in = False; st.rerun()

# Auto-load bestaande PDF's
if st.session_state.vector_store is None and os.path.exists("uploads"):
    existing_files = [os.path.join("uploads", f) for f in os.listdir("uploads") if f.endswith(".pdf")]
    if existing_files:
        st.session_state.vector_store = initialize_vector_store(existing_files)