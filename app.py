import streamlit as st
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
import fitz
import os

# 1. Pagina Configuratie - Nu ingesteld op 'collapsed' bij opstarten
st.set_page_config(
    page_title="OERbot - Dulon College", 
    page_icon="📚", 
    layout="centered",
    initial_sidebar_state="collapsed"
)

# 2. Geavanceerde CSS (Huisstijl & Light Mode behouden)
def apply_custom_css():
    st.markdown(f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Nunito:wght@300;400;700&display=swap');
        
        [data-testid="stAppViewContainer"], .main, [data-testid="stHeader"] {{
            background-color: white !important;
            color: #1f1f1f !important;
        }}
        [data-testid="stSidebar"] {{ background-color: #f0f2f6 !important; }}
        
        /* Zorg dat tekst Nunito gebruikt, maar sluit iconen uit zodat ze laden */
        p, h1, h2, h3, h4, label, .stMarkdown {{
            color: #1f1f1f !important;
            font-family: 'Nunito', sans-serif !important;
        }}
        
        /* Voorkom dat Nunito de systeem-iconen van Streamlit overschrijft */
        span:not(.material-icons):not([data-testid="stIcon"]):not([class*="st-"]) {{
            font-family: 'Nunito', sans-serif !important;
        }}
        
        @media (max-width: 640px) {{
            [data-testid="column"] {{ width: 100% !important; flex: 1 1 100% !important; min-width: 100% !important; }}
        }}
        .stButton>button {{
            background-color: white !important;
            color: #e5241d !important;
            border-radius: 10px !important;
            border: 1.5px solid #e5241d !important;
            padding: 5px 15px !important;
            height: 2.8em !important;
            font-weight: 600 !important;
            margin-bottom: 8px !important;
            width: 100% !important;
        }}
        .stButton>button:hover {{ background-color: #e5241d !important; color: white !important; }}
        [data-testid="stChatMessage"] {{ background-color: #f8f9fa !important; border: 1px solid #eee !important; border-radius: 15px !important; }}
        </style>
    """, unsafe_allow_html=True)

apply_custom_css()

# 3. API & AI Model
api_key = st.secrets.get("openai_api_key")
admin_user = st.secrets.get("admin_username")
admin_pass = st.secrets.get("admin_password")
llm = ChatOpenAI(model="gpt-5-nano", api_key=api_key, temperature=0.5)

# 4. Vector Store Initialisatie
@st.cache_resource
def initialize_vector_store(pdf_paths):
    try:
        embeddings = OpenAIEmbeddings(api_key=api_key)
        all_documents = []
        
        for path in pdf_paths:
            if not os.path.exists(path): continue
            doc = fitz.open(path)
            file_name = os.path.basename(path)
            for page in doc:
                text = page.get_text().strip()
                if text:
                    all_documents.append(Document(
                        page_content=text, 
                        metadata={"page": page.number + 1, "source": file_name}
                    ))
        
        if not all_documents:
            return None

        return Chroma.from_documents(
            all_documents, 
            embeddings, 
            persist_directory=os.path.join(os.getcwd(), "chroma_db", "shared_pdf")
        )
    except Exception as e:
        st.error(f"Fout bij verwerken documenten: {e}")
        return None

# 5. Session State beheer
if 'messages' not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hoi! Ik ben OERbot 😊. Ik heb alle reglementen doorgelezen. Waar kan ik je vandaag mee helpen?"}]
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'show_disclaimer' not in st.session_state:
    st.session_state.show_disclaimer = False

# AUTO-LOAD LOGIC (GitHub Persistence)
if 'vector_store' not in st.session_state:
    upload_dir = "uploads"
    if os.path.exists(upload_dir):
        all_pdfs = [os.path.join(upload_dir, f) for f in os.listdir(upload_dir) if f.endswith(".pdf")]
        if all_pdfs:
            st.session_state.vector_store = initialize_vector_store(all_pdfs)
        else:
            st.session_state.vector_store = None
    else:
        st.session_state.vector_store = None

# 6. Branding
col1, col2, col3 = st.columns([1,3,1])
with col2:
    if os.path.exists("logo.png"): st.image("logo.png", use_container_width=True)
    else: st.title("🤖 OERbot")
st.markdown("<p style='text-align: center; opacity: 0.8; font-size: 0.9em;'>Jouw hulp voor vragen over de OER op het Dulon College.</p>", unsafe_allow_html=True)

# 7. Centrale Chat Logica (With Loading Feedback)
def handle_query(query):
    st.session_state.messages.append({"role": "user", "content": query})
    
    if st.session_state.vector_store is None:
        st.session_state.messages.append({"role": "assistant", "content": "Ik help je graag, maar ik heb nog geen documenten kunnen inladen. 👍"})
    else:
        with st.spinner("OERbot is even aan het zoeken in de documenten... 📚"):
            results = st.session_state.vector_store.similarity_search_with_score(query, k=4)
            docs = [r[0] for r in results if r[1] < 0.6]

            if not docs:
                response = "Ik kan je hier helaas alleen helpen met informatie uit de OER. Deze vraag staat niet in de OER, dus kan ik je hier niets over zeggen. Als klasgenoot zou ik je wel aanraden om dit even te bespreken met je coach of opleiding. 😊"
            else:
                context_text = "\n\n".join([f"Bron [{d.metadata['source']}]: {d.page_content}" for d in docs])
                
                system_prompt = f"""
                Jij bent OERbot, een vriendelijke klasgenoot op het Dulon College.
                
                STRIKTE OPDRACHT:
                - Je bent een afgesloten systeem. Je mag UITSLUITEND antwoorden geven op basis van de verstrekte OER-documenten.
                - Als een vraag NIET over de OER gaat, weiger dan beleefd.
                
                STIJL:
                - Varieer je begroetingen en gebruik B1-taal. 
                - Vermeld ALTIJD de bron op een NIEUWE REGEL met 📖.
                
                CONTEXT: {context_text}
                """
                chat_template = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{question}")])
                formatted = chat_template.format_messages(question=query)
                
                full_response = "".join([chunk.content for chunk in llm.stream(formatted)])
                response = full_response
            
            st.session_state.messages.append({"role": "assistant", "content": response})

# 8. Quick Actions
st.markdown("#### Waar wil je meer over weten?")
q_col1, q_col2 = st.columns(2, gap="small")
with q_col1:
    if st.button("🔄 Herkansingen", use_container_width=True): handle_query("Hoe werkt een herkansing?"); st.rerun()
    if st.button("🤒 Ziek bij examen", use_container_width=True): handle_query("Wat als ik ziek ben voor een examen?"); st.rerun()
with q_col2:
    if st.button("🚫 Fraude regels", use_container_width=True): handle_query("Wat gebeurt er bij fraude?"); st.rerun()
    if st.button("👨‍🏫 Persoonlijke hulp", use_container_width=True): handle_query("Wie helpt bij persoonlijke omstandigheden?"); st.rerun()

st.divider()

# 9. Chat Geschiedenis
bot_icon, user_icon = ("custom_bot_image.png" if os.path.exists("custom_bot_image.png") else "🤖"), ("user_logo.png" if os.path.exists("user_logo.png") else None)
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar=bot_icon if message["role"] == "assistant" else user_icon):
        st.markdown(message["content"])

# 10. Chat Input
if chat_input := st.chat_input("Stel je vraag aan OERbot..."):
    handle_query(chat_input); st.rerun()

# 11. Zijbalk Admin
with st.sidebar:
    if not st.session_state.logged_in:
        st.title("Admin")
        u, p = st.text_input("User"), st.text_input("Pass", type="password")
        if st.button("Login"):
            if u == admin_user and p == admin_pass: st.session_state.logged_in = True; st.rerun()
    else:
        st.title("Beheer")
        uploaded_files = st.file_uploader("Upload OER PDF's", type="pdf", accept_multiple_files=True)
        if uploaded_files:
            os.makedirs("uploads", exist_ok=True)
            saved_paths = []
            for uploaded_file in uploaded_files:
                pdf_path = os.path.join("uploads", uploaded_file.name)
                with open(pdf_path, "wb") as f: f.write(uploaded_file.getbuffer())
                saved_paths.append(pdf_path)
            
            st.session_state.vector_store = initialize_vector_store(saved_paths)
            st.success("Bestanden verwerkt!")
        if st.button("Uitloggen"): st.session_state.logged_in = False; st.rerun()

    st.sidebar.markdown("---")
    if st.sidebar.button("Algemene Voorwaarden"): st.session_state.show_disclaimer = not st.session_state.show_disclaimer
    if st.session_state.show_disclaimer: st.sidebar.info("Disclaimer: Aan antwoorden kunnen geen rechten worden ontleend.")