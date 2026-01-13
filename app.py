import streamlit as st
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
import fitz
import os

# 1. Pagina Configuratie
st.set_page_config(page_title="OERbot - Dulon College", page_icon="📚", layout="centered")

# 2. Geavanceerde CSS (Light Mode Force & Mobiele Styling)
def apply_custom_css():
    st.markdown(f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Nunito:wght@300;400;700&display=swap');
        
        /* FORCEER LIGHT MODE */
        [data-testid="stAppViewContainer"], .main, [data-testid="stHeader"] {{
            background-color: white !important;
            color: #1f1f1f !important;
        }}
        [data-testid="stSidebar"] {{
            background-color: #f0f2f6 !important;
        }}
        p, h1, h2, h3, h4, span, label {{
            color: #1f1f1f !important;
            font-family: 'Nunito', sans-serif !important;
        }}

        /* STYLING VOOR DE KNOPPEN */
        .stButton>button {{
            background-color: #e5241d !important;
            color: white !important;
            border-radius: 12px !important;
            border: none !important;
            padding: 10px 20px !important;
            height: 4em !important;
            width: 100% !important;
            font-weight: bold !important;
            margin-bottom: 10px !important;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            transition: all 0.3s ease;
        }}
        .stButton>button:hover {{
            background-color: #b31b17 !important;
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }}

        /* CHAT BUBBELS */
        [data-testid="stChatMessage"] {{
            background-color: #f8f9fa !important;
            border: 1px solid #eee !important;
            border-radius: 15px !important;
        }}
        </style>
    """, unsafe_allow_html=True)

apply_custom_css()

# 3. Gegevens en Modellen
api_key = st.secrets.get("openai_api_key")
admin_user = st.secrets.get("admin_username")
admin_pass = st.secrets.get("admin_password")

# Gebruik GPT-4o voor de beste Persona-ervaring
llm = ChatOpenAI(model="gpt-4o", api_key=api_key, temperature=0.4)

# 4. Vector Store Functie
@st.cache_resource
def initialize_vector_store(pdf_path):
    try:
        embeddings = OpenAIEmbeddings(api_key=api_key)
        doc = fitz.open(pdf_path)
        documents = []
        for page in doc:
            text = page.get_text().strip()
            if text:
                documents.append(Document(
                    page_content=text, 
                    metadata={"page": page.number + 1}
                ))
        return Chroma.from_documents(
            documents, embeddings, 
            persist_directory=os.path.join(os.getcwd(), "chroma_db", "shared_pdf")
        )
    except:
        return None

# 5. Session States
if 'messages' not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hoi! Ik ben OERbot 😊. Jouw hulpje voor alle vragen over het examenreglement. Waar kan ik je vandaag mee helpen?"}]
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None

# 6. Logo Branding
col1, col2, col3 = st.columns([1,3,1])
with col2:
    if os.path.exists("logo.png"):
        st.image("logo.png", use_container_width=True)
    else:
        st.title("🤖 OERbot")

st.markdown("<p style='text-align: center; opacity: 0.8;'>Jouw klasgenoot voor vragen over de OER op het Dulon College.</p>", unsafe_allow_html=True)

# 7. Centrale Functie voor Vragen
def handle_query(query):
    st.session_state.messages.append({"role": "user", "content": query})
    
    if st.session_state.vector_store is None:
        st.session_state.messages.append({"role": "assistant", "content": "Oeps! Ik kan de OER nog niet lezen. Vraag de beheerder om de PDF te uploaden! 👍"})
    else:
        results = st.session_state.vector_store.similarity_search_with_score(query, k=3)
        docs = [r[0] for r in results if r[1] < 0.6]

        if not docs:
            response = "Ik kan je hier helaas alleen helpen met informatie uit de OER. Deze vraag staat niet in de OER, dus kan ik je hier niets over zeggen. 😊"
        else:
            context_text = "\n\n".join([d.page_content for d in docs])
            # DE OERBOT PERSONA PROMPT
            system_prompt = f"""
            Jij bent OERbot, een vriendelijke MBO-klasgenoot op het Dulon College.
            DOEL: Studenten helpen met vragen over examinering uit het document: “20240710_Examenreglement ROC A12 2024-2025 versie 1.0.pdf”.
            
            STIJL:
            - Praat warm, rustig, geruststellend en behulpzaam.
            - Gebruik 'je' en 'jij'. Taalniveau B1 (geen moeilijke woorden).
            - Gebruik emoji's (😊, 👍, ❗, 😔) op een natuurlijke manier.
            - Je verzint niets. Alleen informatie uit de OER gebruiken.
            
            STRUCTUUR VAN JE ANTWOORD:
            1. Bevestig gevoel (bijv: "Goed dat je dit even checkt!").
            2. Samenvatting van de regel uit de OER.
            3. ALTIJD bronvermelding met artikelnummer (bijv: artikel 11 lid 2).
            4. Duidelijke Call to Action (wat moet de student doen?).
            5. Positieve afsluiting.

            CONTEXT: {context_text}
            """
            
            chat_template = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", "{question}")
            ])
            
            formatted = chat_template.format_messages(question=query)
            full_response = ""
            for chunk in llm.stream(formatted):
                full_response += chunk.content
            st.session_state.messages.append({"role": "assistant", "content": full_response})

# 8. Quick Actions (Knoppen met betere spacing)
st.markdown("#### Waar wil je meer over weten?")
q_col1, q_col2 = st.columns(2, gap="small")
with q_col1:
    if st.button("🔄 Herkansingen"):
        handle_query("Hoe werkt een herkansing?")
        st.rerun()
    if st.button("🤒 Ziek bij examen"):
        handle_query("Wat moet ik doen als ik ziek ben voor een examen?")
        st.rerun()
    if st.button("📝 Vrijstellingen"):
        handle_query("Hoe kan ik vrijstelling aanvragen voor een vak?")
        st.rerun()
with q_col2:
    if st.button("🚫 Fraude regels"):
        handle_query("Wat gebeurt er als ik word verdacht van fraude?")
        st.rerun()
    if st.button("⚖️ Klacht indienen"):
        handle_query("Hoe kan ik een klacht of bezwaar indienen?")
        st.rerun()
    if st.button("👨‍🏫 Persoonlijke hulp"):
        handle_query("Wie kan mij helpen bij persoonlijke omstandigheden?")
        st.rerun()

st.divider()

# 9. Chat Geschiedenis
bot_icon = "custom_bot_image.png" if os.path.exists("custom_bot_image.png") else "🤖"

for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar=bot_icon if message["role"] == "assistant" else None):
        st.markdown(message["content"])

# 10. Chat Input
if chat_input := st.chat_input("Stel je eigen vraag aan OERbot..."):
    handle_query(chat_input)
    st.rerun()

# 11. Beheerder Sidebar
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
        uploaded_file = st.file_uploader("Upload OER PDF", type="pdf")
        if uploaded_file:
            os.makedirs("uploads", exist_ok=True)
            pdf_path = os.path.join("uploads", uploaded_file.name)
            with open(pdf_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            with open("uploads/pdf_name.txt", "w") as f:
                f.write(uploaded_file.name)
            st.session_state.vector_store = initialize_vector_store(pdf_path)
            st.success("PDF Verwerkt!")
        if st.button("Uitloggen"):
            st.session_state.logged_in = False
            st.rerun()

# PDF Auto-load
if st.session_state.vector_store is None and os.path.exists("uploads/pdf_name.txt"):
    with open("uploads/pdf_name.txt", "r") as f:
        name = f.read().strip()
    path = os.path.join("uploads", name)
    if os.path.exists(path):
        st.session_state.vector_store = initialize_vector_store(path)