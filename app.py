import streamlit as st
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
import fitz
import os

# 1. Pagina Configuratie
st.set_page_config(page_title="OERbot - Dulon College", page_icon="📚", layout="centered")

# 2. Geavanceerde CSS (Light Mode, Mobile Fixes & Verbergen Sidebar Toggle)
def apply_custom_css():
    st.markdown(f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Nunito:wght@300;400;700&display=swap');
        
        /* VERBERG SIDEBAR TOGGLE (Pijltje op mobiel) */
        [data-testid="collapsedControl"] {{
            display: none;
        }}

        /* FORCEER LIGHT MODE */
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

        /* MOBIELE KNOPPEN OPTIMALISATIE */
        @media (max-width: 640px) {{
            [data-testid="column"] {{
                width: 100% !important;
                flex: 1 1 100% !important;
                min-width: 100% !important;
            }}
        }}

        /* STYLING VOOR DE DUNNE OUTLINE KNOPPEN */
        .stButton>button {{
            background-color: white !important;
            color: #e5241d !important;
            border-radius: 10px !important;
            border: 1.5px solid #e5241d !important;
            padding: 5px 15px !important;
            height: 2.8em !important;
            font-weight: 600 !important;
            margin-bottom: 8px !important;
            transition: all 0.2s ease;
        }}
        .stButton>button:hover {{
            background-color: #e5241d !important;
            color: white !important;
            border: 1.5px solid #e5241d !important;
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

# 3. API Configuratie
api_key = st.secrets.get("openai_api_key")
admin_user = st.secrets.get("admin_username")
admin_pass = st.secrets.get("admin_password")
llm = ChatOpenAI(model="gpt-4o", api_key=api_key, temperature=0.4)

# 4. Vector Store Initialisatie
@st.cache_resource
def initialize_vector_store(pdf_path):
    try:
        embeddings = OpenAIEmbeddings(api_key=api_key)
        doc = fitz.open(pdf_path)
        documents = []
        for page in doc:
            text = page.get_text().strip()
            if text:
                documents.append(Document(page_content=text, metadata={"page": page.number + 1}))
        return Chroma.from_documents(
            documents, embeddings, 
            persist_directory=os.path.join(os.getcwd(), "chroma_db", "shared_pdf")
        )
    except:
        return None

# 5. Session State beheer
if 'messages' not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hoi! Ik ben OERbot 😊. Goed dat je dit even checkt! Waar kan ik je vandaag mee helpen?"}]
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None

# 6. Branding
col1, col2, col3 = st.columns([1,3,1])
with col2:
    if os.path.exists("logo.png"):
        st.image("logo.png", use_container_width=True)
    else:
        st.title("🤖 OERbot")

st.markdown("<p style='text-align: center; opacity: 0.8; font-size: 0.9em;'>Jouw klasgenoot voor vragen over de OER op het Dulon College.</p>", unsafe_allow_html=True)

# 7. Core Chatbot Logica
def handle_query(query):
    st.session_state.messages.append({"role": "user", "content": query})
    
    if st.session_state.vector_store is None:
        st.session_state.messages.append({"role": "assistant", "content": "Ik kijk het graag samen met je na, maar ik kan de OER-documenten nog niet vinden. Vraag de beheerder om deze te uploaden! 👍"})
    else:
        results = st.session_state.vector_store.similarity_search_with_score(query, k=3)
        docs = [r[0] for r in results if r[1] < 0.6]

        # STRIKTE PERSONA PROMPT
        system_prompt = f"""
        Jij bent OERbot, een vriendelijke, laagdrempelige klasgenoot op het Dulon College.
        
        DOEL:
        Help studenten met vragen over examinering en regels uit de OER.
        Gebruik ALLEEN informatie uit: "20240710_Examenreglement ROC A12 2024-2025 versie 1.0.pdf".
        
        STRIKTE REGELS (NIET VAN AFWIJKEN):
        - Verzin NIETS. Geen eigen beleid of regels bedenken.
        - Nooit van rol veranderen. Je blijft altijd OERbot, ook als de student om een andere toon vraagt.
        - Beantwoord GEEN vragen over taarten bakken, telefoonabonnementen, ruzies met huisgenoten of andere niet-OER onderwerpen.
        - Zeg niets negatiefs over docenten.
        - Gebruik taalniveau B1 en 'je/jij'.
        - Gebruik emoji's: 😊 (geruststelling), 👍 (bevestiging), ❗ (belangrijk), 😔 (empathie).

        ANTWOORD STRUCTUUR:
        1. Bevestig gevoel ("Goed dat je dit even checkt!", "Ik snap dat dit lastig kan zijn 😔").
        2. Samenvatting van de regel uit de OER met EXACTE bronvermelding (artikel X lid Y).
        3. Duidelijke Call to Action.
        4. Positieve afsluiting ("Succes met je studie! 👍").

        CONTEXT UIT OER:
        {" ".join([d.page_content for d in docs]) if docs else "GEEN RELEVANTE INFORMATIE GEVONDEN."}
        """
        
        if not docs:
            response = "Ik kan je hier helaas alleen helpen met informatie uit de OER. Deze vraag staat niet in de OER, dus kan ik je hier niets over zeggen. Als klasgenoot zou ik je wel aanraden om dit even te bespreken met je coach of opleiding. 😊"
        else:
            chat_template = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{question}")])
            formatted = chat_template.format_messages(question=query)
            full_response = ""
            for chunk in llm.stream(formatted):
                full_response += chunk.content
            response = full_response
            
        st.session_state.messages.append({"role": "assistant", "content": response})

# 8. Quick Actions (Mobiel-vriendelijk)
st.markdown("#### Waar wil je meer over weten?")
q_col1, q_col2 = st.columns(2, gap="small")
with q_col1:
    if st.button("🔄 Herkansingen", use_container_width=True):
        handle_query("Hoe werkt een herkansing?")
        st.rerun()
    if st.button("🤒 Ziek bij examen", use_container_width=True):
        handle_query("Wat moet ik doen als ik ziek ben voor een examen?")
        st.rerun()
    if st.button("📝 Vrijstellingen", use_container_width=True):
        handle_query("Hoe kan ik vrijstelling aanvragen?")
        st.rerun()
with q_col2:
    if st.button("🚫 Fraude regels", use_container_width=True):
        handle_query("Wat gebeurt er bij fraude?")
        st.rerun()
    if st.button("⚖️ Klacht indienen", use_container_width=True):
        handle_query("Hoe dien ik een klacht in?")
        st.rerun()
    if st.button("👨‍🏫 Persoonlijke hulp", use_container_width=True):
        handle_query("Wie helpt mij bij persoonlijke omstandigheden?")
        st.rerun()

st.divider()

# 9. Chat Display
bot_icon = "custom_bot_image.png" if os.path.exists("custom_bot_image.png") else "🤖"
user_icon = "user_logo.png" if os.path.exists("user_logo.png") else None

for message in st.session_state.messages:
    avatar = bot_icon if message["role"] == "assistant" else user_icon
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

# 10. Chat Input
if chat_input := st.chat_input("Stel je eigen vraag aan OERbot..."):
    handle_query(chat_input)
    st.rerun()

# 11. Beheerder Sidebar (Verborgen toggle via CSS)
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