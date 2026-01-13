import streamlit as st
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
import fitz
import os

# 1. Pagina Configuratie
st.set_page_config(page_title="OERbot - Dulon College", page_icon="📚", layout="centered")

# 2. Geavanceerde CSS (Outline Buttons, Dunner & Fix voor Icons)
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

        /* STYLING VOOR DE DUNNE OUTLINE KNOPPEN */
        .stButton>button {{
            background-color: white !important;
            color: #e5241d !important;
            border-radius: 10px !important;
            border: 2px solid #e5241d !important;
            padding: 5px 15px !important;
            height: 2.8em !important;
            width: 100% !important;
            font-weight: 600 !important;
            margin-bottom: 8px !important;
            transition: all 0.2s ease;
        }}
        .stButton>button:hover {{
            background-color: #e5241d !important;
            color: white !important;
            box-shadow: 0 2px 5px rgba(229, 36, 29, 0.3);
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

# 3. Gegevens ophalen
api_key = st.secrets.get("openai_api_key")
admin_user = st.secrets.get("admin_username")
admin_pass = st.secrets.get("admin_password")

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
                documents.append(Document(page_content=text, metadata={"page": page.number + 1}))
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

st.markdown("<p style='text-align: center; opacity: 0.8; font-size: 0.9em;'>Jouw klasgenoot voor vragen over de OER op het Dulon College.</p>", unsafe_allow_html=True)

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
            system_prompt = f"""
            Jij bent OERbot, een vriendelijke MBO-klasgenoot op het Dulon College.
            DOEL: Studenten helpen met vragen over examinering uit de OER.
            STIJL: B1-niveau, warm, empathisch, emoji's. Gebruik 'je'/'jij'.
            
            ANTWOORD STRUCTUUR:
            1. Bevestig gevoel.
            2. Samenvatting regel uit OER met bron (artikel X).
            3. Call to Action.
            4. Positieve afsluiting.

            CONTEXT: {context_text}
            """
            chat_template = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{question}")])
            formatted = chat_template.format_messages(question=query)
            full_response = ""
            for chunk in llm.stream(formatted):
                full_response += chunk.content
            st.session_state.messages.append({"role": "assistant", "content": full_response})

# 8. Quick Actions (Dunne Outline Buttons)
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

# 9. Chat Geschiedenis (Fix voor Icons)
bot_icon = "custom_bot_image.png" if os.path.exists("custom_bot_image.png") else "🤖"

for message in st.session_state.messages:
    # Als de rol 'user' is, gebruiken we None voor het standaard icoon, anders het bot-icoon
    current_avatar = bot_icon if message["role"] == "assistant" else None
    with st.chat_message(message["role"], avatar=current_avatar):
        st.markdown(message["content"])

# 10. Chat Input
if chat_input := st.chat_input("Stel je eigen vraag aan OERbot..."):
    handle_query(chat_input)
    st.rerun()

# 11. Beheerder Sidebar
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