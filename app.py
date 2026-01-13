import streamlit as st
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
import fitz
import os

# 1. Pagina Configuratie
st.set_page_config(page_title="OERbot - Dulon College", page_icon="📚")

# 2. Custom CSS voor de Dulon Rode Kleur (#e5241d)
def apply_custom_css():
    st.markdown(f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Nunito:wght@300;400;700&display=swap');
        
        /* Algemene font */
        html, body, [class*="css"]  {{
            font-family: 'Nunito', sans-serif;
        }}

        /* Accent kleur voor knoppen */
        .stButton>button {{
            background-color: #e5241d;
            color: white;
            border-radius: 10px;
            border: none;
            width: 100%;
        }}
        .stButton>button:hover {{
            background-color: #b31b17;
            color: white;
        }}

        /* Chat bubbels styling */
        .stChatMessage {{
            border-radius: 15px;
        }}

        /* Sidebar styling */
        [data-testid="stSidebar"] {{
            background-color: #f9f9f9;
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
                documents.append(Document(
                    page_content=text, 
                    metadata={"page": page.number + 1}
                ))
        vector_store = Chroma.from_documents(
            documents, embeddings, 
            persist_directory=os.path.join(os.getcwd(), "chroma_db", "shared_pdf")
        )
        return vector_store
    except Exception as e:
        return None

# 5. Session States initialiseren
if 'messages' not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hoi! Ik ben OERbot 😊. Jouw hulpje voor alle vragen over het examenreglement. Waar kan ik je vandaag mee helpen?"}]
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None

# 6. Logo en Branding bovenaan
col1, col2, col3 = st.columns([1,2,1])
with col2:
    if os.path.exists("logo.png"):
        st.image("logo.png", use_container_width=True)
    else:
        st.title("🤖 OERbot")

st.caption("Jouw klasgenoot voor vragen over de OER op het Dulon College.")

# 7. Functie voor het verwerken van vragen (voor Chat én Knoppen)
def handle_query(query):
    # Voeg gebruikersvraag toe aan de chat
    st.session_state.messages.append({"role": "user", "content": query})
    
    if st.session_state.vector_store is None:
        response = "Oeps! Ik kan de OER nog niet lezen. Vraag de beheerder om een PDF te uploaden via het menu aan de linkerkant. 👍"
        st.session_state.messages.append({"role": "assistant", "content": response})
    else:
        # Zoek context
        results = st.session_state.vector_store.similarity_search_with_score(query, k=3)
        docs = [r[0] for r in results if r[1] < 0.6]

        if not docs:
            response = "Ik kan je hier helaas alleen helpen met informatie uit de OER. Deze vraag staat niet in de OER, dus kan ik je hier niets over zeggen. 😊"
        else:
            context_text = "\n\n".join([d.page_content for d in docs])
            system_prompt = f"""
            Jij bent OERbot, een vriendelijke MBO-klasgenoot.
            Stijl: B1-niveau, warm, empathisch, emoji's (😊, 👍, ❗, 😔).
            Gebruik ALLEEN deze context: {context_text}
            Structuur: 1. Bevestig gevoel. 2. Samenvatting met bron (artikel X). 3. Call to action. 4. Positieve afsluiting.
            """
            
            chat_template = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", "{question}")
            ])
            
            # Genereer antwoord
            formatted = chat_template.format_messages(question=query)
            full_response = ""
            for chunk in llm.stream(formatted):
                full_response += chunk.content
            st.session_state.messages.append({"role": "assistant", "content": full_response})

# 8. Quick Actions (Knoppen)
st.markdown("### Waar wil je meer over weten?")
q_col1, q_col2 = st.columns(2)
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

# 9. Chat Geschiedenis Tonen
bot_icon = "custom_bot_image.png" if os.path.exists("custom_bot_image.png") else "🤖"

for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar=bot_icon if message["role"] == "assistant" else None):
        st.markdown(message["content"])

# 10. Chat Input
if chat_input := st.chat_input("Stel je eigen vraag..."):
    handle_query(chat_input)
    st.rerun()

# 11. Beheerder Sidebar (Onveranderd)
if not st.session_state.logged_in:
    with st.sidebar.expander("Beheerder Login"):
        u = st.text_input("User")
        p = st.text_input("Pass", type="password")
        if st.button("Login"):
            if u == admin_user and p == admin_pass:
                st.session_state.logged_in = True
                st.rerun()
else:
    st.sidebar.title("Beheer")
    uploaded_file = st.sidebar.file_uploader("Upload OER PDF", type="pdf")
    if uploaded_file:
        os.makedirs("uploads", exist_ok=True)
        pdf_path = os.path.join("uploads", uploaded_file.name)
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        with open("uploads/pdf_name.txt", "w") as f:
            f.write(uploaded_file.name)
        st.session_state.vector_store = initialize_vector_store(pdf_path)
        st.sidebar.success("PDF Verwerkt!")
    if st.sidebar.button("Uitloggen"):
        st.session_state.logged_in = False
        st.rerun()

# Auto-load logic voor PDF
if st.session_state.vector_store is None and os.path.exists("uploads/pdf_name.txt"):
    with open("uploads/pdf_name.txt", "r") as f:
        name = f.read().strip()
    path = os.path.join("uploads", name)
    if os.path.exists(path):
        st.session_state.vector_store = initialize_vector_store(path)