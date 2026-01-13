import streamlit as st
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
import fitz
import os

# 1. Page configuration
st.set_page_config(page_title="OERbot - Je OER Klasgenoot", page_icon="📚")

# 2. Custom CSS for Dulon College / OERbot vibe
def apply_custom_css():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Nunito:wght@300;400;700&display=swap');
        html, body, [class*="css"]  {
            font-family: 'Nunito', sans-serif;
        }
        .stChatMessage {
            border-radius: 15px;
            margin-bottom: 10px;
        }
        </style>
    """, unsafe_allow_html=True)

apply_custom_css()

# 3. Retrieve Secrets
api_key = st.secrets.get("openai_api_key")
admin_user = st.secrets.get("admin_username")
admin_pass = st.secrets.get("admin_password")

if not api_key:
    st.error("OpenAI API Sleutel ontbreekt in de instellingen!")
    st.stop()

# Initialize Chat Model (Smarter at following persona instructions)
llm = ChatOpenAI(model="gpt-4o", api_key=api_key, temperature=0.5)

# 4. Vector Store Logic
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
                    metadata={"page": page.number + 1, "source": os.path.basename(pdf_path)}
                ))

        persist_directory = os.path.join(os.getcwd(), "chroma_db", "shared_pdf")
        vector_store = Chroma.from_documents(
            documents, 
            embeddings, 
            persist_directory=persist_directory
        )
        return vector_store
    except Exception as e:
        st.error(f"Fout bij het laden van het document: {str(e)}")
        return None

# 5. Initialize Session States
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'messages' not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hoi! Ik ben OERbot 😊. Goed dat je er bent! Heb je een vraag over het examenreglement of de OER? Ik kijk graag even met je mee!"}]
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

# 6. Admin Sidebar
st.sidebar.title("OERbot Beheer")
if not st.session_state.logged_in:
    u = st.sidebar.text_input("Gebruikersnaam")
    p = st.sidebar.text_input("Wachtwoord", type="password")
    if st.sidebar.button("Inloggen"):
        if u == admin_user and p == admin_pass:
            st.session_state.logged_in = True
            st.rerun()
        else:
            st.sidebar.error("Onjuiste gegevens")
else:
    st.sidebar.success("Ingelogd als beheerder")
    uploaded_file = st.sidebar.file_uploader("Upload OER PDF", type="pdf")
    if uploaded_file:
        os.makedirs("uploads", exist_ok=True)
        pdf_path = os.path.join("uploads", uploaded_file.name)
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        with open("uploads/pdf_name.txt", "w") as f:
            f.write(uploaded_file.name)
        st.session_state.vector_store = initialize_vector_store(pdf_path)
        st.sidebar.success(f"{uploaded_file.name} is nu actief!")

# Auto-load logic
if st.session_state.vector_store is None and os.path.exists("uploads/pdf_name.txt"):
    with open("uploads/pdf_name.txt", "r") as f:
        name = f.read().strip()
    path = os.path.join("uploads", name)
    if os.path.exists(path):
        st.session_state.vector_store = initialize_vector_store(path)

# 7. Main Interface
st.title("🤖 OERbot")
st.caption("Jouw klasgenoot voor vragen over het examenreglement op het Dulon College.")

# FAQ Section
st.markdown("### Veelgestelde vragen")
col1, col2 = st.columns(2)
with col1:
    if st.button("Hoe werkt herkansing?"):
        st.session_state.messages.append({"role": "user", "content": "Hoe werkt een herkansing?"})
with col2:
    if st.button("Wat als ik ziek ben?"):
        st.session_state.messages.append({"role": "user", "content": "Wat moet ik doen als ik ziek ben voor een examen?"})

# 8. Chat Display
bot_avatar = "custom_bot_image.png" if os.path.exists("custom_bot_image.png") else "🤖"

for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar=bot_avatar if message["role"] == "assistant" else None):
        st.markdown(message["content"])

# 9. Chat Logic
if prompt := st.chat_input("Stel je vraag aan OERbot..."):
    # User message
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    if st.session_state.vector_store is None:
        with st.chat_message("assistant", avatar=bot_avatar):
            msg = "Ik heb nog geen documenten kunnen vinden om te lezen. Vraag even aan je docent of ze de OER willen uploaden! 👍"
            st.markdown(msg)
            st.session_state.messages.append({"role": "assistant", "content": msg})
    else:
        # Search context
        results = st.session_state.vector_store.similarity_search_with_score(prompt, k=3)
        # Filter logic: lower score = better match in Chroma
        docs = [r[0] for r in results if r[1] < 0.6] 

        if not docs:
            response = "Ik kan je hier helaas alleen helpen met informatie uit de OER. Deze vraag staat niet in de OER, dus kan ik je hier niets over zeggen. Als klasgenoot zou ik je wel aanraden om dit even te bespreken met je coach of opleiding. 😊"
        else:
            context_text = "\n\n".join([d.page_content for d in docs])
            
            system_prompt = """
            Jij bent OERbot, een vriendelijke, laagdrempelige klasgenoot op het Dulon College. 
            Je helpt studenten bij vragen over de OER (Onderwijs- en Examenregeling).

            STIJLREGELS:
            - Taalstijl: B1-niveau. Simpel, stap voor stap, geen moeilijke woorden.
            - Toon: Warm, rustig, geruststellend, alsof je naast de student in de klas zit.
            - Gebruik 'je' en 'jij'.
            - Gebruik emoji's (😊, 👍, ❗, 😔) op een natuurlijke manier.
            - Nooit formeel of afstandelijk. Nooit commanderend.
            - Begin vaak met zinnen als: "Goed dat je dit even checkt!", "Snap dat dit verwarrend voelt." of "Ik kijk het graag samen met je na."

            INHOUDSREGELS:
            - Gebruik ALLEEN de meegeleverde context uit het bestand: 20240710_Examenreglement ROC A12 2024-2025 versie 1.0.pdf.
            - Verzin niets. Als het er niet staat, zeg je dat vriendelijk.
            - Geef GEEN persoonlijke meningen of negatieve opmerkingen over docenten.
            - Geef ALTIJD een bronvermelding (bijv. artikel 11 lid 2) bij elke feitelijke regel.

            ANTWOORD STRUCTUUR:
            1. Bevestig het gevoel van de student.
            2. Geef een samenvatting van de regels uit de OER (met bronvermelding).
            3. Geef een duidelijke 'Call to action' (Wat moet de student nu doen?).
            4. Sluit altijd positief af (bijv. "Succes met je studie! 👍").

            CONTEXT:
            {context}
            """
            
            chat_template = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", "{question}")
            ])
            
            # Use streaming for better UI experience
            with st.chat_message("assistant", avatar=bot_avatar):
                response_container = st.empty()
                full_response = ""
                
                formatted_prompt = chat_template.format_messages(context=context_text, question=prompt)
                
                for chunk in llm.stream(formatted_prompt):
                    full_response += chunk.content
                    response_container.markdown(full_response + "▌")
                
                response_container.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})