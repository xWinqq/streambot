 # De import statements laden de benodigde bibliotheken en modules voor de applicatie.

import streamlit as st

from langchain.schema import Document

from langchain_openai import OpenAIEmbeddings, OpenAI

from langchain_community.vectorstores import Chroma

from langchain.prompts import ChatPromptTemplate

import fitz

import os


# Page configuration with custom icon

st.set_page_config(page_title="Examenreglement Chatbot", page_icon="📚")


# Function to apply custom CSS

def apply_custom_css():

    custom_css = """

    <style>

    @import url('https://fonts.googleapis.com/css2?family=Nunito:wght@300;400;700&display=swap');


    .stApp h1, div, p, span, label, h1, h2, h3, h4, h5, h6 {

        font-family: 'Meiryo', 'Nunito', 'Arial', sans-serif;

    }

    </style>

    """

    st.markdown(custom_css, unsafe_allow_html=True)


# Apply custom CSS

apply_custom_css()


# De OpenAI API-sleutel wordt opgehaald uit de Streamlit secrets.

api_key = st.secrets["openai_api_key"]


# Initialize the language model

llm = OpenAI(api_key=api_key)


# De functie initialize vector store laadt een pdf-bestand, extraheert de tekst en slaat deze op in een vector store. Wanneer de tekst van het pdf wordt uitgelezen wordt dit omgezet in vectoren, dit zorgt ervoor dat het taalmodel de tekst kan begrijpen en er vragen over kan antwoorden. Deze vectoren worden dan in een vector store opgeslagen zodat dit niet elke keer omgezet hoeft te worden in vectoren.


@st.cache_resource

def initialize_vector_store(pdf_path):

    """Initialize the vector store with the PDF content."""

    try:

        embeddings = OpenAIEmbeddings(api_key=api_key)


        # Load PDF with PyMuPDF (fitz)

        doc = fitz.open(pdf_path)


        # Extract text from each page and create Document objects

        documents = []

        for page in doc:

            text = page.get_text().strip()

            if text:  # Check if the text is not empty

                documents.append(

                    Document(

                        page_content=text,

                        metadata={"page": page.number + 1}

                    )

                )


        # Specify the directory for Chroma's database

        persist_directory = os.path.join(os.getcwd(), "chroma_db", "shared_pdf")


        # Create a new database

        vector_store = Chroma.from_documents(

            documents,

            embeddings,

            persist_directory=persist_directory

        )

        vector_store.persist()  # Save the database


        return vector_store

    except Exception as e:

        st.error(f"Error initializing vector store: {str(e)}")

        return None


# Ensure session state is initialized

if 'similarity_threshold' not in st.session_state:

    st.session_state.similarity_threshold = 0.50


if 'messages' not in st.session_state:

    st.session_state.messages = [{"role": "assistant", "content": "Welkom! Hoe kan ik je helpen met het examenreglement?"}]


# Set default similarity threshold if not defined

similarity_threshold = st.session_state.similarity_threshold


st.title("📚 Examenreglement Chatbot")


# Controleert of de gebruiker is ingelogd en toont het login formulier indien nodig.

def login(username, password):

    return username == st.secrets["admin_username"] and password == st.secrets["admin_password"]


# Check if the user is logged in

if "logged_in" not in st.session_state:

    st.session_state.logged_in = False


if not st.session_state.logged_in:

    st.sidebar.title("Login")

    username = st.sidebar.text_input("Username")

    password = st.sidebar.text_input("Password", type="password")

    if st.sidebar.button("Login"):

        if login(username, password):

            st.session_state.logged_in = True

            st.sidebar.success("Login successful")

            st.rerun()

        else:

            st.sidebar.error("Invalid username or password")

else:

    # Administrator page

    st.sidebar.title("Administrator")

    uploaded_file = st.sidebar.file_uploader("Upload a new PDF", type="pdf")

    similarity_threshold = st.sidebar.slider("Set similarity search strictness", 0.0, 1.0, 0.5)


    # Process PDF upload

    if uploaded_file:

        pdf_name = uploaded_file.name

        pdf_path = os.path.join("uploads", pdf_name)


        # Ensure the uploads directory exists.

        os.makedirs("uploads", exist_ok=True)


        with open(pdf_path, "wb") as f:

            f.write(uploaded_file.getbuffer())

        st.sidebar.success(f"PDF '{pdf_name}' succesvol geüpload!")


        # Save the PDF name in session state and a text file

        st.session_state.pdf_name = pdf_name

        with open("uploads/pdf_name.txt", "w") as f:

            f.write(pdf_name)


        # Initialize vector store with the new PDF

        vector_store = initialize_vector_store(pdf_path)

        st.session_state.vector_store = vector_store


    # Display the uploaded PDF

    if 'pdf_name' in st.session_state:

        pdf_name = st.session_state.pdf_name

        pdf_path = os.path.join("uploads", pdf_name)

        st.sidebar.markdown(f"**Geüpload PDF:** {pdf_name}")

        with open(pdf_path, "rb") as f:

            st.sidebar.download_button("Download PDF", f, file_name=pdf_name)

            st.sidebar.button("Verwijder PDF", on_click=lambda: os.remove(pdf_path) or os.remove("uploads/pdf_name.txt") or st.session_state.pop('pdf_name') or st.experimental_rerun())


# Load the shared PDF for all users

if "vector_store" not in st.session_state:

    if os.path.exists("uploads/pdf_name.txt"):

        with open("uploads/pdf_name.txt", "r") as f:

            pdf_name = f.read().strip()

        pdf_path = os.path.join("uploads", pdf_name)

        st.session_state.pdf_name = pdf_name

        st.session_state.vector_store = initialize_vector_store(pdf_path)

    else:

        st.error("Geen gedeelde PDF gevonden. Upload een PDF als admin.")


vector_store = st.session_state.vector_store


# Veelgestelde vragen sectie met rechthoekige blokken en klikbare acties

st.markdown("### Veelgestelde vragen")


# Styling met rechthoekige blokken

faq_style = """

    <style>

        .faq-box {

            border: 1px solid #ccc;

            border-radius: 8px;

            padding: 15px;

            margin-bottom: 10px;

            background-color: #f9f9f9;

            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);

        }

        .faq-box:hover {

            background-color: #f1f1f1;

        }

    </style>

"""

st.markdown(faq_style, unsafe_allow_html=True)


# FAQ blokken

faq_questions = {

    "Hoe kan ik vrijstelling aanvragen?": "Als je een vak al hebt gehaald bij een vorige opleiding kun je een vrijstelling aanvragen. Hierbij komen wel een paar eisen bij kijken:\nHet vak moet al met een voldoende zijn afgerond.\nHet vak is binnen 10 jaar behaald.\n\nAls je aan alle eisen voldoet kun je langs jouw SLB'er gaan. Vervolgens hoor je vanzelf of je ook daadwerkelijk vrijstelling krijgt of niet.",

    "Hoe kan ik extra tijd aanvragen?": "Als je recht hebt op extra tijd zal dit al besproken zijn tijdens het introductie gesprek. Ben je dit vergeten moet je langs je SLB'er gaan om alsnog een aanvraag in te dienen. Die gaat vervolgens in actie als jij recht hebt op vrijstelling.",

    "Herkansingen": "Per examen zijn er twee pogingen beschikbaar. Als het na de 2de poging niet is gelukt zal er een aanvraag worden gestuurd naar de examencommissie om nog een extra poging te krijgen. Bespreek wel van te voren met jouw docent af wat er fout is gegaan om zeker te weten dat het vak dit keer wel gehaald wordt.",

    "Wat als je ziek bent geworden?": "Maak direct contact met jouw … Er zal op een ander moment weer een nieuwe examenpoging ingepland worden."

}


for question, answer in faq_questions.items():

    with st.expander(question):

        st.write(answer)


# Display chat history; Hier wordt de chat gestart zodat de bot begint met de tekst: “Welkom! Hoe kan ik je helpen met het examenreglement?”. Ook wordt hier de chat geschiedenis laten zien zodat je de chat kunt teruglezen, hiermee lijkt het meer op een soort WhatsApp gesprek en worden de vorige inputs niet meteen verwijderd.

for message in st.session_state.messages:

    with st.chat_message(message["role"], avatar="custom_bot_image.png" if message["role"] == "assistant" else None):

        st.markdown(message["content"])


# Chat input; Hier wordt de gebruikersinvoer verwerkt, met deze invoer wordt het antwoord verwerkt. Eerst wordt er gekeken of de vraag overeenkomt met de inhoud van het examenreglement. Dat wordt gedaan met de similarity search. Met de similarity search worden de vectoren van het pdf-bestand vergeleken met de vraag, dit zorgt ervoor dat de bot geen vragen kan beantwoorden die niet over het examenreglement gaan

if prompt := st.chat_input("Stel je vraag hier..."):

    # Display user message

    st.chat_message("user").markdown(prompt)

    st.session_state.messages.append({"role": "user", "content": prompt})


    # Check if the user input matches a FAQ option

    if prompt.lower() == "hoe kan ik vrijstelling aanvragen?":

        st.chat_message("assistant", avatar="custom_bot_image.png").markdown(

            "Om vrijstelling aan te vragen, moet je een verzoek indienen bij de examencommissie."

        )

        st.session_state.messages.append({"role": "assistant", "content": "Om vrijstelling aan te vragen, moet je een verzoek indienen bij de examencommissie."})

    elif prompt.lower() == "hoe kan ik extra tijd krijgen op mijn toets?":

        st.chat_message("assistant", avatar="custom_bot_image.png").markdown(

            "Extra tijd aanvragen kan via een formulier bij de onderwijsadministratie. Vraag naar de procedure op je opleiding."

        )

        st.session_state.messages.append({"role": "assistant", "content": "Extra tijd aanvragen kan via een formulier bij de onderwijsadministratie. Vraag naar de procedure op je opleiding."})

    else:

        # Voeg hier je standaard chatbot-verwerking toe

        try:

            if vector_store is None:

                st.error("Vector store niet beschikbaar")

            else:

                # Search for relevant context in the selected vectorstore

                results = vector_store.similarity_search_with_score(prompt, k=2)

                docs = []

                for r in results:

                    if r[1] < similarity_threshold:

                        docs.append(r[0])

                if not docs:

                    st.chat_message("assistant", avatar="custom_bot_image.png").markdown(

                        "Dit valt niet binnen het bereik van het examenreglement, dus ik kan hier geen antwoord op geven.")

                    st.session_state.messages.append({"role": "assistant",

                                                      "content": "Dit valt niet binnen het bereik van het examenreglement, dus ik kan hier geen antwoord op geven."})

                else:

                    # Use the context of the found documents

                    context = " ".join([doc.page_content for doc in docs])  # Use 'page_content'


                    # Create chat prompt with context

                    chat_prompt = ChatPromptTemplate.from_messages([

                        ("""Je bent een behulpzame assistent die alleen vragen beantwoordt over het examenreglement.


                        GEBRUIK DEZE STRUCTUUR:

1. Begin met een korte vraag die de hoofdvraag herhaalt

2. Schrijf 'Zo werkt het:'

3. Geef 3-5 korte stappen met streepjes (-)

4. Voeg 1-2 tips toe onder 'Tips:'


VOORBEELD ANTWOORD 1:

Wil je een vak overslaan omdat je het al hebt gehaald?


Zo werkt het:

- Ga naar je SLB'er

- Vertel welk vak je al hebt gehaald

- Je SLB'er helpt je met een formulier

- Je krijgt binnen 2 weken antwoord


Tips:

- Neem je oude cijferlijst mee

- Je SLB'er helpt je met invullen


VOORBEELD ANTWOORD 2:

Ben je ziek op de dag van je toets?


Zo werkt het:

- Bel de school voor 9 uur

- Vertel dat je ziek bent

- Je krijgt een nieuwe datum voor je toets


Tips:

- Bewaar het nummer van school in je telefoon

- Bel ook je SLB'er even


GEBRUIK DEZE EENVOUDIGE WOORDEN:

- toets (niet: examen, tentamen, assessment)

- SLB'er (niet: mentor, begeleider)

- tijd (niet: termijn, periode)

- formulier (niet: document, aanvraag)

- regels (niet: voorwaarden, eisen)

- antwoord (niet: uitslag, resultaat)


SCHRIJFTIPS:

- Schrijf zoals je praat

- Gebruik 'je' en 'jij'

- Maximaal 6 woorden per zin

- Gebruik geen moeilijke woorden

- Begin elke stap met een werkwoord

- Gebruik actieve zinnen


                        Context: {context}"""),

                        ("human", "{question}")

                    ])


                    # Generate answer with context

                    formatted_prompt = chat_prompt.format(context=context, question=prompt)

                    response = llm(formatted_prompt)  # OpenAI model returns a string directly


                    # Simplify and format the response

                   # response = validator.process_response(response)


                    # Display assistant message

                    st.chat_message("assistant", avatar="custom_bot_image.png").markdown(response)  # Use the string directly

                    st.session_state.messages.append({"role": "assistant", "content": response})


        except Exception as e:

            st.error(f"Er is iets misgegaan: {str(e)}") 