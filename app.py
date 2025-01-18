import os
import streamlit as st
import logging
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain_community.retrievers import PineconeHybridSearchRetriever
from pinecone import Pinecone
from langchain_community.embeddings import HuggingFaceEmbeddings
from pinecone_text.sparse import BM25Encoder
from langchain_core.output_parsers import StrOutputParser
import nltk
import dotenv

dotenv.load_dotenv()

nltk.download('punkt_tab')

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY") 
HF_TOKEN = os.getenv("HF_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
index_name = "freecource-hybrid-search"


# Initialize Pinecone
try:
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(index_name)
except Exception as e:
    logging.error("Failed to initialize Pinecone: ", e)
    raise


# Initialize the embeddings and retriever
os.environ["HF_TOKEN"] = HF_TOKEN
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
bm25encoder = BM25Encoder().default()
retriever = PineconeHybridSearchRetriever(embeddings=embeddings, sparse_encoder=bm25encoder, index=index)


# Query the Pinecone index
def query_search(query):
    try:
        query_result = retriever.invoke(query)
        if "PineconeApiException" in query_result:
            logging.info(f"Error in search response: {query_result}")
            return None 
        return query_result
    except Exception as e:
        logging.error(f"Error during Pinecone search: {e}")
        return None  


# Define the template for the assistant
template = """
You are a smart search assistant for Analytics Vidhya's free courses. Your goal is to help users find the most relevant free courses based on their natural language queries.

Here is the information about the available courses:
{course_data}

When a user provides a search query, analyze the query to understand the intent and context.

User query: "{query}"

Respond with the following:

1. **If the query is a casual or non-course-related interaction (e.g., "hi", "hello", "how are you?"):**
   - Respond in a friendly and engaging manner.
   - Explain your capabilities, and behave like an intelligent chatbot
   - Keep the response as short and natural as possible.

2. **If the query is about company information:**
    - Respond with a brief explanation of the company and its offerings.
    - Provide a link to the company's website for more details -> https://www.analyticsvidhya.com/

3. **If relevant courses are found:**
   - Begin with a small, friendly explanation summarizing how the courses meet the user's needs.
   - List the relevant courses (if available) line by line, each including:
     - SNO. **Title**: [Course Title] End of Line
            - **Description**: [Summarized to 1-2 lines] End of Line
            - **Link to Enroll**: [Course Link] End of Line
   - In the end, suggest exploring more courses at https://courses.analyticsvidhya.com/pages/all-free-courses

4. **If no relevant courses match the query:**
   - Respond with a polite suggestion to explore more courses at https://courses.analyticsvidhya.com/pages/all-free-courses and an encouraging paragraph, such as:  
     "It seems like we don’t have a course that matches your exact query at the moment. However, Analytics Vidhya offers a variety of free courses across data science, machine learning, and analytics. Feel free to explore the full catalog to find something interesting."
   - You may also ask the user for more specific details or rephrase their query.
   
5. **In case of a casual conversation without course relevance:**
   - Respond in a friendly, chatbot-like manner that stays true to the platform's purpose, even when no course data is returned.
   - Provide helpful responses, ask engaging questions, and encourage exploration of the platform.

Keep your responses clear, concise, and conversational, adapting dynamically to the user's query.
"""

# Initialize the prompt template
prompt = PromptTemplate(
    input_variables=["course_data", "query"],
    template=template
)


# Initialize Groq Inferencer
try:
    os.environ["GROQ_API_KEY"] = GROQ_API_KEY
    llm = ChatGroq(
        model_name="llama-3.3-70b-versatile",
        temperature=0,
    )
except Exception as e:
    logging.error(f"Failed to initialize Groq: {e}")
    raise


# Initialize the chain
chain = prompt | llm | StrOutputParser()


# Query handling function
def retrive_answers(query):
    doc_search = query_search(query)
    if doc_search is None:
        doc_search = "No embeddings were found based on the user's query so chat in natural way while sticking to the tone, intent"
  
    inputs = {
        "course_data": doc_search,
        "query": query
    }
    try:
        answer = chain.invoke(inputs)
        return answer
    except Exception as e:
        logging.error(f"Error invoking chain for query {query}: {e}")
        return "There was an issue processing your query. Please try again later."


# Streamlit UI
st.set_page_config(
    page_title="Analytics Vidhya Assistant",
    page_icon="🤖",
    layout="wide",
)

st.title("🎓 Analytics Vidhya's Free Course's Assistant 🤖")
st.write(
    "Welcome to the **Analytics Vidhya Free Course Assistant**! 🧑‍💻 I'm here to help you find the most "
    "relevant free courses available on the **Analytics Vidhya** platform."
)

# Initialize the chat interface
with st.container():
    chat_con = st.container()
    query_con = chat_con.container()

    dg_con = chat_con.container(height=450)
    query = st.chat_input("Hi! I'm your assistant. How can I help you find a course today? 😊", disabled=False)

    if "messages" not in st.session_state:
        
        st.session_state.messages = [
            {"role": "ai", "content": "Hello! I'm your friendly assistant here to help you find the best free courses on Analytics Vidhya. Just tell me what you're looking for! 📚"}
        ]

    with dg_con:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                if message["role"] == "user":
                    st.markdown(message['content'])
                else:
                    st.markdown(message['content'])

        if query:
            st.session_state.messages.append({"role": "user", "content": query})
            with st.chat_message("user"):
                st.markdown(query)

            ai_response = retrive_answers(query)

            st.session_state.messages.append({"role": "ai", "content": ai_response})
            with st.chat_message("ai"):
                st.markdown(ai_response)
