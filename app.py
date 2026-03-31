import streamlit as st
from dotenv import load_dotenv

# --- PRO IMPORT: Importing our own modules ---
from ingest import ingest_document
from query import get_response
from quiz_generator import generate_assessment

load_dotenv()

st.set_page_config(page_title="AutoAssess AI", page_icon="🎓")
st.title("🎓 AutoAssess AI")
st.write("The app is starting... please wait while models load.")

if "db" not in st.session_state:
    st.session_state.db = None

# SIDEBAR: Only handles the UI for uploading
with st.sidebar:
    st.header("1. Upload Material")
    uploaded_file = st.file_uploader("Upload a Training PDF", type="pdf")
    
    if uploaded_file and st.button("Process Document"):
        with st.spinner("Analyzing..."):
            with open("temp.pdf", "wb") as f:
                f.write(uploaded_file.read())
            
            # CALLING THE INGESTION MODULE
            st.session_state.db = ingest_document("temp.pdf")
            st.success("Document Ready!")

# MAIN TABS: Only handles the interaction logic
if st.session_state.db:
    tab1, tab2 = st.tabs(["💬 Chat", "📝 Quiz"])

    with tab1:
        query = st.text_input("Ask a question:")
        if query:
            # CALLING THE QUERY MODULE
            answer, sources = get_response(st.session_state.db, query)
            st.write(answer)

    with tab2:
        if st.button("Generate Quiz"):
            # CALLING THE ASSESSMENT MODULE
            quiz = generate_assessment(st.session_state.db)
            for i, q in enumerate(quiz):
                st.write(f"**Q{i+1}: {q['question']}**")
                st.radio("Options", q['options'], key=f"quiz_{i}")
else:
    st.info("Please upload a document to begin.")