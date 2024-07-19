from langchain_experimental.agents import create_csv_agent
from langchain_community.llms import OpenAI
from langchain_community.chat_models import ChatOpenAI
from dotenv import load_dotenv
import os
import streamlit as st
import pandas as pd
from io import StringIO

def main():
    load_dotenv()

    # Check if the API key is set in the environment
    if "OPENAI_API_KEY" not in os.environ:
        raise ValueError("The OPENAI_API_KEY environment variable is not set.")

    st.set_page_config(page_title="Ask your CSV")
    st.header("Ask your CSV a question")

    model_options = [
        "gpt-3.5-turbo-instruct",
        "gpt-4o-mini",
        "gpt-4o",
        "gpt-4",
        "gpt-4-turbo",
        "gpt-3.5-turbo"
    ]
    
    selected_model = st.selectbox("Select a model", model_options)

    csv_file = st.file_uploader("Upload a CSV file", type="csv")
    if csv_file is not None:
        csv_data = pd.read_csv(csv_file)
        csv_string = csv_data.to_csv(index=False)
        csv_file_like = StringIO(csv_string)

        user_question = st.text_input("Ask a question about your CSV: ")

        if user_question:
            with st.spinner(text="In progress..."):
                # Determine which class to use based on the selected model
                if selected_model == "gpt-3.5-turbo-instruct":
                    llm = OpenAI(api_key=os.environ["OPENAI_API_KEY"], model_name=selected_model, temperature=0)
                else:
                    llm = ChatOpenAI(api_key=os.environ["OPENAI_API_KEY"], model=selected_model, temperature=0)
                
                agent = create_csv_agent(
                    llm,
                    csv_file_like,
                    verbose=True,
                    allow_dangerous_code=True
                )
                response = agent.run(user_question)
                st.write(response)

if __name__ == "__main__":
    main()
