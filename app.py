import streamlit as st
import pandas as pd
import os
import re
from dotenv import load_dotenv
from sqlalchemy import create_engine
from langchain_groq import ChatGroq
from langchain_community.utilities import SQLDatabase
from langchain.chains import create_sql_query_chain
import plotly.express as px

# ---------------- LOAD ENV ---------------- #
load_dotenv()
groq_key = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY")

if not groq_key:
    st.error("⚠️ API key missing")
    st.stop()

# ---------------- LLM ---------------- #
llm = ChatGroq(
    groq_api_key=groq_key,
    model="llama-3.3-70b-versatile"
)

# ---------------- UI ---------------- #
st.set_page_config(page_title="AI SQL Data Analyst", layout="wide")
st.title("🧠 AI SQL Data Analyst Agent")

# ---------------- FILE UPLOAD ---------------- #
file = st.file_uploader("Upload CSV", type=["csv"])

if file:
    df = pd.read_csv(file)

    st.subheader("📄 Data Preview")
    st.dataframe(df.head())

    # ---------------- CREATE SQL DB ---------------- #
    engine = create_engine("sqlite:///data.db", echo=False)
    df.to_sql("data", engine, if_exists="replace", index=False)

    db = SQLDatabase(engine)

    # ---------------- USER INPUT ---------------- #
    question = st.text_input("Ask a question")

    submit = st.button("🚀 Submit Query")

    if submit and question:

        # ---------------- SQL GENERATION ---------------- #
        chain = create_sql_query_chain(llm, db)

        raw_response = chain.invoke({
            "question": question + """
            Return ONLY a valid SQLite SQL query.
            Use table name 'data'.
            Do not add LIMIT unless the user asks for top N records.
            Do not include explanations, markdown, SQLQuery, SQLResult, or Answer.
            """
        })

        # ---------------- CLEAN SQL ---------------- #
        sql_query = str(raw_response)

        # Handle ```sql ... ``` blocks
        match = re.search(
            r"```sql\s*(.*?)\s*```",
            sql_query,
            re.DOTALL | re.IGNORECASE
        )

        if match:
            sql_query = match.group(1)

        # Remove markdown remnants
        sql_query = sql_query.replace("```sql", "")
        sql_query = sql_query.replace("```", "")

        # Remove LangChain labels
        if "SQLQuery:" in sql_query:
            sql_query = sql_query.split("SQLQuery:")[-1]

        if "SQLResult:" in sql_query:
            sql_query = sql_query.split("SQLResult:")[0]

        if "Answer:" in sql_query:
            sql_query = sql_query.split("Answer:")[0]

        sql_query = sql_query.strip()

        st.subheader("🧠 Generated SQL")
        st.code(sql_query)

        # ---------------- EXECUTION ---------------- #
        try:
            result_df = pd.read_sql(sql_query, engine)

            st.subheader("📊 Result")
            st.dataframe(result_df)

            # ---------------- AUTO VISUALIZATION ---------------- #
            if len(result_df.columns) >= 2:
                col1 = result_df.columns[0]
                col2 = result_df.columns[1]

                fig = px.bar(result_df, x=col1, y=col2)
                st.plotly_chart(fig)

        except Exception as e:
            st.error(f"❌ Error: {e}")
