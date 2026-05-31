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

        try:
            # ---------------- SQL GENERATION ---------------- #
            chain = create_sql_query_chain(llm, db)

            raw_response = chain.invoke({
                "question": question + " Return ONLY SQL query. Use table name 'data'."
            })

            # ---------------- DEBUG ---------------- #
            #st.subheader("🔍 Raw LLM Response")
            #st.code(str(raw_response))

            # ---------------- CLEAN SQL ---------------- #
            #sql_query = str(raw_response)

            # Extract SQL from markdown blocks
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

            # ---------------- SHOW CLEAN SQL ---------------- #
            st.subheader("🧠 Generated SQL")
            st.code(sql_query, language="sql")

            # ---------------- EXECUTION ---------------- #
            result_df = pd.read_sql(sql_query, engine)

            st.subheader("📊 Result")
            st.dataframe(result_df)

            # ---------------- AUTO VISUALIZATION ---------------- #
            if len(result_df.columns) >= 2:

                x_col = result_df.columns[0]
                y_col = result_df.columns[1]

                fig = px.bar(
                    result_df,
                    x=x_col,
                    y=y_col,
                    title=f"{y_col} by {x_col}"
                )

                st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"❌ Error: {e}")
