import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()


def get_llm():
    return ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "OPENAI_MODEL"),
        temperature=0,
    )


def get_embeddings():
    return OpenAIEmbeddings(
        model=os.getenv(
            "EMBEDDING_MODEL"
        )
    )
    
if __name__ == "__main__":
    llm = get_llm()
    embeddings = get_embeddings()
    print("LLM and embeddings initialized successfully.")