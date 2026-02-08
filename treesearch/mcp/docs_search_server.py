from pathlib import Path
from typing import Literal, get_args

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from mcp.server.fastmcp import FastMCP
from config import get_config

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger("rag")
load_dotenv()
mcp = FastMCP("Documentation search")

config = get_config()

if config.local_llm.embedding_mode == "api":
    
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")

else:
    embedding_model = OpenAIEmbeddings(
        model=config.local_llm.local_embedding_model,
        base_url=config.local_llm.base_url,
        api_key="not needed",
        check_embedding_ctx_length=False,
            )


VECTOR_STORES_BASE_PTH = Path("./ragEmbeddings")
VECTOR_STORE_NAMES = Literal["omnirec", "lenskit", "recbole"]


def load_vector_store(name: str) -> FAISS:
    vector_store_pth = VECTOR_STORES_BASE_PTH / name
    if not vector_store_pth.exists():
        raise FileNotFoundError(
            f"Could not read in store at '{vector_store_pth}'! Did you generate or download the embeddings first?"
        )
    return FAISS.load_local(
        str(vector_store_pth),
        embedding_model,
        allow_dangerous_deserialization=True,
    )


VECTOR_STORES: dict[str, FAISS] = {
    name: load_vector_store(name) for name in get_args(VECTOR_STORE_NAMES)
}


@mcp.tool()
def documentation_query(
    library: VECTOR_STORE_NAMES, query: str, k: int = 4
) -> dict:
    logger.info("RAG query called")
    logger.info("Library: %s", library)
    logger.info("Query: %s", query)
    logger.info("Top-k: %d", k)

    results = VECTOR_STORES[library].similarity_search_with_score(query, k=k)

    logger.info("Retrieved %d results", len(results))

    formatted_results = []
    for i, (doc, score) in enumerate(results, 1):
        source = doc.metadata.get("source", "Unknown")

        logger.info("Result %d", i)
        logger.info("  Source: %s", source)
        logger.info("  Score (distance): %.6f", score)
        logger.info("  Preview: %s", doc.page_content[:200].replace("\n", " "))

        formatted_results.append(
            {
                "source": source,
                "score": float(score),
                "text": doc.page_content,
                "metadata": doc.metadata,
            }
        )

    response = {
        "library": library,
        "query": query,
        "results": formatted_results,
    }

    logger.info("RAG response ready and returned to agent")
    return response



def main():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
