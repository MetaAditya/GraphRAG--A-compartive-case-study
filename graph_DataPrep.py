from dotenv import load_dotenv
import os
from langchain_neo4j import Neo4jGraph

from langchain_core.runnables import (
    RunnableBranch,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts.prompt import PromptTemplate
from pydantic import BaseModel, Field
# from langchain_core.pydantic_v1 import BaseModel, Field
from typing import Tuple, List
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import WikipediaLoader
from langchain_text_splitters import TokenTextSplitter
from langchain_openai import ChatOpenAI
from langchain_experimental.graph_transformers import LLMGraphTransformer

from langchain_neo4j import Neo4jVector
from langchain_openai import OpenAIEmbeddings
from langchain_neo4j.vectorstores.neo4j_vector import remove_lucene_chars

load_dotenv()

AURA_INSTANCENAME = os.environ["AURA_INSTANCENAME"]
NEO4J_URI = os.environ["NEO4J_URI"]
NEO4J_USERNAME = os.environ["NEO4J_USERNAME"]
NEO4J_PASSWORD = os.environ["NEO4J_PASSWORD"]


AUTH = (NEO4J_USERNAME, NEO4J_PASSWORD)


chat = ChatOpenAI(temperature=0, model="gpt-4o-mini")


kg = Neo4jGraph(
    url=NEO4J_URI,
    username=NEO4J_USERNAME,
    password=NEO4J_PASSWORD,
 )


#Load the Data from Wikipedia
raw_documents = WikipediaLoader(query="The story of Indian Economy").load()

print("Raw Data Extracted from Wikipedia")

#Load specific topics from wikipedia
to_split_docs=[raw_documents[0], #Economy of India
               raw_documents[1], #Economic history of India
               raw_documents[3], #1991 Indian economic crisis
               raw_documents[4], #Economy of South Asia
               raw_documents[5], #Indian National Congress
               raw_documents[8]] #Economy of the British Empire



#chunking
text_splitter = TokenTextSplitter(chunk_size=512, chunk_overlap=24)
documents = text_splitter.split_documents(to_split_docs)

print("Raw Data Chunked into smaller pieces")


#Transform th data to graph data
llm_transformer = LLMGraphTransformer(llm=chat)
graph_documents = llm_transformer.convert_to_graph_documents(documents)

print("graph Data created from the text data using LLMGraphTransformer")

#Add the graph data to Graph DB on Neo4j
res = kg.add_graph_documents(
    graph_documents,
    include_source=True,
    baseEntityLabel=True,
)



print("graph Data  added to Neo4j Graph Database")

#Create the Embeddings
#Embeddings are created on Document, the text data associated with entire text corpus

vector_index = Neo4jVector.from_existing_graph(
    OpenAIEmbeddings(),
    search_type="hybrid",
    node_label="Document",
    text_node_properties=["text"],
    embedding_node_property="embedding",
)


print("graph Data  added to Neo4j Graph Database is vectorized using OpenAIEmbeddings and stored as a Vector Index in Neo4j")

print("Data Sucessfully added to Neo4j Graph Database and Vector Index created")