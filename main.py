from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

# 1. Load file
loader = TextLoader("data/notes.txt")
documents = loader.load()

# 2. Split text into chunks
splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=20)
docs = splitter.split_documents(documents)

# 3. Convert to embeddings
embeddings = OpenAIEmbeddings()

# 4. Store in vector DB
db = FAISS.from_documents(docs, embeddings)

# 5. Create retriever
retriever = db.as_retriever()

# 6. LLM
llm = ChatOpenAI(model_name="gpt-4")

# 7. RAG Chain
qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# 8. Ask question
query = input("Ask: ")
answer = qa.run(query)

print("\nAnswer:", answer)
