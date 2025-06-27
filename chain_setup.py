from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.memory import ConversationBufferMemory
from embedding import CustomEmbeddingFunction
import os
import chainlit as cl
from file_processor import process_file

def initialize_chain(file_path):
    # 初始化文本分割器
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

    # 读取文件内容
    text = process_file(file_path)

    # 检查文本是否为空
    if not text:
        raise ValueError(f"文件 {file_path} 内容为空，无法处理。")

    # 将文本分割成小块
    texts = text_splitter.split_text(text)

    # 为每个文本块创建元数据
    metadatas = [{"source": f"{i}-pl"} for i in range(len(texts))]

    # 使用自定义的嵌入函数创建 Chroma 向量存储
    embedding_function = CustomEmbeddingFunction()

    # 每次调用时都创建一个新的向量存储
    docsearch = Chroma.from_texts(
        texts=texts,
        embedding=embedding_function,
        metadatas=metadatas,
        collection_name=f"doc_collection_{hash(file_path)}",  # 唯一集合名称
        persist_directory=None  # 禁用持久化或指定唯一路径
    )

    # 初始化聊天消息历史记录
    message_history = ChatMessageHistory()

    # 初始化对话缓冲区内存
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        chat_memory=message_history,
        return_messages=True,
    )

    # 创建对话检索链
    chain = ConversationalRetrievalChain.from_llm(
        ChatOpenAI(
            model_name= str(os.environ.get("LLM_MODEL")), 
            temperature=0, 
            streaming=True,
            base_url=str(os.environ.get("LLM_BINDING_HOST")),  # 本地模型的 API 地址
            api_key=str(os.environ.get("OPENAI_API_KEY"))  # 本地模型的 API 密钥
        ),
        chain_type="stuff",
        retriever=docsearch.as_retriever(),  # 设置检索器
        memory=memory,  # 设置内存
        return_source_documents=True,  # 返回源文档
    )

    return chain