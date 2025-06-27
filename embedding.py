from langchain_openai import OpenAIEmbeddings
import os

class CustomEmbeddingFunction:
    def __init__(self):
        # 使用环境变量中的 API 密钥、模型名称和基础 URL 初始化 OpenAIEmbeddings
        self.embedding_model = OpenAIEmbeddings(
            openai_api_key=str(os.environ.get("EMBEDDING_BINDING_API_KEY")),
            model=str(os.environ.get("EMBEDDING_MODEL")),
            openai_api_base=str(os.environ.get("EMBEDDING_BINDING_HOST"))  # 自定义基础 URL
        )

    def embed_documents(self, texts):
        if not texts:
            print("文本列表为空，无法生成嵌入向量。")
            raise ValueError("文本列表为空。请检查输入。")
        
        # 使用 OpenAIEmbeddings 的 embed_documents 方法生成嵌入向量
        embeddings = self.embedding_model.embed_documents(texts)
        
        # 确保返回的嵌入向量不为空
        if not embeddings:
            raise ValueError("嵌入向量列表为空。请检查嵌入服务的响应。")
        
        return embeddings

    def embed_query(self, text):
        # 使用 OpenAIEmbeddings 的 embed_query 方法生成单个文本的嵌入向量
        embedding = self.embedding_model.embed_query(text)
        
        if not embedding:
            raise ValueError("嵌入向量为空。请检查嵌入服务的响应。")
        
        return embedding