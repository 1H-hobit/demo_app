import os
import requests
import json
from typing import Literal, Optional, Generator, List, Dict, Callable, Any
from dataclasses import dataclass, field
from dotenv import load_dotenv
load_dotenv()  # 加载当前目录下的 .env 文件

@dataclass
class QueryConfig:
    """LightRAG 查询配置参数"""
    mode: Literal["local", "global", "hybrid", "naive", "mix", "bypass"] = "mix"
    only_need_context: bool = False  # 修正为布尔类型
    only_need_prompt: bool = False   # 修正为布尔类型
    response_type: str = "Bullet Points"
    #stream: bool = True  # 默认启用流式
    top_k: int = field(default_factory=lambda: int(os.getenv("TOP_K", "60")))
    max_token_for_text_unit: int = field(default_factory=lambda: int(os.getenv("MAX_TOKEN_TEXT_CHUNK", "4000")))
    max_token_for_global_context: int = field(default_factory=lambda: int(os.getenv("MAX_TOKEN_RELATION_DESC", "4000")))
    max_token_for_local_context: int = field(default_factory=lambda: int(os.getenv("MAX_TOKEN_ENTITY_DESC", "4000")))
    hl_keywords: List[str] = field(default_factory=list)
    ll_keywords: List[str] = field(default_factory=list)
    #conversation_history: List[Dict[str, str]] = field(default_factory=list)
    history_turns: int = field(default_factory=lambda: int(os.getenv("HISTORY_TURNS", "3")))


class LightRAGClient:
    def __init__(self, base_url="http://localhost:9721", api_key=None, default_config=None):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.default_config = default_config or QueryConfig()
        
        # 改用请求头传递API密钥
        if api_key:
            self.session.headers.update({"X-API-Key": api_key})  # 修正认证头部


    def query(
        self,
        question: str,
        config: Optional[QueryConfig] = None
    ) -> Generator[str, None, None]:
        """
        执行查询请求
        
        :param question: 用户问题
        :param config: 自定义查询配置
        """
        # 合并配置
        final_config = self._merge_configs(config)
        
        # 构建请求体
        payload = {
            "query": question,
            **self._config_to_dict(final_config),
        }

        # 发送请求
        with self.session.post(
            url=f"{self.base_url}/query/stream",
            json=self._clean_payload(payload),
            stream=True,
            timeout=30
        ) as response:
            yield from self._handle_response(response)

    def _merge_configs(self, custom_config: Optional[QueryConfig]) -> QueryConfig:
        """合并默认配置和自定义配置"""
        if not custom_config:
            return self.default_config
            
        merged = QueryConfig()
        for field in vars(merged):
            default_val = getattr(self.default_config, field)
            custom_val = getattr(custom_config, field, default_val)
            setattr(merged, field, custom_val)
        return merged

    def _config_to_dict(self, config: QueryConfig) -> Dict:
        """转换配置对象为字典"""
        return {
            "mode": config.mode,
            "only_need_context": config.only_need_context,
            "only_need_prompt": config.only_need_prompt,
            "response_type": config.response_type,
            #"stream": config.stream,
            "top_k": config.top_k,
            "max_token_for_text_unit": config.max_token_for_text_unit,
            "max_token_for_global_context": config.max_token_for_global_context,
            "max_token_for_local_context": config.max_token_for_local_context,
            "hl_keywords": config.hl_keywords,
            "ll_keywords": config.ll_keywords,
            #"conversation_history": config.conversation_history[-2*config.history_turns:],
            #"ids": config.ids,
            #"user_prompt": config.user_prompt
        }

    def _clean_payload(self, payload: Dict) -> Dict:
        """清理空值参数"""
        return {k: v for k, v in payload.items() if v is not None}

    def _handle_response(self, response: requests.Response) -> Generator[str, None, None]:
            """处理流式响应"""
            response.raise_for_status()
            
            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode('utf-8').strip()
                    try:
                        data = json.loads(decoded_line)
                        if "response" in data:
                            # 直接逐字符返回
                            for char in data["response"]:
                                yield char
                    except json.JSONDecodeError:
                        continue

