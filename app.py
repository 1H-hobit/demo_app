import chainlit as cl
from chainlit.input_widget import Select, Switch, Slider
from chainlit.sync import run_sync
from openai import AsyncOpenAI
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from PIL import Image
import torch
import os
import json
# from minirag import MiniRAG, QueryParam
# from minirag.llm.openai import openai_complete_if_cache, openai_embed
# from minirag.utils import EmbeddingFunc
import requests
import chainlit as cl
from chainlit.data.sql_alchemy import SQLAlchemyDataLayer
from local_storage import MinIOStorageClient
import asyncio
from chain_setup import initialize_chain
#from ocr_utils import process_image_ocr
from file_processor import process_file
import ast
from functions.FunctionManager import FunctionManager
import inspect
import importlib
from openai.types.chat.chat_completion_message import ChatCompletionMessage  # 假设文件在相同目录
import re
from playwright.async_api import async_playwright
import urllib.parse
import random
from transformers import BitsAndBytesConfig
from f5_tts.api import F5TTS
from pathlib import Path
import uuid
from modelscope import AutoModel, AutoTokenizer
import torchvision.transforms as T
from transformers import TextIteratorStreamer
from threading import Thread
from torchvision.transforms.functional import InterpolationMode  # 图像插值模式
from ligrag_function_class import QueryConfig , LightRAGClient
import opencc
from decord import VideoReader, cpu
import numpy as np
from mcp import ClientSession

@cl.data_layer
def get_data_layer():
    storage_provider = MinIOStorageClient(
        endpoint = "localhost:9000",  # MinIO 服务器地址
        access_key = os.environ.get("data_layer_access_key"),
        secret_key = os.environ.get("data_layer_secret_key"),
        bucket_name= "my-bucket",  # Bucket 名称
    )
    return SQLAlchemyDataLayer(
        conninfo="postgresql+asyncpg://postgres:kobe@localhost:5432/chainlit_db",
        storage_provider=storage_provider
    )

# ======================== tiktoken_cache_dir配置初始化 ========================
# 使用原始字符串避免转义问题
tiktoken_cache_dir = r"D:\chainlit\chainlit-datalayer\demo_app\tiktoken_cache"

os.makedirs(tiktoken_cache_dir, exist_ok=True)  # 确保目录存在

cache_file_path = os.path.join(tiktoken_cache_dir, "fb374d419588a4632f3f557e76b4b70aebbca790")

if not os.path.exists(cache_file_path):
    with open(cache_file_path, 'w') as f:
        f.write('')  # 创建一个空文件

assert os.path.exists(cache_file_path), f"缓存文件 {cache_file_path} 不存在，请检查路径或手动创建该文件。"

TF_ENABLE_ONEDNN_OPTS=0
torch.backends.cuda.enable_flash_sdp(False)  # 禁用 flash attention

# ======================== 模型配置初始化 ========================
client = AsyncOpenAI(
    base_url= os.environ.get("LLM_BINDING_HOST"),
    api_key = os.environ.get("OPENAI_API_KEY")
)


# ======================== 全局变量声明 ========================
# 在文件开头（函数外部）声明全局变量
image_processor = None
tts = None
image_model = None
is_stop = False
is_tools = False
is_mcp_tools = False
ocr_model = None
ocr_tokenizer = None
num_patches_list = None
video_prefix = ''
MAX_ITER = 100
# 配置项
TEMP_DIR = Path("tts_temp")
TEMP_DIR.mkdir(exist_ok=True)
valid_modes = ["text_mode", "image_mode", "knowledge_mode", "web_search_mode", "qa_mode", "image_ocr_ai_mode","calling_tools", "voice_mode", "mcp_calling_tools"]

# ======================== commands消息处理 ========================
commands = [
    {"id": "file_upload", "icon": "Upload", "description": "上传文件"},
    {"id": "Tool_reset_memory", "icon": "memory-stick", "description": "工具重置记忆"},
    {"id": "Code_Runner", "icon": "square-terminal", "description": "代码运行器"},
]

# 语音模型加载方式为懒加载
def get_tts():
    if not hasattr(cl.user_session, "tts"):
        # 每个用户会话只加载一次模型
        cl.user_session.tts = F5TTS(
            model="F5TTS_v1_Base",
            ckpt_file=r"D:\chainlit\models\F5TTS_v1_Base\model_1250000.safetensors"
        )
    return cl.user_session.tts

# ======================== OCR模型加载 ========================
# 配置参数
MODEL_NAME = r"D:\chainlit\models\InternVL3-14B-Instruct"
# ======================== OCR模型加载 ========================
def load_ocr_model():
    global ocr_model
    global ocr_tokenizer

    ocr_model = AutoModel.from_pretrained(
        MODEL_NAME,
        #load_in_8bit=True,
        load_in_4bit=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        device_map="auto",
        trust_remote_code=True).eval()
    
    
    ocr_tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True
    )
    
    print("OCR模型已就绪, 请上传图片开始识别")


# ======================== MiniRAG初始化 ========================
# async def openai_llm_complete(prompt, max_tokens=1024, **kwargs):
#     return await openai_complete_if_cache(
#         base_url=os.environ.get("LLM_BINDING_HOST"),
#         api_key=os.environ.get("OPENAI_API_KEY"),
#         model=os.environ.get("LLM_MODEL"),
#         prompt=prompt,
#         max_tokens=max_tokens,
#         **kwargs
#     )

# async def openai_embedding_func(texts):
#     return await openai_embed(
#         texts=texts,
#         model=os.environ.get("EMBEDDING_MODEL"),
#         base_url=os.environ.get("EMBEDDING_BINDING_HOST"),
#         api_key=os.environ.get("EMBEDDING_BINDING_API_KEY"),
#     )

# rag = MiniRAG(
#     working_dir=r"D:\chainlit\chainlit-datalayer\demo_app\RAG_working_dir",
#     llm_model_func=openai_llm_complete,
#     llm_model_max_token_size=1024,
#     llm_model_name=os.environ.get("LLM_MODEL"),
#     embedding_func=EmbeddingFunc(
#         embedding_dim=int(os.environ.get("EMBEDDING_DIM")),
#         max_token_size=1024,
#         func=openai_embedding_func
#     ),
# )

# ======================== 加载插件功能 ========================
plugin_dirs = [
    d for d in os.listdir('plugins')
    if os.path.isdir(os.path.join('plugins', d)) and d != '__pycache__'
]

functions = []
for dir in plugin_dirs:
    try:
        with open(f'plugins/{dir}/config.json', 'r') as f:
            config = json.load(f)
        enabled = config.get('enabled', True)
    except FileNotFoundError:
        enabled = True

    if not enabled:
        continue

    module = importlib.import_module(f'plugins.{dir}.functions')
    functions.extend([
        obj for name, obj in inspect.getmembers(module) if inspect.isfunction(obj)
    ])

function_manager = FunctionManager(functions=functions)
tools = [{"type": "function", "function": fn} for fn in function_manager.generate_functions_array()]

functions_json = json.dumps(
    [fn["function"] for fn in tools],
    indent=2,
    ensure_ascii=False
)

# ======================== 模型系统使用提示 ========================
language = os.environ.get("SUMMARY_LANGUAGE") or "chinese"
# 修改系统提示的结构，每次生成时强制包含工具信息
def get_system_message(functions_json , language):
    return f"""
            您是一个高级人工智能助手，通过工具调用与用户交互。
            当前可用工具（必须严格按规范调用）：
            {functions_json}
            作为开放解释器，您是能执行代码完成任何目标的世界级程序员：
            1. 执行代码时拥有用户机器的完全权限
            2. 可以安装新包、访问互联网
            3. 遇到失败会自动重试
            4. 当用户提到一个文件名时，指的是您当前目录中现有的文件，可以当前路径处理文件。

            请始终按以下流程处理：
            1. 分析需求并制定计划，尽量简化计划的步骤。
            2. 选择合适工具或直接执行代码，不要试图在一个代码块中完成所有事情，这一点至关重要。
            您应该尝试某件事，打印有关它的信息，然后从那里继续进行微小的、明智的步骤。
            您永远不会在第一次尝试时就成功，而尝试一次完成所有事情通常会导致您看不到的错误。
            3. 严格使用合法JSON格式调用工具参数
            请使用{language}交流
            """
system_message = get_system_message(functions_json , language)  # 每次生成时重新获取


# ======================== 查看有所有函数与相关参数 ========================
# 将 tools 转换为 JSON 格式
tools_json = json.dumps(tools, indent=4)
# 打印 JSON 格式的 tools
#print ("tools_json:\n",tools_json)


# ======================== 设置默认工作目录 ========================
# 打印当前目录
#print("当前目录:", os.getcwd())
# 设置默认保存路径
os.chdir(os.getcwd())

# ======================== 用户认证 ========================
@cl.password_auth_callback
def auth_callback(username: str, password: str):
    if (username, password) == ("admin", "admin"):
        return cl.User(
            identifier="admin", 
            metadata={"role": "admin", "provider": "credentials"}
        )
    return None

# ======================== on_chat_start ========================
@cl.on_chat_start
async def on_chat_start():
    """聊天开始时初始化设置"""
    # 初始化默认会话数据

    cl.user_session.set("message_history", [{
        "role": "system",
        "content": system_message
    }])

    default_mode = "text_mode"
    cl.user_session.set("processing_mode", default_mode)
    cl.user_session.set("conversation_history", [])
    cl.user_session.set("resume_data", None)
    current_mode = default_mode
    await update_mode_selector(current_mode)  # 同步更新设置面板
    

# ======================== on_chat_resume ========================
@cl.on_chat_resume
async def on_chat_resume(conversation: dict):

    cl.user_session.set("resume_data", None)  
    cl.user_session.set("resume_data", conversation)
    session_id = conversation.get("id")
    print(f"恢复会话ID: {session_id}")
    if resume_data := load_resume_data(session_id):
        saved_mode = resume_data.get("processing_mode")
        
        current_mode = saved_mode if saved_mode in valid_modes else "text_mode"
        if current_mode == "calling_tools":
            await cl.context.emitter.set_commands(commands)
        else:
            await cl.context.emitter.set_commands([])

        await update_mode_selector(current_mode)  # 同步更新设置面板

def load_resume_data(session_id: str):
    save_dir = os.path.join(os.path.dirname(__file__), "conversation")  # 获取脚本所在目录
    file_path = os.path.join(save_dir, f"conversation_{session_id}.json")
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None
         
# ======================== on_chat_end ========================
@cl.on_chat_end
async def on_chat_end():
    """会话结束自动保存"""
    session_id = None
    conversation = cl.user_session.get("resume_data")
    if conversation:
        session_id = conversation.get("id")
        print(f"会话结束自动保存ID: {session_id}")
    #print (f"会话结束自动保存数据: {conversation}")
    if session_id:
        save_resume_data(session_id, {
            "resume_data": cl.user_session.get("resume_data"),
            "processing_mode": cl.user_session.get("processing_mode")  # 新增模式保存
        })
    cl.user_session.set("resume_data", None)  # 清除会话数据


def save_resume_data(session_id: str, data: dict):
    save_dir = os.path.join(os.path.dirname(__file__), "conversation")  # 获取脚本所在目录
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, f"conversation_{session_id}.json")
    # Save the data
    with open(file_path, "w") as f:
        json.dump(data, f)


# ======================== on_stop ========================
@cl.on_stop
async def stop_chat():
    global is_stop
    is_stop = True
    print("会话已终止")


# ======================== on_message_tools消息处理 ========================
async def on_message_tools(message: cl.Message):
    global is_stop
    cur_iter = 0
    tool_call_id = True
    user_message = message.content
    cl.user_session.set("user_message", user_message)
    message_history = cl.user_session.get("message_history", [])
    message_history.append({"role": "user", "content": user_message})

    # 消息历史管理（保持原有逻辑）
    if len(message_history) > 20:
        message_history = [message_history[0]] + message_history[-39:]

    # 命令处理（保持原有逻辑）
    if message.command == "file_upload":
        file_upload_user_message = cl.user_session.set("user_message", "上载文件")
        result = await function_manager.call_function("need_file_upload",  {"user_message": file_upload_user_message})
        #message_history.append({"role": "assistant", "content": result})
        message_history = [{
            "role": "system", 
            "content": get_system_message(functions_json, language)
        }]
        await cl.Message(content=f"处理结果:\n{result}", language="json").send()
        tool_call_id = await tool_calls(message_history)


    if message.command == "Tool_reset_memory":
        message_history = [{
            "role": "system", 
            "content": get_system_message(functions_json, language)
        }]
        await asyncio.sleep(0.1)  # 添加短暂延迟确保UI更新
        await cl.Message(content="记忆已重置").send()

    # 命令处理（保持原有逻辑）
    if message.command == "Code_Runner":
        #user_message = f"```py\n{user_message}\n```"
        await cl.Message(content=user_message , language="python").send()
        result = await function_manager.call_function("python_exec", {"code": user_message})
        message_history.append({"role": "assistant", "content": result})
        await cl.Message(content=f"处理结果:\n{result}" , language="json").send()
        tool_call_id = await tool_calls(message_history)

    while cur_iter < MAX_ITER and not is_stop:
        
        # 每次请求前重新注入系统提示
        current_system_message = get_system_message(functions_json, language)
        
        # 检查 enhanced_history 是否已经包含系统消息
        system_message_exists = any(
            msg["role"] == "system" and msg["content"] == current_system_message
            for msg in message_history
        )
        
        # 如果没有系统消息，则添加
        if not system_message_exists:
            enhanced_history = [
                {"role": "system", "content": current_system_message},
                *[msg for msg in message_history if msg["role"] != "system"]
            ]
        else:
            enhanced_history = message_history

        if is_stop:
            is_stop = False
            break

        #如果 tool_call_id 无效或不存在，则终止循环。
        if not tool_call_id:
            break

        #print (enhanced_history)
        tool_call_id = await tool_calls(enhanced_history)
        
        cur_iter += 1
        continue


# ======================== main消息处理 ========================
async def main(message: cl.Message):
    # 获取当前模式
    current_mode = cl.user_session.get("processing_mode")
    print(f"当前模式: {current_mode}")  # 添加调试输出
    global is_tools
    global is_mcp_tools

    # 模式有效性验证
    if current_mode not in valid_modes:
        await cl.Message("⚠️ 检测到无效模式，已重置为文本处理").send()
        current_mode = "text_mode"
        cl.user_session.set("processing_mode", current_mode)
    
    if current_mode != "calling_tools":  
        is_tools = False
    else:
        is_tools = True

    if current_mode != "mcp_calling_tools":  
        is_mcp_tools = False
    else:
        is_mcp_tools = True

    # 根据模式处理消息
    if current_mode == "image_mode":
        # 图片处理逻辑
        if not message.elements or not message.elements[0].mime.startswith("image"):
            await cl.Message("❌ 图片模式需要上传图片文件").send()
            return
        #await update_mode_selector(current_mode)  # 同步更新设置面板
        await handle_image_input(message, cl.user_session.get("conversation_history", []))


    elif current_mode == "knowledge_mode":
        # 知识库处理逻辑
        #await update_mode_selector(current_mode)  # 同步更新设置面板
        await handle_post_upload_actions(message.content)

    
    elif current_mode == "web_search_mode":
        # 联网搜索处理逻辑
        #await update_mode_selector(current_mode)  # 同步更新设置面板
        await handle_web_search(message.content)

    
    elif current_mode == "image_ocr_ai_mode":
        # 图片OCR处理逻辑
        await handle_image_ai_ocr(message)

    # elif current_mode == "image_ocr_mode":
    #     # 图片OCR处理逻辑
    #     #await update_mode_selector(current_mode)  # 同步更新设置面板
    #     await handle_image_ocr(message)


    elif current_mode == "qa_mode":
        # 文件问答处理逻辑
        #await update_mode_selector(current_mode)  # 同步更新设置面板
        await qa_text_input(message.content)


    elif current_mode == "voice_mode":
        # 语音问答处理逻辑
        #await update_mode_selector(current_mode)  # 同步更新设置面板
        await handle_voice_mode(message)

    else:
        # 默认文本处理
        #await update_mode_selector(current_mode)  # 同步更新设置面板
        await handle_text_input(
            cl.user_session.get("conversation_history", []),
            message.content
        )
        
    cl.user_session.set("processing_mode", current_mode)
    # 获取当前对话的所有消息
    context = cl.chat_context.to_openai()
    #print(context)  # 打印对话上下文
    cl.user_session.set("conversation_history", context) # 保存对话历史   


# ======================== 主要运行chainlit消息处理 ========================
@cl.on_message
async def run_conversation(message: cl.Message):
    global is_tools
    global is_mcp_tools
    if is_tools:
        await on_message_tools(message)

    print ("is_mcp_tools:\n",is_mcp_tools)

    if is_mcp_tools:
        await on_message_mcp_tools(message)
    else:
        await main(message)


# ======================== ChatSettings设置更新时被调用 ========================
@cl.on_settings_update
async def setup_agent(settings):
    # 声明使用全局变量
    global image_processor
    global image_model
    global is_tools
    global is_mcp_tools
    global tts
    global ocr_model
    global ocr_tokenizer

    print("Setup agent with following settings: ", settings)
    # Setup agent with following settings:  {'mode_selector': 'knowledge_mode'}
    # 从settings字典中获取mode_selector的值
    mode_value = settings.get("mode_selector", "text_mode")  # 第二个参数是可选的默认值
    if settings.get("describe") == None:  #对应的值是否不为空
        await cl.Message(content=f"处理模式为: {mode_value}").send()
    cl.user_session.set("processing_mode", mode_value)

    if mode_value != "mcp_calling_tools":  
        is_mcp_tools = False
    else:
        is_mcp_tools = True

    if mode_value != "calling_tools":  
        await cl.context.emitter.set_commands([])
        is_tools = False
    else:
        await cl.context.emitter.set_commands(commands)
        is_tools = True

    if mode_value != "image_ocr_ai_mode":  # 如果不是 ai图片ocr
        await update_mode_selector(mode_value)  # 同步更新设置面板
        if ocr_tokenizer is not None and ocr_model is not None:
            model_msg = cl.Message(content="正在卸载ocr模型...")
            await model_msg.send()
            await asyncio.sleep(0.1)  # 添加短暂延迟确保UI更新
            unload_ocr_model()  # 调用卸载函数
            model_msg.content = "卸载ocr模型成功"
            await model_msg.update()
            ocr_tokenizer = None
            ocr_model = None

    if mode_value != "voice_mode":  # 如果不是 voice_mode
        if tts is not None:
            model_msg = cl.Message(content="正在卸载语音模型...")
            await model_msg.send()
            await asyncio.sleep(0.1)  # 添加短暂延迟确保UI更新
            unload_voice_model()
            model_msg.content = "卸载语音模型成功"
            await model_msg.update()
            tts = None


    if mode_value != "image_mode":  # 如果不是 image_mode
        if image_processor is not None and image_model is not None:
            model_msg = cl.Message(content="正在卸载图片模型...")
            await model_msg.send()
            await asyncio.sleep(0.1)  # 添加短暂延迟确保UI更新
            unload_image_model_and_processor(image_processor, image_model)  # 卸载图片模型
            model_msg.content = "卸载图片模型成功"
            await model_msg.update()
            image_processor = None
            image_model = None        

    if mode_value == "qa_mode":

        model_msg = cl.Message(content="正在准备...")
        await model_msg.send()
        await asyncio.sleep(0.1)  # 添加短暂延迟确保UI更新

        # 加载模型
        tts = get_tts()  # 替换原来的全局引用

        await asyncio.sleep(0.1)  # 添加短暂延迟确保UI更新
        model_msg.content = "准备成功"
        await model_msg.update()

        files = None
        # 在代码开始处设置 chain 为空值
        cl.user_session.set("chain", None)
        # 等待用户上传文件
        while files is None:
            files = await cl.AskFileMessage(
                content="请上传一个文件以开始！",  # 提示用户上传文件
                accept=["*/*"],  # 接受任何文件类型
                max_size_mb=100,  # 文件最大大小为20MB
                timeout=300,  # 超时时间为180秒
            ).send()
        file_path = files[-1].path  # 获取用户上传的第一个文件路径
        file_name = files[-1].name  # 获取用户上传的第一个文件名
        print (file_name)
        # 清理 Chroma 的默认持久化数据（如果存在）
        if os.path.exists("./chroma"):
            import shutil
            shutil.rmtree("./chroma")
        async def process_file_qa_mode(file):
            # 初始化对话链
            chain = initialize_chain(file)
            return chain
        
        await asyncio.sleep(0.1)  # 添加短暂延迟确保UI更新
        # 发送消息，提示正在处理文件
        msg = cl.Message(content=f"正在处理 `{file_name}`...")
        await msg.send()
        await asyncio.sleep(0.1)  # 添加短暂延迟确保UI更新

        if file_name:
            if '.wav' in file_name:
                ref_text = tts.transcribe(file_path) 
                converter = opencc.OpenCC('t2s')
                ref_text = converter.convert(ref_text)
                msg.content = ref_text
            else:
                # 使用异步线程执行阻塞操作
                loop = asyncio.get_event_loop()
                chain = await loop.run_in_executor(None, lambda: asyncio.run(process_file_qa_mode(file_path)))
                # 通知用户系统已准备好
                msg.content = f"处理 `{file_name}` 完成。你现在可以提问了！"
                # 将对话链存储在用户会话中
                cl.user_session.set("chain", chain)
                
        await msg.update()


    if mode_value == "image_ocr_ai_mode":
        await update_image_ocr_ai_mode_selector(mode_value)  # 同步更新设置面板
        print (settings.get("describe"))
        if settings.get("describe") == True:
            cl.user_session.set("image_describe", "请用一段自然语言的句子详细描述图片，所有物品需要明确数量，并说明相关场景元素位于场景图片的那个方位。")
            await update_image_ocr_ai_mode_selector(mode_value , True)
        elif settings.get("describe") == False:
            cl.user_session.set("image_describe", "请识别图片中的文字, 不要添加额外文字如图中所有文字之类，直接输出原文文字。")
            await update_image_ocr_ai_mode_selector(mode_value , False)
        elif settings.get("describe") == None:
            model_msg = cl.Message(content="正在加载ocr模型...")
            await model_msg.send()
            await asyncio.sleep(0.1)  # 添加短暂延迟确保UI更新
            # 加载模型
            load_ocr_model()  # 替换原来的全局引用
            await asyncio.sleep(0.1)  # 添加短暂延迟确保UI更新
            model_msg.content = "ocr模型加载成功"
            await model_msg.update()
            cl.user_session.set("image_describe", "请用一段自然语言的句子详细描述图片，所有物品需要明确数量，并说明相关场景元素位于场景图片的那个方位。")
  
    if mode_value == "voice_mode":
        await update_voice_mode_selector(mode_value)  # 同步更新设置面板
        print (settings.get("voice_mode_describe"))
        if settings.get("voice_mode_describe") == True:
            cl.user_session.set("voice_mode_response", "开启模型语音回复")
            await update_voice_mode_selector(mode_value , True)
        elif settings.get("voice_mode_describe") == False:
            cl.user_session.set("voice_mode_response", "")  # 关闭根据输入框消息生成语音
            await update_voice_mode_selector(mode_value , False)
        elif settings.get("voice_mode_describe") == None:
            model_msg = cl.Message(content="正在加载语音模型...")
            await model_msg.send()
            await asyncio.sleep(0.1)  # 添加短暂延迟确保UI更新
            # 加载模型
            tts = get_tts()  # 替换原来的全局引用
            await asyncio.sleep(0.1)  # 添加短暂延迟确保UI更新
            model_msg.content = "语音模型加载成功"
            await model_msg.update()
            cl.user_session.set("voice_mode_response", "开启模型语音回复")

    if mode_value == "image_mode":
        # 直接在主线程中发送消息
        model_msg = cl.Message(content="正在加载图片模型...")
        await model_msg.send()
        await asyncio.sleep(0.1)  # 添加短暂延迟确保UI更新

        # 加载模型
        image_processor = AutoProcessor.from_pretrained(
            'D:\\chainlit\\models\\molmo-7B-D-bnb-4bit',
            trust_remote_code=True,
            torch_dtype='auto',
            use_fast=True,
            device_map='auto'
        )
        image_model = AutoModelForCausalLM.from_pretrained(
            'D:\\chainlit\\models\\molmo-7B-D-bnb-4bit',
            trust_remote_code=True,
            torch_dtype='auto',
            device_map='auto',
            load_in_4bit=True,
        )
        await asyncio.sleep(0.1)  # 添加短暂延迟确保UI更新
        model_msg.content = "图片模型加载成功"
        await model_msg.update()


# ======================== 卸载语音模型和处理器，释放内存 ========================
def unload_voice_model():
    try:
        # 获取用户会话中的模型实例
        if hasattr(cl.user_session, "tts"):
            tts_instance = cl.user_session.tts
            
            # 先释放模型内部组件
            if hasattr(tts_instance, "ema_model"):
                # 将模型移回CPU
                if hasattr(tts_instance.ema_model, "to"):
                    tts_instance.ema_model.to("cpu")
                del tts_instance.ema_model
                
            if hasattr(tts_instance, "vocoder"):
                # 释放声码器资源
                if hasattr(tts_instance.vocoder, "to"):
                    tts_instance.vocoder.to("cpu")
                del tts_instance.vocoder
            
            # 删除用户会话中的模型引用
            del cl.user_session.tts
            
            # 双重垃圾回收机制
            import gc
            for _ in range(3):  # 三次回收确保彻底
                gc.collect()
            
            # 清理GPU缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                
            print("✅ 语音模型已成功卸载")
            
    except Exception as e:
        print(f"❌ 卸载语音模型失败: {str(e)}")
    finally:
        # 确保会话属性清除
        if hasattr(cl.user_session, "tts"):
            del cl.user_session.tts

# ======================== 卸载OCR模型和处理器，释放内存 ========================
def unload_ocr_model():
    global ocr_model
    global ocr_tokenizer
    
    cl.user_session.set("last_one_msg_elements", None)
    
    try:
        # 删除模型和分词器的所有引用（不再尝试移动设备）
        del ocr_model
        del ocr_tokenizer
        
        # 强制垃圾回收
        import gc
        gc.collect()
        
        # 清理GPU缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            
        print("✅ OCR模型已成功卸载")
        
    except Exception as e:
        print(f"❌ 卸载OCR模型失败: {str(e)}")
    finally:
        # 确保全局变量置空
        ocr_model = None
        ocr_tokenizer = None


# ======================== 卸载图片模型和处理器，释放内存 ========================
def unload_image_model_and_processor(image_processor, image_model):
    """
    卸载模型和处理器，释放内存
    :param image_processor: 已加载的处理器对象
    :param model: 已加载的模型对象
    """

    cl.user_session.set("last_one_msg_elements", None)

    try:
        # 确保模型和处理器在GPU上
        if hasattr(image_model, 'device'):
            device = str(image_model.device)
            if 'cuda' in device:
                # 将模型移回CPU
                image_model.to('cpu')
        
        # 删除模型的所有引用
        if image_model is not None:
            # 清除模型参数
            for param in image_model.parameters():
                if param is not None:
                    del param
            # 清除模型缓冲区
            for buffer in image_model.buffers():
                if buffer is not None:
                    del buffer
            # 删除模型本身
            del image_model
        
        # 删除处理器
        if image_processor is not None:
            del image_processor
        
        # 强制垃圾回收
        import gc
        gc.collect()
        
        # 清理GPU缓存
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()  # 额外的清理
        
        print("✅ 模型和处理器已成功卸载，内存已释放")
        
    except Exception as e:
        print(f"❌ 卸载过程中发生错误: {str(e)}")


# ======================== 模式选择回调 ========================
async def update_mode_selector(mode: str):
    """更新设置面板的Select组件显示"""
    await cl.ChatSettings(
        [
            Select(
                id="mode_selector",
                label="选择处理模式",
                items={
                    "文本对话": "text_mode",
                    "图片分析": "image_mode",
                    "知识库查询": "knowledge_mode",
                    "联网搜索": "web_search_mode",
                    "文件问答": "qa_mode",
                    "AI图片OCR": "image_ocr_ai_mode",
                    "调用tools工具": "calling_tools",
                    "语音回复": "voice_mode",
                    "调用MCP工具": "mcp_calling_tools",
                },
                initial_value=mode,
            ),
        ]
    ).send()


async def update_image_ocr_ai_mode_selector(mode: str,initial=True):
    """更新设置面板的Select组件显示"""
    await cl.ChatSettings(
        [
            Select(
                id="mode_selector",
                label="选择处理模式",
                items={
                    "文本对话": "text_mode",
                    "图片分析": "image_mode",
                    "知识库查询": "knowledge_mode",
                    "联网搜索": "web_search_mode",
                    "文件问答": "qa_mode",
                    "AI图片OCR": "image_ocr_ai_mode",
                    "调用tools工具": "calling_tools",
                    "语音回复": "voice_mode",
                    "调用MCP工具": "mcp_calling_tools",
                },
                initial_value=mode,
            ),
            Switch(id="describe", label="默认开启详细描述, 关闭并且输入信息少于2个字符则只会OCR图片文字", initial=initial),
        ]
    ).send()

async def update_voice_mode_selector(mode: str,initial=True):
    """更新设置面板的Select组件显示"""
    await cl.ChatSettings(
        [
            Select(
                id="mode_selector",
                label="选择处理模式",
                items={
                    "文本对话": "text_mode",
                    "图片分析": "image_mode",
                    "知识库查询": "knowledge_mode",
                    "联网搜索": "web_search_mode",
                    "文件问答": "qa_mode",
                    "AI图片OCR": "image_ocr_ai_mode",
                    "调用tools工具": "calling_tools",
                    "语音回复": "voice_mode",
                    "调用MCP工具": "mcp_calling_tools",
                },
                initial_value=mode,
            ),
            Switch(id="voice_mode_describe", label="默认开启模型语音回复, 关闭则根据输入框消息生成语音", initial=initial),
        ]
    ).send()



# ======================== MCP工具调用处理 ========================

# 当 MCP 连接时触发的异步函数
@cl.on_mcp_connect
async def on_mcp(connection, session: ClientSession):
    # 获取当前 MCP 连接的所有可用工具
    result = await session.list_tools()
    # 整理工具信息，将其转换为字典形式
    tools = [{
        "name": t.name,
        "description": t.description,
        "input_schema": t.inputSchema,
        } for t in result.tools]
    
    # 从用户会话中获取已有的 MCP 工具列表
    mcp_tools = cl.user_session.get("mcp_tools", {})
    # 将当前连接的工具添加到 MCP 工具列表中
    mcp_tools[connection.name] = tools
    # 更新用户会话中的 MCP 工具列表
    cl.user_session.set("mcp_tools", mcp_tools)


# 2. 处理MCP连接断开
@cl.on_mcp_disconnect
async def on_mcp_disconnect(name: str, session: ClientSession):
    await cl.Message(content=f"MCP服务器 '{name}' 已断开连接").send()


# 工具调用步骤的异步函数
@cl.step(type="tool") 
async def call_mcp_tool(tool_name, tool_input):
    """调用MCP工具并返回结果"""
    # 获取当前步骤用于跟踪工具调用
    current_step = cl.context.current_step
    current_step.name = tool_name
    
    # 查找工具所在的MCP连接
    mcp_tools = cl.user_session.get("mcp_tools", {})
    mcp_name = None
    for connection_name, tools in mcp_tools.items():
        if any(tool.get("name") == tool_name for tool in tools):
            mcp_name = connection_name
            break
    
    # 错误处理：未找到工具
    if not mcp_name:
        error_msg = f"工具 {tool_name} 未在任何MCP连接中找到"
        current_step.output = json.dumps({"error": error_msg})
        return error_msg
    
    # 获取MCP会话
    mcp_session, _ = cl.context.session.mcp_sessions.get(mcp_name)
    if not mcp_session:
        error_msg = f"MCP连接 {mcp_name} 未找到"
        current_step.output = json.dumps({"error": error_msg})
        return error_msg
    
    try:
        # 实际调用MCP工具
        result = await mcp_session.call_tool(tool_name, tool_input)
        current_step.output = result
        return result
    except Exception as e:
        error_msg = f"调用工具 {tool_name} 时出错: {str(e)}"
        current_step.output = json.dumps({"error": error_msg})
        return error_msg



# ======================== on_message_mcp_tools消息处理 ========================
async def on_message_mcp_tools(message: cl.Message):

    cur_iter = 0
    tool_call_id = True
    # 获取用户会话中的 MCP 工具字典
    mcp_tools_dict = cl.user_session.get("mcp_tools", {})

    # 合并所有连接的工具列表
    all_tools = []
    for tools_list in mcp_tools_dict.values():
        all_tools.extend(tools_list)


    # 转换工具格式以适应OpenAI的函数调用要求
    openai_tools = []
    for tool in all_tools:  # 现在tool是字典
        openai_tool = {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool["input_schema"]
            }
        }
        openai_tools.append(openai_tool)

    # 打印工具名称列表，方便调试
    print("openai_tools:\n",openai_tools)

    user_message = message.content
    cl.user_session.set("user_message", user_message)
    message_history = cl.user_session.get("message_history", [])
    message_history.append({"role": "user", "content": user_message})

    # 消息历史管理（保持原有逻辑）
    if len(message_history) > 20:
        message_history = [message_history[0]] + message_history[-39:]

    while cur_iter < MAX_ITER:
        # 每次请求前重新注入系统提示
        current_system_message = get_system_message(openai_tools, language)
        
        # 检查 enhanced_history 是否已经包含系统消息
        system_message_exists = any(
            msg["role"] == "system" and msg["content"] == current_system_message
            for msg in message_history
        )
        
        # 如果没有系统消息，则添加
        if not system_message_exists:
            enhanced_history = [
                {"role": "system", "content": current_system_message},
                *[msg for msg in message_history if msg["role"] != "system"]
            ]
        else:
            enhanced_history = message_history

        #如果 tool_call_id 无效或不存在，则终止循环。
        if not tool_call_id:
            break

        #print (enhanced_history)
        tool_call_id = await tool_mcp_calls(enhanced_history, openai_tools)

        cur_iter += 1
        continue


# ======================== tool_mcp_calls流式请求处理 ========================
async def tool_mcp_calls(message_history: list, tools:any):
    # 工具调用处理（非流式）
    full_resp = await client.chat.completions.create(
        model=os.environ.get("LLM_MODEL"),
        messages=message_history,
        tools=tools,
        tool_choice="auto",
        timeout=300.0,  # 增加超时时间
        temperature=0
    )
    #print ("full_resp:\n", full_resp)
    openai_message = full_resp.choices[0].message
    content = openai_message.content or ""

    # 处理工具调用请求
    if full_resp.choices[0].message.tool_calls:
        # 取第一个工具调用
        tool_call = full_resp.choices[0].message.tool_calls[0]
        tool_use_id = tool_call.id
        tool_name = tool_call.function.name
        tool_input = json.loads(tool_call.function.arguments)
        
        # 调用其他MCP工具
        result = await call_mcp_tool(tool_name, tool_input)
        print ("results:\n",result)

        message_history = cl.user_session.get("message_history", [])
        # 将工具调用结果添加到消息历史
        message_history.extend([
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": tool_use_id,
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "arguments": json.dumps(tool_input)
                        }
                    }
                ]
            },
            {
                "role": "tool",
                "content": str(result),
                "tool_call_id": tool_use_id
            }
        ])
        return tool_use_id
    else:
        # 没有工具调用，返回最终响应
        await handle_text_input(message_history , content)
        return None


# ======================== python_exec函数的code修复缩进与字符转义 ========================
def fix_indentation(code_str):
    lines = []
    indent_level = 0
    block_keywords = {'for', 'if', 'else', 'elif', 'while', 'def', 'class', 'try', 'except', 'with'}
    
    for line in code_str.split('\n'):
        stripped = line.strip()
        if not stripped:
            lines.append('')
            continue
        
        # 检测是否是块关键字（如 if, for 等）
        is_block = any(
            re.match(rf'^{kw}\b.*:$', stripped)
            for kw in block_keywords
        )
        
        # 处理 else/elif 的特殊情况
        if stripped.startswith(('else', 'elif')):
            indent_level = max(0, indent_level - 1)
        
        # 添加当前缩进
        lines.append('    ' * indent_level + stripped)
        
        # 如果是块关键字，增加缩进
        if is_block:
            indent_level += 1
            
    return '\n'.join(lines)


# ======================== 工具调用处理 ========================
@cl.step(type="tool")
async def process_tool_calls(openai_message: ChatCompletionMessage) -> dict:
    """处理工具调用, 新增tool_call_id生成"""
    print ("tool_calls:\n", openai_message.tool_calls)

    results = []
    for tool_call in openai_message.tool_calls:
        function_name = tool_call.function.name
        arguments_str = tool_call.function.arguments

        print ("* 原始arguments_str:\n", arguments_str)
        tool_call_id = tool_call.id 

        try:
            arguments = json.loads(arguments_str)
        except:
            arguments = ast.literal_eval(
                arguments_str
            )
        print ("* 第一步arguments:\n", arguments)

        if 'code' in arguments:
            code_str = arguments['code']

            try:
                # 处理代码块标记
                if code_str.startswith('```py'):
                    code_str = code_str[5:].lstrip()
                if code_str.endswith('```'):
                    code_str = code_str[:-3].rstrip()
                # 更新并重新序列化
                arguments['code'] = code_str
        
            except json.JSONDecodeError:
                # 非JSON格式则直接处理原始字符串
                if arguments.startswith('```py'):
                    arguments = arguments[5:].lstrip()
                if arguments.endswith('```'):
                    arguments = arguments[:-3].rstrip()

            print ("* 第二步arguments:\n", arguments)
        

            # 检测try/except存在时不修复缩进
            if 'try' not in code_str and 'except' not in code_str:
                code_str = (
                    arguments['code']
                    #.replace('\\n', '\n')
                    #.replace('\\t', "'\\t'")  # 字符串替换：直接替换 \\t 为 '\t'。原字符串中的 \\t 表示字面的反斜杠和 t，替换后 '\t' 中的 \t 会被 Python 识别为转义后的制表符。
                    #.replace("\\'", "'")   
                    #.replace('\\"', '"')
                    #.replace("\\'", "'")   #  步骤解释器的代码多出现\'  原始代码是'    可以问大模型 \'转换成'，代码如何实现，用replace方法
                )
                # 修复缩进
                code_str = fix_indentation(code_str)
                arguments['code'] = code_str
                print ("* 第三步arguments:\n", arguments)

        current_step = cl.context.current_step
        current_step.name = function_name
        if current_step.name == "python_exec":
            # 输出python语言的输入步骤解释器
            arguments_code_py = f"```py\n{arguments['code']}\n```"
            current_step.input = arguments_code_py
        else:
            current_step.input = arguments

        try:
            if function_name == "python_exec":
                function_response = await function_manager.call_function(function_name, arguments)

            else:
                function_response = await function_manager.call_function(function_name, arguments)
            current_step.output = function_response
            current_step.language = "json"
            results.append({
                "tool_call_id": tool_call_id,
                "function_name": function_name,
                "arguments": arguments,
                "function_response": function_response
            })

        except Exception as e:
            print(f"函数{function_name}执行失败: {str(e)}")
            results.append({
                "tool_call_id": tool_call_id,
                "function_name": function_name,
                "arguments": arguments,
                "function_response": f"函数{function_name}执行失败: {str(e)}"
            })
    
    return results


# ======================== function_calls流式请求处理 ========================
async def tool_calls(message_history: list):
    # 第一阶段：工具调用处理（非流式）
    full_resp = await client.chat.completions.create(
        model=os.environ.get("LLM_MODEL"),
        messages=message_history,
        tools=tools,
        tool_choice="auto",
        timeout=300.0,  # 增加超时时间
        temperature=0
    )
    #print ("full_resp:\n", full_resp)
    openai_message = full_resp.choices[0].message
    content = openai_message.content or ""

    
    #如果变量tool_calls属性为空，就调用`handle_text_input`函数, 并且返回`None`代表没有函数可以调用，跳出循环。
    if not openai_message.tool_calls:
        await handle_text_input(message_history , content)
        return None
    else:
        # 处理工具调用
        results = await process_tool_calls(openai_message)
        #print ("results:\n",results)
        if results:  # results 是一个列表
            for result in results:  # 遍历列表中的每个结果
            
                # 显示函数返回结果
                if result['function_name'] == "python_exec":
                    if "function_response" in result and "success" in result['function_response'].lower():
                        message_content = f"python_exec 函数执行成功，返回结果如下：\n{result['function_response']}"
                    else:
                        message_content = f"python_exec 函数执行失败，返回结果如下：\n{result.get('function_response', '无返回结果')}"
                else:
                    message_content = f"{result['function_name']} 函数执行成功，返回结果如下：\n{result['function_response']}"


            message_history = cl.user_session.get("message_history", [])
            print ("message_content:\n",message_content)
            message_history.append({"role": "user", "content": message_content})

        return results

# ======================== 图片OCR处理模块 ========================
# async def handle_image_ocr(message: cl.Message):
#     if not message.elements or not message.elements[0].mime.startswith("image"):
#         await cl.Message("❌ 请上传图片文件").send()
#         return
    
#     image = message.elements[0]
#     await cl.Message("🖼️ 正在进行OCR处理...").send()

#     await asyncio.sleep(0.1)  # 添加短暂延迟确保UI更新

#     try:
#         ocr_text = await process_image_ocr(image.path)
#         if ocr_text:
#             await cl.Message(f"**OCR识别内容**\n\n{ocr_text}").send()
#         else:
#             await cl.Message("❌ 未识别到有效文本").send()
#     except Exception as e:
#         await cl.Message(f"❌ OCR处理失败: {str(e)}").send()
#     finally:
#         message.elements = []  # 清空消息中的图片元素

# ======================== 语音回复处理模块 ========================
async def handle_voice_mode(message: cl.Message):
    global tts

    voice_mode_response = cl.user_session.get("voice_mode_response")

    if not voice_mode_response:
        response = message.content
    else:
        response = await handle_text_input(
            cl.user_session.get("conversation_history", []),
            message.content
        )
    msg = cl.Message(content=f"语音生成中...")
    await msg.send()

    print("当前目录:", os.getcwd())  #当前目录: D:\chainlit\chainlit-datalayer\demo_app\tmp
    # 生成语音
    audio_file = os.path.join("..", str(TEMP_DIR), f"{uuid.uuid4()}.wav")
    print("audio_file:", audio_file)

    #ref_text = tts.transcribe("../tts_temp/seedtts_ref_zh_1.wav") 

    async def tts_infer(response , audio_file):
        tts.infer(
                ref_file="../tts_temp/seedtts_ref_zh_1.wav",  #../是上一级目录
                ref_text="对于疫情大家不要轻视但也不用过度恐慌做到不哄抢物品不哄抬物价.",  # 传递空字符串占位
                gen_text=response,
                file_wave=str(audio_file),
            )
        return audio_file
    
    await asyncio.sleep(0.1)  # 添加短暂延迟确保UI更新

    # 使用异步线程执行阻塞操作
    loop = asyncio.get_event_loop()
    audio_file = await loop.run_in_executor(None, lambda: asyncio.run(tts_infer(response , str(audio_file))))

    # 读取并发送音频
    #with open(audio_file, "rb") as f:
    #    audio_data = f.read()

    await msg.remove()
    
    audio_element = cl.Audio(
        path=str(audio_file),  # 显式转换为字符串
        #content=audio_data,
        mime="audio/wav",
        auto_play=True
    )

    #audio_file.resolve().unlink(missing_ok=True)  # 更安全的清理

    await asyncio.sleep(0.1)  # 添加短暂延迟确保UI更新
    
    await cl.Message(
        content="",
        elements=[audio_element]
    ).send()

# ======================== ai_图片OCR处理模块 ========================
# 定义函数：根据时间范围和视频参数计算需要抽取的帧索引
def get_index(bound, fps, max_frame, first_idx=0, num_segments=32):
    """参数说明：
    bound: 时间范围元组(start_sec, end_sec)
    fps: 视频帧率（帧/秒）
    max_frame: 视频总帧数
    first_idx: 起始帧索引（默认0）
    num_segments: 需要分割的视频段数（默认32）"""
    
    # 处理时间边界
    if bound:  # 如果指定了时间范围
        start, end = bound[0], bound[1]  # 获取开始和结束时间（秒）
    else:  # 未指定则使用极大范围
        start, end = -100000, 100000
    
    # 计算起始和结束帧索引
    start_idx = max(first_idx, round(start * fps))  # 转换为帧索引，确保不小于first_idx
    end_idx = min(round(end * fps), max_frame)  # 结束帧不超过视频最大帧
    
    # 计算每个视频段的长度（以帧为单位）
    seg_size = float(end_idx - start_idx) / num_segments
    
    # 生成每个视频段的中心帧索引
    frame_indices = np.array([
        int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
        for idx in range(num_segments)
    ])
    return frame_indices  # 返回32个均匀分布的帧索引


# 定义视频加载和处理函数
def load_video(video_path, bound=None, input_size=448, max_num=1, num_segments=32):
    """参数说明：
    video_path: 视频文件路径
    bound: 时间范围（秒）
    input_size: 输入图像尺寸（默认448x448）
    max_num: 最大分块数量（动态预处理用）
    num_segments: 视频分割段数"""
    
    # 初始化视频阅读器
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)  # 使用CPU单线程读取
    max_frame = len(vr) - 1  # 获取视频总帧数（索引从0开始）
    fps = float(vr.get_avg_fps())  # 获取视频平均帧率

    # 初始化存储容器
    pixel_values_list = []  # 存储处理后的图像张量
    num_patches_list = []   # 存储每帧的分块数量
    
    # 创建图像预处理流水线
    transform = build_transform(input_size=input_size)  # 包含缩放、归一化等操作
    
    # 获取需要处理的帧索引
    frame_indices = get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments)
    
    # 遍历每个选定帧进行处理
    for frame_index in frame_indices:
        # 读取帧并转换为PIL图像
        img = Image.fromarray(vr[frame_index].asnumpy()).convert('RGB')
        
        # 动态预处理（可能包含分块、缩略图处理等）
        img = dynamic_preprocess(img, 
                                image_size=input_size,
                                use_thumbnail=True,
                                max_num=max_num)
        
        # 对每个分块应用预处理
        pixel_values = [transform(tile) for tile in img]
        pixel_values = torch.stack(pixel_values)  # 堆叠分块张量
        
        # 记录分块数量和预处理结果
        num_patches_list.append(pixel_values.shape[0])
        pixel_values_list.append(pixel_values)
    
    # 合并所有帧的分块数据
    pixel_values = torch.cat(pixel_values_list)
    return pixel_values, num_patches_list



# ImageNet数据集的标准化参数
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
# ai_ocr图像预处理
def build_transform(input_size):
    """构建图像预处理流水线"""
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    return T.Compose([
        # 确保图像为RGB格式
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        # 调整大小并使用双三次插值
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),  # 转换为张量
        T.Normalize(mean=MEAN, std=STD)  # 标准化
    ])

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    """寻找最接近的目标宽高比"""
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height  # 原始图像面积
    # 遍历所有候选比例
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        # 选择差异最小的比例
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:  # 相同差异时选择面积更大的
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    """动态图像预处理：将图像分割为多个子图"""
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height  # 原始宽高比
    
    # 生成所有可能的宽高比组合
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) 
        for i in range(1, n + 1) for j in range(1, n + 1) 
        if i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    
    # 找到最佳比例并计算目标尺寸
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]  # 总块数
    
    # 调整大小并分割图像
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        # 计算每个子图的坐标
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        processed_images.append(resized_img.crop(box))  # 裁剪子图
        
    # 可选添加缩略图
    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size)))
    return processed_images

def ai_ocr_preprocess(image_file, input_size=448, max_num=12):
        """加载并预处理图像"""
        image = Image.open(image_file).convert('RGB')
        transform = build_transform(input_size)
        images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(img) for img in images]  # 应用预处理
        return torch.stack(pixel_values)  # 堆叠为张量


async def handle_image_ai_ocr(message: cl.Message):
    global ocr_model
    global ocr_tokenizer
    global num_patches_list
    global video_prefix
    #ocr_model = cl.user_session.get("ocr_model")
    #ocr_tokenizer = cl.user_session.get("ocr_tokenizer")
    conversation_history = cl.user_session.get("conversation_history", [])

    if message.elements:  # 先判断elements是否存在且非空
        cl.user_session.set("last_one_msg_elements", message.elements[-1])

    last_one_msg_elements = cl.user_session.get("last_one_msg_elements", None)
    
    # 验证图片上传
    if not last_one_msg_elements:
        pixel_values = cl.user_session.get("pixel_values", None)

    if last_one_msg_elements:
        if last_one_msg_elements.mime.startswith("video"):  
            video_element = last_one_msg_elements
            print(f"Received video: {video_element.path}")
            video_path = video_element.path
            # 加载并处理视频（设置8个时间段，每个时间取1个分块）
            pixel_values, num_patches_list = load_video(video_path, num_segments=8, max_num=1)

            # 确保数据在GPU上
            if torch.cuda.is_available():
                # 将数据转换为bfloat16格式并转移到GPU
                pixel_values = pixel_values.to(torch.bfloat16).cuda()
                
            # 构造视频前缀：为每个帧生成"FrameX: <image>\n"格式的文本
            video_prefix = ''.join([f'Frame{i+1}: <image>\n' for i in range(len(num_patches_list))])
            cl.user_session.set("pixel_values", pixel_values)

    if last_one_msg_elements:
        if last_one_msg_elements.mime.startswith("image"): 
            # 获取图片路径并预处理
            image_element = last_one_msg_elements
            print(f"Received image: {image_element.path}")
            # 转换张量到模型设备
            pixel_values = ai_ocr_preprocess(image_element.path).to(torch.bfloat16)
            # 确保数据在GPU上
            if torch.cuda.is_available():
                pixel_values = pixel_values.to('cuda')
            num_patches_list = [pixel_values.size(0)]
            cl.user_session.set("pixel_values", pixel_values)

    msg = await cl.Message("🖼️ 正在进行OCR处理...").send()
    await asyncio.sleep(0.1)  # 确保UI更新
    
    try:
        # 准备流式输出
        streamer = TextIteratorStreamer(ocr_tokenizer, timeout=60)
        generation_config = {
            "max_new_tokens": 4000,
            "do_sample": False,
            "streamer": streamer
        }
        
        # 创建消息对象
        msg_ocr = cl.Message(content="")
        await msg_ocr.send()


        if last_one_msg_elements:
            if last_one_msg_elements.mime.startswith("image"):  # 新增视频处理分支
                image_message_describe = cl.user_session.get("image_describe")
                print ("image_describe:\n",image_message_describe)
                if image_message_describe and len(message.content) < 2:
                    message.content = image_message_describe
            elif last_one_msg_elements.mime.startswith("video"):  # 新增视频处理分支
                video_message_describe = "详细描述这个视频"
                print ("video_describe:\n",video_message_describe)
                if video_message_describe and len(message.content) < 2:
                    message.content = video_message_describe
        else:
            message.content = message.content


        if last_one_msg_elements:
            print ("last_one_msg_elements:\n",last_one_msg_elements)
            if last_one_msg_elements.mime.startswith("video"):  # 新增视频处理分支
                # 启动生成线程
                Thread(target = ocr_model.chat, kwargs={
                    "tokenizer": ocr_tokenizer,
                    "pixel_values": pixel_values,  # 使用处理后的张量
                    "question": (video_prefix + message.content),
                    "generation_config": generation_config,
                    "num_patches_list": num_patches_list,
                    "history": conversation_history,  # 关键修改：每次使用空历史
                    "return_history": False  # 不再需要返回历史
                }).start()

            if last_one_msg_elements.mime.startswith("image"):  # 新增视频处理分支
                # 启动生成线程
                Thread(target = ocr_model.chat, kwargs={
                    "tokenizer": ocr_tokenizer,
                    "pixel_values": pixel_values,  # 使用处理后的张量
                    "question": ("<image>\n" + message.content),
                    "generation_config": generation_config,
                    "num_patches_list": num_patches_list,
                    "history": conversation_history,  # 关键修改：每次使用空历史
                    "return_history": False  # 不再需要返回历史
                }).start()
        else:
            # 启动生成线程
            print ("message.content:\n",message.content)
            Thread(target = ocr_model.chat, kwargs={
                "tokenizer": ocr_tokenizer,
                "pixel_values": None,  # 使用处理后的张量
                "question": message.content,
                "generation_config": generation_config,
                "history": conversation_history,  # 关键修改：每次使用空历史
                "return_history": False  # 不再需要返回历史
            }).start()   

        # 流式响应
        response = ''
        # Loop through the streamer to get the new text as it is generated
        for token in streamer:
            if token == ocr_model.conv_template.sep:
                continue
            #print(token, end="\n", flush=True)  # Print each new chunk of generated text on the same line
            if "<|im_end|>" in token:
                token = token.replace("<|im_end|>", "")
                if token:  # 如果删除后还有内容，继续发送剩余部分
                    await msg_ocr.stream_token(token)
                continue
            #response += token
            await msg_ocr.stream_token(token)
        
        response = msg_ocr.content

        if response.startswith('\[') and response.endswith('\]'):
            response = f"$${response[2:-2]}$$"

        if response == "content" or response == "role":
            response = "图片没有文字"
        print(f"最终response:\n{response}")
        msg_ocr.content = response
        await msg_ocr.update()
        # 更新消息前手动设置最终内容
        await msg.remove()

        conversation_history.extend([
            {"role": "user", "content":message.content},
            {"role": "assistant", "content": response}
        ])
        
        cl.user_session.set("conversation_history", conversation_history)

        
    except Exception as e:
        await cl.Message(f"❌ 处理失败: {str(e)}").send()


# ======================== 图片处理模块 ========================
async def handle_image_input(message: cl.Message, history: list):
    image = message.elements[0]
    print(f"Received image: {image.path}")

    query_msg = None  # 显式初始化
    try:
        query_msg = await cl.Message("🖼️ 正在分析图片...").send()
        # 原有图片处理逻辑
        
        response = await process_image(
            image.path,
            message.content if (message.content and len(message.content) > 2) 
            else "Describe this image."
        )
        full_response = f"**图片识别英文内容**\n\n{response}"
        await cl.Message(full_response).send()
        history.extend([
            {"role": "user", "content": message.content if (message.content and len(message.content) > 2) 
            else "Describe this image."},
            {"role": "assistant", "content": full_response}
        ])
    except Exception as e:
        error_msg = f"❌ 处理失败: {str(e)}"
        if 'query_msg' in locals():
            query_msg.content = error_msg
            await query_msg.update()
        else:
            await cl.Message(content=error_msg).send()
        
    cl.user_session.set("conversation_history", history)


# ======================== 图片处理模块-辅助函数 ========================
async def process_image(image_path, prompt):
    # 将同步阻塞操作包装到异步线程中
    def _sync_process():
        image = Image.open(image_path)
        print("Image opened successfully")
        inputs = image_processor.process(images=[image], text=prompt)
        print("Image processed successfully")
        inputs = {k: v.to(image_model.device).unsqueeze(0) for k, v in inputs.items()}
        print("Inputs prepared successfully")
        output = image_model.generate_from_batch(
            inputs,
            GenerationConfig(
                max_new_tokens=400,
                stop_strings="<|endoftext|>",
                do_sample=True,  # 启用采样
                temperature=0.6,
                top_k=40,
                top_p=0.9,
            ),
            tokenizer=image_processor.tokenizer
        )
        decoded_output = image_processor.tokenizer.decode(output[0, inputs['input_ids'].size(1):], skip_special_tokens=True)
        print(f"Decoded output: {decoded_output}")
        return decoded_output
    try:
        # 使用异步线程执行阻塞操作
        loop = asyncio.get_event_loop()
        decoded_output = await loop.run_in_executor(None, _sync_process)
        return decoded_output
    except Exception as e:
        return f"图片处理错误: {str(e)}"

            
# ======================== 知识查询模块 ========================
async def handle_post_upload_actions(user_input: str):
    # 初始化客户端
    query_msg = None  # 显式初始化
    query = user_input

    api_key = os.getenv("LIGHTRAG_API_KEY")
    if not api_key:
        raise ValueError("LIGHTRAG_API_KEY 环境变量未设置！")
    client = LightRAGClient(api_key=api_key)

    # 自定义配置
    # mode="local", "global", "hybrid", "naive", "mix", "bypass"
    # response_type='Multiple Paragraphs', 'Single Paragraph', 'Bullet Points'
    custom_config = QueryConfig(
        mode="mix",
        response_type="Bullet Points",
    )

    if not (query := query.strip()):
        await cl.Message("❌ 查询内容不能为空").send()
        return 
    query_msg = await cl.Message("").send()

    #这是使用minirag， mode: 指令有这些["light", "naive", "mini"] = "mini"
    #response = rag.query(query, param=QueryParam(mode="light"))
    #await cl.Message(f"**知识库回答**\n\n{response}").send()

    # 执行查询
    full_response = ""
    chunk = ""
    try:
        # 使用异步线程执行阻塞操作
        loop = asyncio.get_event_loop()
        for chunk in await loop.run_in_executor(
            None, 
            lambda: list(client.query(question=query, config=custom_config))
        ):
            full_response += chunk
            await query_msg.stream_token(chunk)
            await query_msg.update()
            #print(chunk, end="\n", flush=True)

        query_msg.content = full_response
        await query_msg.update()
    
    # 修改异常处理部分，添加响应内容打印
    except requests.exceptions.RequestException as e:
        error_msg = "⚠️ 知识库服务连接失败，请检查:\n"
        # 处理无响应对象的情况（如连接拒绝）
        if not hasattr(e, "response") or e.response is None:
            error_msg += f"- 服务未运行或配置错误\n- 错误详情: {str(e)}"
        else:
            try:
                error_detail = e.response.json().get("detail", str(e))
            except (json.JSONDecodeError, AttributeError):
                error_detail = e.response.text[:500]  # 截断避免过长
            
            error_msg += (
                f"- 状态码: {getattr(e.response, 'status_code', '未知')}\n"
                f"- 错误类型: {type(e).__name__}\n"
                f"- 详情: {error_detail}"
            )

        # 补充连接问题排查建议
        error_msg += "\n\n排查步骤:\n1. 确认LightRAG服务已启动\n2. 检查端口9721是否监听\n3. 验证API密钥配置"

        if 'query_msg' in locals():
            query_msg.content = error_msg
            await query_msg.update()
        else:
            await cl.Message(content=error_msg).send()


# ======================== 联网搜索模块 ========================
async def handle_web_search(user_input: str):
    web_search_msg = None  # 显式初始化
    try:
        query = user_input
        if not (query := query.strip()):
            await cl.Message("❌ 查询内容不能为空").send()
            return 
        web_search_msg = await cl.Message(f"🔍 正在联网搜索: {query}").send()
        search_result = asyncio.run(search_web(query))
        print ("search_web:\n" , search_result)
        await cl.Message(f"**联网搜索结果**\n\n{search_result}").send()

    except Exception as e:
        # 处理异常情况
        error_msg = f"❌ 联网搜索失败: {str(e)}"
        if 'web_search_msg' in locals():
            web_search_msg.content = error_msg
            await web_search_msg.update()
        else:
            await cl.Message(content=error_msg).send()


# ======================== 联网搜索模块-辅助函数 ========================
async def search_web(query: str) -> str:
    url = f'https://duckduckgo.com/html/?q={urllib.parse.quote(query)}'
    
    # 随机用户代理列表
    USER_AGENTS = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.4 Safari/605.1.15',
        'Mozilla/5.0 (Linux; Android 10; SM-G981B) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.162 Mobile Safari/537.36'
    ]

    try:
        # 随机选择用户代理并设置超时
        response = requests.get(
            url,
            headers={'User-Agent': random.choice(USER_AGENTS)},
            timeout=10  # 10秒超时
        )
        response.raise_for_status()
        
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(response.text, 'html.parser')
        
        results = []
        seen_urls = set()  # 用于URL去重
        
        for result in soup.select('.result__a'):
            title = result.get_text().strip()
            href = result['href']
            
            try:
                # 解析真实URL
                if href.startswith('/l/?uddg='):
                    parsed = urllib.parse.urlparse(href)
                    params = urllib.parse.parse_qs(parsed.query)
                    encoded_url = params.get('uddg', [href])[0]
                    decoded_url = urllib.parse.unquote(encoded_url)
                    
                    # 二次解析清理跟踪参数
                    final_url = urllib.parse.urlparse(decoded_url)
                    clean_query = urllib.parse.parse_qs(final_url.query)
                    # 移除常见跟踪参数
                    for param in ['utm_', 'fbclid', 'gclid']:
                        clean_query = {k: v for k, v in clean_query.items() if not k.startswith(param)}
                    
                    # 重建干净URL
                    final_url = final_url._replace(
                        query=urllib.parse.urlencode(clean_query, doseq=True),
                        fragment=''  # 移除锚点
                    ).geturl()
                else:
                    final_url = urllib.parse.urlparse(href)._replace(query='', fragment='').geturl()

                # 标准化URL并去重
                final_url = final_url.split('#')[0].rstrip('/')  # 统一格式
                if final_url.lower() in seen_urls:
                    continue
                seen_urls.add(final_url.lower())

                # 验证有效URL格式
                if re.match(r'^https?://(?:www\.)?[a-zA-Z0-9-]+\.[a-zA-Z]{2,}', final_url):
                    results.append((title, final_url))
                
            except Exception as e:
                print(f"解析URL时出错: {str(e)}")
                continue

        # 随机排序并限制结果数量
        random.shuffle(results)
        MAX_RESULTS = 10
        formatted_results = [f"🔗 **{title}**\n{url}" for title, url in results[:MAX_RESULTS]]
        
        return "\n\n".join(formatted_results) if formatted_results else "⚠️ 未找到相关结果"

    except requests.exceptions.Timeout:
        return "⏳ 请求超时，请稍后再试"
    except requests.RequestException as e:
        return f"❌ 网络请求失败: {str(e)}"
    except Exception as e:
        return f"⚠️ 处理出错: {str(e)}"
    
# ======================== OpenAI文本处理模块 ========================
async def handle_text_input(history: list, user_input: str):
    try:   
        # 添加用户新消息到历史
        if "$$" in user_input:
            formula_analysis = """
            - **格式规范与任务要求：**
            1. 禁止使用 \(...\) 或 \[...\]。
            2.替换所有行间公式语法 \[...\] 为 $$...$$。
            3.替换所有判别式语法 \(...\) 为 $$...$$。
            4.替换所有系数语法 \(...\) 为 $$...$$。
            5.错误示例输出：\(a^{n+m}\)，修正：$$a^{n+m}$$，应使用$$包裹。
            ## 公式分析
            - **类型**: 分析公式类型等
            - **步骤**: 数学推导等
            - **应用场景**: 学术论文、物理建模等。
            """
            history.append({"role": "user", "content": "请使用中文交流，" + user_input + formula_analysis})
        else:
            history.append({"role": "user", "content": "请使用中文交流，" + user_input})
        
        msg = cl.Message(content="")
        await msg.send()

        response_stream = await client.chat.completions.create(
            model= os.environ.get("LLM_MODEL"),
            messages=history,
            max_tokens=8192,
            temperature=0.7,
            stream=True,
            timeout=300.0  # 增加超时时间
        )
        
        full_response = ""
        token = ""
        async for chunk in response_stream:
            token = chunk.choices[0].delta.content
            if token:
                #print(token, end="\n", flush=True)
                # 直接替换所有目标字符，无需条件判断
                token = token.replace("\\(", "$")  # 替换反斜杠加(
                token = token.replace("\\)", "$")   # 替换反斜杠加)
                token = token.replace("\\[", "$$")   # 替换反斜杠加[
                token = token.replace("\\]", "$$")   # 替换反斜杠加]
                full_response += token
                await msg.stream_token(token)

        print(f"最终full_response:\n{full_response}")
        full_response = full_response.replace("\\(", "$")  # 替换反斜杠加(
        full_response = full_response.replace("\\)", "$")   # 替换反斜杠加)
        full_response = full_response.replace("\\[", "$$")   # 替换反斜杠加[
        full_response = full_response.replace("\\]", "$$")   # 替换反斜杠加]

        msg.content = full_response
        await msg.update()
        
        # 添加AI回复到历史记录
        history.append({"role": "assistant", "content": full_response})
        
        # 更新用户会话中的历史记录（关键步骤）
        cl.user_session.set("conversation_history", history)

        # 返回完整的响应内容
        return full_response
        
    except Exception as e:
        error_msg = f"失败: {str(e)}"
        await cl.Message(content=error_msg).send()
        return None  # 或者根据需求抛出异常
            

# ======================== qa处理模块 ========================
async def qa_text_input(message: cl.Message):
    try: 
        # 从用户会话中获取对话链
        chain = cl.user_session.get("chain")  # ConversationalRetrievalChain
        cb = cl.AsyncLangchainCallbackHandler()

        # 调用对话链处理用户消息
        res = await chain.ainvoke({"question": message}, callbacks=[cb])
        answer = res["answer"]  # 获取回答
        source_documents = res["source_documents"]  # 获取源文档

        text_elements = []  # 用于存储源文档的文本元素

        # 如果有源文档，将其添加到文本元素中
        if source_documents:
            for source_idx, source_doc in enumerate(source_documents):
                source_name = f"source_{source_idx}"
                # 创建文本元素并添加到列表中
                text_elements.append(
                    cl.Text(
                        content=source_doc.page_content, name=source_name, display="side"
                    )
                )
            source_names = [text_el.name for text_el in text_elements]

            # 如果有源文档名称，将其添加到回答中
            if source_names:
                answer += f"\n来源: {', '.join(source_names)}"
            else:
                answer += "\n未找到来源"

        # 发送回答和源文档给用户
        await cl.Message(content=answer, elements=text_elements).send()

    except Exception as e:
        error_msg = f"失败: {str(e)}"
        await cl.Message(content=error_msg).send()



