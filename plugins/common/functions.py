import os
import chainlit as cl
import asyncio
from typing import Optional, Dict  # 添加缺失的导入
from playwright.async_api import async_playwright
import asyncio
import re
import time  # 新增的导入
import random  # 新增导入语句

async def get_html(url: str):
    # 定义一个请求头，模拟浏览器访问
    """
    When the user's question mentions one or more URL website, you need to analyze and summarize the content of these web pages, and you can call this function to analyze and summarize.
    Parameters:
        url: URL website as a comma-separated string.(required)
    """
    # 改进正则表达式匹配模式
    url_pattern = re.compile(
        r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+[/\w .&?#%=;()$-]*'  # 扩展匹配范围
    )
    urls = url_pattern.findall(url)
    
    if not urls:
        raise ValueError("未检测到有效URL")

    playwright = await async_playwright().start()
    browser = await playwright.chromium.launch(
        headless=True,
        args=['--disable-dev-shm-usage', '--no-sandbox'],
        # 添加浏览器上下文参数
        ignore_default_args=['--enable-automation']
    )
    
    semaphore = asyncio.Semaphore(3)

    async def fetch_page(u):
        async with semaphore:
            page = await browser.new_page()
            try:
                # 添加浏览器User-Agent模拟
                await page.set_extra_http_headers({
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36'
                })
                
                # 添加详细调试日志
                print(f"[DEBUG] 开始处理URL: {u}")
                start_time = time.time()
                
                # 改进等待策略和重试机制
                for attempt in range(3):
                    try:
                        response = await page.goto(
                            u, 
                            wait_until="domcontentloaded",  # 改为更可靠的等待策略
                            timeout=60000,
                            referer='https://www.google.com/'
                        )
                        if response.status >= 400:
                            raise Exception(f"HTTP错误 {response.status}")
                            
                        # 添加二次内容验证
                        content = await page.inner_text('body')
                        if not content.strip():
                            raise ValueError("页面内容为空")
                            
                        print(f"[SUCCESS] {u} 抓取成功 ({time.time()-start_time:.2f}s)")
                        return f"网址:{u}\n网页内容:\n{content}"
                        
                    except Exception as e:
                        print(f"[RETRY {attempt+1}] {u} 错误: {str(e)}")
                        if attempt == 3:
                            raise
                        await asyncio.sleep(1)
                        
            except Exception as e:
                print(f"[FAILED] {u} 最终失败: {str(e)}")
                return f"{u}\n抓取失败: {type(e).__name__} - {str(e)}"
            finally:
                await page.close()

    tasks = [fetch_page(u) for u in urls]
    results = await asyncio.gather(*tasks)
    
    await browser.close()
    await playwright.stop()
    
    return '\n\n'.join(results)


async def need_file_upload(user_message: str) -> Optional[Dict]:
    """
    When the user's question mentions upload files, you need to upload files, you can call this function.
    Parameters:
        user_message: Identifying keywords for the intent of uploading files.(required)
    """
    user_message = cl.user_session.get("user_message", [])
    # 识别上传意图的关键词列表（扩展关键词）
    upload_keywords = ["上载"]
    filetype_keywords = ["文件"]
    
    has_upload = any(keyword in user_message for keyword in upload_keywords)
    has_filetype = any(keyword in user_message for keyword in filetype_keywords)
    
    if not (has_upload and has_filetype):
        return None

    try:
        # 请求文件上传（支持多文件选择）
        files = await cl.AskFileMessage(
            content="请选择要处理的文件（支持多选）",
            accept=["*/*"],  # 允许所有文件类型
            max_size_mb=100,
            timeout=300,
            max_files=5
        ).send()

        if not files:
            return {"status": "cancelled", "message": "用户取消操作"}

        results = []
        for file in files:
            file_path = os.path.abspath(file.name)  # 使用绝对路径
            
            # 使用带缓冲区的分块复制（处理大文件）
            try:
                with open(file.path, 'rb') as src_file:
                    with open(file_path, 'wb') as dst_file:
                        # 分块读取（每次2MB）
                        while True:
                            chunk = src_file.read(2 * 1024 * 1024)
                            if not chunk:
                                break
                            dst_file.write(chunk)
            except Exception as e:
                await cl.Message(f"❌ 文件 {file.name} 复制失败: {str(e)}").send()
                continue

            # 验证文件完整性
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"目标文件 {file_path} 未创建成功")
                
            # 删除临时文件（带重试机制）
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    os.remove(file.path)
                    break
                except PermissionError:
                    if attempt == max_retries - 1:
                        await cl.Message(f"⚠️ 无法清理临时文件 {file.name}").send()
                    await asyncio.sleep(0.5 * (attempt + 1))

            results.append({
                "filename": file.name,
                "path": file_path,
                "size": f"{os.path.getsize(file_path)/1024/1024:.2f}MB",
                "type": file.type
            })

        return {
            "status": "上传成功",
            "files": results,
            "message": f"共上传了{len(results)}个文件",
        }

    except Exception as e:
        await cl.Message(f"❌ 文件处理失败: {str(e)}").send()
        return {
            "status": "error",
            "error": str(e),
            "advice": "请检查文件是否被其他程序占用"
        }


async def need_rename_file(old_path: str, new_path: str):
    """
    When the user's question refers to managing files and requires file rename, you can invoke this function.
    Parameters: old_path: The old path of the file.(required)
    new_path: The new path of the file.(required)
    """
    # 判断old_path是否存在
    if not os.path.exists(old_path):
        return {'description': f"{old_path} is not exist"}
    # 判断new_path是否存在
    if os.path.exists(new_path):
        return {'description': f"{new_path} is already exist"}
    # 重命名文件
    os.rename(old_path, new_path)
    return {'description': f"rename file {old_path} to {new_path} success"}


async def show_images(paths: str):
    """
    If your return contains images in png or jpg format, you can call this function to display the images.
    Parameters: paths: The paths of the images as a comma-separated string.(required)
    """
    path_list = paths.split(',')
    elments = []
    for i, path in enumerate(path_list):
        tmp_image = cl.Image(name=f"image{i}",
                             path=path.strip(),
                             display="inline")
        tmp_image.size = "large"
        elments.append(tmp_image)

    await cl.Message(content="图片已经显示成功了",
                     elements=elments).send()  # type: ignore

    return {"description": "图片已经显示成功了，下面的回复中不再需要展示它了"}