import os
import tempfile
import asyncio
import chardet
import fitz  # PyMuPDF
import textract
from langchain_community.document_loaders import PyMuPDFLoader  # 修改后的导入语句
from ocr_utils import process_image_ocr

def process_file(file_path):
    text = ""
    
    if file_path.lower().endswith('.txt'):
        # 增强的文本文件处理
        with open(file_path, 'rb') as f:
            raw_data = f.read()
            encodings_to_try = [
                'utf-8', 'gb18030', 'gbk', 'gb2312',
                'big5', 'utf-16', 'utf-16le', 'utf-16be',
                'iso-8859-1', 'windows-1252', 'cp1254'
            ]

            for encoding in encodings_to_try:
                try:
                    text = raw_data.decode(encoding, errors='replace')
                    break
                except UnicodeDecodeError:
                    continue

            if not text:
                detected = chardet.detect(raw_data)
                if detected['confidence'] > 0.7:
                    encoding = detected['encoding'].lower().replace('-', '')
                    encoding_alias = {
                        'windows1254': 'cp1254',
                        'iso88591': 'iso-8859-1',
                        'macroman': 'mac-roman',
                        'gb2312': 'gb18030'
                    }
                    encoding = encoding_alias.get(encoding, encoding)
                    try:
                        text = raw_data.decode(encoding, errors='replace')
                    except (UnicodeDecodeError, LookupError):
                        pass

            if not text:
                try:
                    text = raw_data.decode('utf-8', errors='ignore')
                except Exception as final_error:
                    error_msg = f"终极解码失败: {str(final_error)}"
                    raise ValueError(error_msg)

    elif file_path.lower().endswith('.pdf'):
        # 使用 PyMuPDFLoader 加载 PDF 文件
        loader = PyMuPDFLoader(str(file_path))
        docs = loader.load()

        # 假设 docs 是包含所有 Document 对象的列表
        combined_content = ""

        # 遍历每个 Document 对象
        for doc in docs:
            # 提取 page_content 并添加到 combined_content 中
            if doc.page_content.strip():  # 检查 page_content 是否为空
                combined_content += doc.page_content + "\n"  # 添加换行符以分隔不同页的内容
            else:
                # 如果 page_content 为空，使用 easyocr 识别该页
                pdf_document = fitz.open(file_path)
                page = pdf_document.load_page(doc.metadata.get("page", 0))  # 获取当前页

                # 将 PDF 页面转换为图片
                pix = page.get_pixmap()
                temp_image_path = tempfile.mktemp(suffix=".png")
                pix.save(temp_image_path)

                # 使用 easyocr 识别图片内容
                loop = asyncio.get_event_loop()
                ocr_text = loop.run_until_complete(process_image_ocr(temp_image_path))

                # 将 OCR 识别的内容添加到 combined_content
                combined_content += ocr_text + "\n"

                # 删除临时图片文件
                os.remove(temp_image_path)

        # 打印或返回合并后的字符串
        text = combined_content
        print(text)
    
    else:
        # 其他文件类型使用 textract 处理
        text = textract.process(file_path).decode('utf-8')
    
    return text