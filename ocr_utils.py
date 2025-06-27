import cv2
import numpy as np
from PIL import Image
import asyncio
from paddleocr import PaddleOCR

async def process_image_ocr(image_path):
    try:
        # 打开图片
        image = Image.open(image_path)
        
        # 放大图片（例如放大3倍）
        width, height = image.size
        image = image.resize((width * 10, height * 10), Image.BILINEAR)
        
        # 将图片转换为 numpy 数组
        image_np = np.array(image)
        
        # 图像预处理：灰度化、二值化、去噪
        gray_image = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
        
        # 使用Otsu阈值化
        _, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 形态学操作：膨胀和腐蚀
        kernel = np.ones((2, 2), np.uint8)  # 使用较小的核
        dilated_image = cv2.dilate(binary_image, kernel, iterations=1)
        eroded_image = cv2.erode(dilated_image, kernel, iterations=1)
        
        # 去噪
        denoised_image = cv2.fastNlMeansDenoising(eroded_image, h=10)
        
        # 使用PaddleOCR进行识别
        def _sync_ocr():
            ocr = PaddleOCR(use_angle_cls=True, lang="ch")  # 使用中文模型
            result = ocr.ocr(denoised_image, cls=True)
            ocr_text = [line[1][0] for line in result[0]]  # 提取识别结果中的文本
            return '\n'.join(ocr_text)
        
        # 将同步阻塞操作放入线程池中执行
        loop = asyncio.get_event_loop()
        ocr_text = await loop.run_in_executor(None, _sync_ocr)
        
        return ocr_text
    except Exception as e:
        return f"OCR处理错误: {str(e)}"