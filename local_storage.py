import os
from minio import Minio
from minio.error import S3Error
from typing import Dict, Optional
from chainlit.data.storage_clients.base import BaseStorageClient
import io

class MinIOStorageClient(BaseStorageClient):
    def __init__(self, endpoint: str, access_key: str, secret_key: str, bucket_name: str, secure: bool = False):
        self.client = Minio(endpoint, access_key, secret_key, secure=secure)
        self.bucket_name = bucket_name
        self.endpoint = endpoint  # 保存 endpoint 用于生成 URL
        if not self.client.bucket_exists(bucket_name):
            self.client.make_bucket(bucket_name)

    async def upload_file(self, object_key: str, data: bytes, mime: str, overwrite: bool = False) -> Optional[Dict[str, str]]:
        try:
            # 将字符串数据转换为字节对象
            if isinstance(data, str):
                data = data.encode('utf-8')
            
            # 将字节数据包装为 BytesIO 对象
            data_stream = io.BytesIO(data)
                        
            self.client.put_object(
                self.bucket_name,
                object_key,
                data_stream,
                length=len(data),
                content_type=mime
            )
            # 使用保存的 endpoint 生成 URL
            url = f"http://{self.endpoint}/{self.bucket_name}/{object_key}"
            return {"url": url, "object_key": object_key}
        except S3Error as e:
            print(f"上传失败: {e}")
            return None

    async def delete_file(self, object_key: str) -> bool:
        try:
            self.client.remove_object(self.bucket_name, object_key)
            return True
        except S3Error as e:
            print(f"删除失败: {e}")
            return False

    async def get_read_url(self, object_key: str) -> str:
        try:
            # 使用保存的 endpoint 生成 URL
            url = f"http://{self.endpoint}/{self.bucket_name}/{object_key}"
            return url
        except S3Error as e:
            print(f"获取 URL 失败: {e}")
            raise FileNotFoundError(f"文件未找到: {object_key}")