import asyncio
import subprocess
import sys
import chainlit as cl
import json  # 新增导入 ast 模块
from .executor import PythonExecutor


async def python_exec(code: str, language: str = "python"):
    """
    Execute code. \nNote: This endpoint current supports a REPL-like environment for Python only.\n\nArgs:\n    request (CodeExecutionRequest): The request object containing the code to execute.\n\nReturns:\n    CodeExecutionResponse: The result of the code execution.
    Parameters: code: (str, required): A Python code snippet for execution in a Jupyter environment, where variables and imports from previously executed code are accessible. The code must not rely on external variables/imports not available in this environment, and must print a dictionary `{"type": "<type>", "path": "<path>", "status": "<status>"}` as the last operation. `<type>` can be "image", "file", or "content", `<path>` is the file path (not needed if `<type>` is "content"), `<status>` indicates execution status. Display operations should save output as a file with path returned in the dictionary. If tabular data is generated, it should be directly returned as a string. The code must end with a `print` statement.the end must be print({"type": "<type>", "path": "<path>", "status": "<status>"})
    Parameters: language: (str, required): python
    """
    myexcutor = PythonExecutor()
    # 检查并删除代码开头可能存在的 ```py 或 ``` 字符
    if code.startswith('```py'):
        code = code[5:].lstrip()  # 删除 ```py 并去除左侧空白字符
    elif code.startswith('```'):
        code = code[3:].lstrip()  # 删除 ``` 并去除左侧空白字符
    
    # 检查并删除代码尾部可能存在的 ``` 字符
    if code.endswith('```'):
        code = code[:-3].rstrip()  # 删除 ``` 并去除右侧空白字符
    code_output = myexcutor.execute(code)
    #print(f"REPL execution result:\n{code}")
    return str(code_output)


async def need_install_package(package_name: str) -> dict:
    """
    If the user's question mentions installing packages, and the packages need to be installed,
    you can call this function.
    Parameters: package_name: The name of the package.(required)
    """
    # check if package is already installed
    cmd_check = [sys.executable, "-m", "pip", "show", package_name]
    proc = subprocess.Popen(cmd_check, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, _ = proc.communicate()
    if out:
        return {"description": f"{package_name} is already installed"}

    # install package if it's not installed
    cmd_install = [sys.executable, "-m", "pip", "install", package_name]
    process = await asyncio.create_subprocess_exec(
        *cmd_install, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )

    stdout, stderr = await process.communicate()

    if process.returncode != 0:
        await cl.Message(content=f"Failed to install {package_name}.").send()
        return {"description": f"Error installing {package_name}: {stderr.decode()}"}
    await cl.Message(content=f"Successfully installed {package_name}.").send()
    return {"description": f"{package_name} has been successfully installed"}