import abc
import sys
import io
import ast
import subprocess
from contextlib import redirect_stdout
from loguru import logger
import os
import traceback

logger.configure(
    handlers=[
        {
            "sink": sys.stderr,
            "format": "<green>{time}</green> <level>{message}</level>",
            "colorize": True,
        }
    ]
)


# 确保 ./tmp 目录存在
tmp_dir = os.path.abspath(os.path.join(".", "tmp"))
os.makedirs(tmp_dir, exist_ok=True)

# 设置默认保存路径
os.chdir(tmp_dir)


class Executor(abc.ABC):
    @abc.abstractmethod
    def execute(self, code: str) -> str:
        pass


class PythonExecutor(Executor):
    def __init__(self):
        self.locals = {}

    def execute(self, code: str) -> str:
        logger.info("Executing Python code:\n{}", code)
        output = io.StringIO()
        
        try:
            tree = ast.parse(code, mode="exec")
            with redirect_stdout(output):
                # 添加__builtins__以支持模块导入
                exec_locals = {'__builtins__': __builtins__}
                
                compiled = compile(tree, "<string>", "exec")
                exec(compiled, exec_locals, exec_locals)
                
                # 合并前保留原有locals内容
                self.locals.update(exec_locals)
                
                # 处理表达式时使用合并后的完整上下文
                for node in tree.body:
                    if isinstance(node, ast.Expr):
                        expr = compile(ast.Expression(node.value), "<string>", "eval")
                        result = eval(expr, self.locals)
                        if result is not None:
                            print(result)
        except Exception as e:
            logger.error("Error executing Python code", exc_info=True)  # 添加完整堆栈跟踪
            return f"Error executing Python code:\n\n{str(e)}\n\nFull traceback:\n{traceback.format_exc()}"
            
        return f"Success executing Python code:\n\n{output.getvalue()}"


class CppExecutor(Executor):
    def execute(self, code: str) -> str:
        cpp_file = "./tmp/script.cpp"
        with open(cpp_file, "w") as f:
            f.write(code)
        try:
            subprocess.run(["g++", cpp_file, "-o", "./tmp/a.out"], check=True)
            output = subprocess.run(
                ["./tmp/a.out"], capture_output=True, text=True, check=True
            )
            return output.stdout
        except subprocess.CalledProcessError as e:
            # Here we include e.stderr in the output.
            raise subprocess.CalledProcessError(e.returncode, e.cmd, output=e.stderr)
        

class RustExecutor(Executor):
    def execute(self, code: str) -> str:
        rust_file = "./tmp/script.rs"
        with open(rust_file, "w") as f:
            f.write(code)
        try:
            subprocess.run(["rustc", rust_file, "-o", "./tmp/script"], check=True)
            output = subprocess.run(
                ["./tmp/script"], capture_output=True, text=True, check=True
            )
            return output.stdout
        except subprocess.CalledProcessError as e:
            # Here we include e.stderr in the output.
            raise subprocess.CalledProcessError(e.returncode, e.cmd, output=e.stderr)
