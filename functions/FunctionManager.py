import inspect
import json
import re

import requests


class FunctionManager:
    def __init__(self, functions=None):
        self.functions = {}
        self.excluded_functions = {"inspect", "create_engine"}  # 添加这行
        if functions:
            for func in functions:
                self.functions[func.__name__] = func

    def add_function(self, func):
        self.functions[func.__name__] = func

    def generate_functions_array(self):
        type_mapping = {
            "str": "string",
            "int": "integer",
            "float": "number",
            "bool": "boolean",
            "list": "array",
            "dict": "object",
        }
        functions_array = []

        for function_name, function in self.functions.items():
            if function_name in self.excluded_functions:  # 添加这行
                continue
            # 获取函数的文档字符串和参数列表
            docstring = function.__doc__
            parameters = inspect.signature(function).parameters

            # 提取函数描述
            docstring_lines = docstring.strip().split("\n") if docstring else []
            function_description = docstring_lines[0].strip() if docstring_lines else ""

            # 解析参数列表并生成函数描述
            function_info = {
                "name": function_name,
                "description": function_description,
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],  # Add a required field
                },
            }

            for parameter_name, parameter in parameters.items():
                # 获取参数的注释
                parameter_annotation = parameter.annotation
                if parameter_annotation == inspect.Parameter.empty:
                    continue

                # 如果注解是一个类型，获取它的名字
                # 如果注解是一个字符串，直接使用它
                if isinstance(parameter_annotation, type):
                    parameter_annotation_name = parameter_annotation.__name__.lower()
                else:
                    parameter_annotation_name = parameter_annotation.lower()

                # 提取参数描述
                param_description_pattern = rf"{parameter_name}: (.+)"
                param_description_match = [
                    re.search(param_description_pattern, line)
                    for line in docstring_lines
                ]
                param_description = next(
                    (match.group(1) for match in param_description_match if match), ""
                )

                # 添加参数描述
                parameter_description = {
                    "type": type_mapping.get(
                        parameter_annotation_name, parameter_annotation_name
                    ),
                    "description": param_description,
                }
                function_info["parameters"]["properties"][
                    parameter_name
                ] = parameter_description

                # If the parameter has no default value, add it to the required field.
                if parameter.default == inspect.Parameter.empty:
                    function_info["parameters"]["required"].append(parameter_name)

            functions_array.append(function_info)

        return functions_array

    async def call_function(self, function_name, args_dict):
        """
        异步调用指定名称的函数，并传入参数字典。

        Args:
            function_name (str): 要调用的函数名称。
            args_dict (dict): 传递给函数的参数字典。

        Returns:
            str: 函数的返回结果，如果是复杂类型（元组、列表、字典、集合），则转换为 JSON 字符串返回。

        Raises:
            ValueError: 如果函数名称不存在于注册的函数列表中。
        """
        if function_name not in self.functions:
            raise ValueError(f"Function '{function_name}' not found")

        function = self.functions[function_name]
        print("函数：\n", function)  # 打印函数对象信息（调试用）
        print("参数：\n", args_dict)  # 打印参数字典（调试用）
        res = await function(**args_dict)  # 异步调用函数并传入参数
        
        # 处理返回结果：
        # 1. 如果是字符串，直接返回
        if isinstance(res, str):
            return res
        # 2. 如果是元组、列表、字典或集合，转换为 JSON 字符串
        elif isinstance(res, (tuple, list, dict, set)):
            if isinstance(res, set):
                res = list(res)  # 将集合转换为列表（JSON 不支持集合类型）
            try:
                # 转换为 JSON 字符串（ensure_ascii=False 确保中文字符正常显示）
                res_str = json.dumps(res, ensure_ascii=False)
                # 再解析回对象，确保格式正确（避免多层嵌套的 JSON 字符串问题）
                res_obj = json.loads(res_str)
                res = json.dumps(res_obj, ensure_ascii=False)
                print("json字符串res: \n", res)  # 打印转换后的 JSON（调试用）
            except json.JSONDecodeError as e:
                # 如果 JSON 转换失败，返回错误信息
                print(f"JSON 解析错误: {str(e)}")
                res = json.dumps({"error": f"JSON 解析错误: {str(e)}"}, ensure_ascii=False)
        return res  # 返回最终结果
    
    
# 测试
def get_current_weather(location: str, unit: str = "celsius"):
    """
    Get the current weather in a given location.

    Parameters:
        - location: The city and state, e.g. San Francisco, CA
        - unit: The unit of temperature (celsius or fahrenheit)
    """
    return {"temperature": "22", "unit": "celsius", "description": "Sunny"}


# 定义一个方法来根据传进来的url地址，读取网页的内容
def get_html(url: str):
    # 定义一个请求头，模拟浏览器访问
    """
    Get the html content of the url.if user provide the url,then return the html content of the url.
    Parameters:
        url: The url of the website. (required)
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36"
    }
    # 发送请求
    response = requests.get(url, headers=headers)
    # 返回网页内容
    return response.text


if __name__ == "__main__":
    pass
    # function_manager = FunctionManager(functions=[search_by_bard])
    # functions_array = function_manager.generate_functions_array()
    # print(functions_array)

    # result = function_manager.call_function('get_current_weather', {'location': 'San Francisco, CA', 'unit': 'celsius'})
    # print(result)
