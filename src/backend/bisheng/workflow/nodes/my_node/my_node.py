from typing import Any, Dict
from bisheng.workflow.nodes.base import BaseNode


class MyNode(BaseNode):
    """
    你的节点名称 - 简要描述节点功能
    """

    def __init__(self, *args, **kwargs):
        """初始化节点"""
        super().__init__(*args, **kwargs)
        
        # 从节点配置中获取参数
        # self.node_params 包含前端传入的所有参数
        self._param1 = self.node_params.get('param1', '')
        self._param2 = self.node_params.get('param2', 0)
        
        # 初始化其他必要的资源
        # 例如：API客户端、数据库连接等

    def _run(self, unique_id: str) -> Dict[str, Any]:
        """
        核心执行逻辑
        
        Args:
            unique_id: 本次执行的唯一标识
            
        Returns:
            Dict: 返回结果字典，key为输出变量名，value为变量值
                 这些输出可以被其他节点通过 node_id.key 引用
        """
        # 1. 获取其他节点的变量（如果需要）
        # input_value = self.get_other_node_variable('input_node.output_key')
        
        # 2. 执行核心业务逻辑
        try:
            result = self._execute_business_logic()
            
            # 3. 如果需要输出给用户（可选）
            # if self._output_user:
            #     self.graph_state.save_context(content=result, msg_sender='AI')
            
            # 4. 返回输出结果
            return {
                'output': result,  # 输出变量，其他节点可通过 node_id.output 引用
                # 可以返回多个输出
                # 'output2': result2,
            }
            
        except Exception as e:
            # 错误处理
            raise Exception(f"节点执行失败: {str(e)}")

    def _execute_business_logic(self) -> Any:
        """
        业务逻辑实现（可选，用于代码组织）
        """
        # 实现具体的业务逻辑
        pass

    def parse_log(self, unique_id: str, result: dict) -> Any:
        """
        解析节点执行日志（用于前端展示）
        
        Returns:
            List[List[Dict]]: 日志数据结构
            外层列表：轮次（如果节点执行多次）
            内层列表：每轮的日志条目
            Dict: 单条日志
                - key: 变量名或标签
                - value: 值
                - type: 类型 (params/variable/tool/key/file)
        """
        return [[
            {
                "key": "input_param",
                "value": self._param1,
                "type": "params"
            },
            {
                "key": f"{self.id}.output",
                "value": result.get('output', ''),
                "type": "variable"
            }
        ]]

    # 可选方法：如果需要处理用户输入
    # def handle_input(self, user_input: dict) -> Any:
    #     """处理用户输入"""
    #     self.node_params.update(user_input)

    # 可选方法：如果节点需要用户交互
    # def get_input_schema(self) -> Any:
    #     """返回需要用户输入的表单描述"""
    #     return None