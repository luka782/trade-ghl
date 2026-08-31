# 自定义因子

将 `_example.py` 复制为不以下划线开头的新文件，例如 `reversal_5.py`，然后修改
`FactorMetadata` 和 `compute()`。文件必须导出：

```python
FACTOR = YourFactor()
```

或：

```python
FACTORS = (FirstFactor(), SecondFactor())
```

开发模式下保存文件会触发后端重载；否则请重启后端。加载失败不会阻止服务启动，错误会
出现在 `GET /api/factors` 的 `warnings` 字段。

约束：

- 输出 `Series` 必须与输入行情保持相同索引。
- T 日因子只能读取 T 日及以前的数据，不得使用负数 `shift`。
- 在 `required_columns` 中声明所需字段。
- 需要的历史窗口写入 `lookback`，API 会额外加载预热数据。
