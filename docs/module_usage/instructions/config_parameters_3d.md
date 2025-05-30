---
comments: true
---

# PaddleX 3d任务模型配置文件参数说明

# Global
<table>
<thead>
<tr>
<th>参数名</th>
<th>数据类型</th>
<th>描述</th>
<th>默认值</th>
</tr>
</thead>
<tbody>
<tr>
<td>model</td>
<td>str</td>
<td>指定模型名称</td>
<td>yaml文件中指定的模型名称</td>
</tr>
<tr>
<td>mode</td>
<td>str</td>
<td>指定模式（check_dataset/train/evaluate/export/predict）</td>
<td>check_dataset</td>
</tr>
<tr>
<td>dataset_dir</td>
<td>str</td>
<td>数据集路径</td>
<td>yaml文件中指定的数据集路径</td>
</tr>
<tr>
<td>device</td>
<td>str</td>
<td>指定使用的设备</td>
<td>yaml文件中指定的设备id</td>
</tr>
<tr>
<td>output</td>
<td>str</td>
<td>输出路径</td>
<td>"output"</td>
</tr>
<tr>
<td>load_cam_from</td>
<td>str</td>
<td>cam分支的预训练参数路径</td>
<td>yaml文件中指定的cam分支的预训练参数路径</td>
</tr>
<tr>
<td>load_lidar_from</td>
<td>str</td>
<td>lidar分支的预训练参数路径</td>
<td>yaml文件中指定的lidar分支的预训练参数路径</td>
</tr>
<tr>
<td>datart_prefix</td>
<td>bool</td>
<td>数据集路径是否需要加上根路径</td>
<td>True</td>
</tr>
<tr>
<td>version</td>
<td>str</td>
<td>数据集版本号</td>
<td>"mini"</td>
</tr>
</tbody>
</table>

# CheckDataset
<table>
<thead>
<tr>
<th>参数名</th>
<th>数据类型</th>
<th>描述</th>
<th>默认值</th>
</tr>
</thead>
<tbody>
<tr>
<td>convert.enable</td>
<td>bool</td>
<td>是否进行数据集格式转换</td>
<td>False (当前不支持)</td>
</tr>
<tr>
<td>split.enable</td>
<td>bool</td>
<td>是否重新划分数据集</td>
<td>False (当前不支持)</td>
</tr>
</tbody>
</table>

# Train
### 3d任务公共参数
<table>
<thead>
<tr>
<th>参数名</th>
<th>数据类型</th>
<th>描述</th>
<th>默认值</th>
</tr>
</thead>
<tbody>
<tr>
<td>epochs_iters</td>
<td>int</td>
<td>模型对训练数据的重复学习次数</td>
<td>yaml文件中指定的重复学习次数</td>
</tr>
<tr>
<td>batch_size</td>
<td>int</td>
<td>批大小</td>
<td>yaml文件中指定的批大小</td>
</tr>
<tr>
<td>learning_rate</td>
<td>float</td>
<td>初始学习率</td>
<td>yaml文件中指定的初始学习率</td>
</tr>
<tr>
<td>warmup_steps</td>
<td>str</td>
<td>训练模型的开始阶段，进行热身训练的步数</td>
<td>yaml文件中指定的热身训练的步数</td>
</tr>
</tbody>
</table>

# Evaluate
<table>
<thead>
<tr>
<th>参数名</th>
<th>数据类型</th>
<th>描述</th>
<th>默认值</th>
</tr>
</thead>
<tbody>
<tr>
<td>batch_size</td>
<td>int</td>
<td>批大小</td>
<td>yaml文件中指定的批大小
</td>
</tr>
<tr>
<td>weight_path</td>
<td>str</td>
<td>评估模型路径</td>
<td>默认训练产出的本地路径，当指定为None时，表示使用官方权重</td>
</tr>
</tbody>
</table>

# Export
<table>
<thead>
<tr>
<th>参数名</th>
<th>数据类型</th>
<th>描述</th>
<th>默认值</th>
</tr>
</thead>
<tbody>
<tr>
<td>weight_path</td>
<td>str</td>
<td>导出模型的动态图权重路径</td>
<td>默认训练产出的本地路径，当指定为None时，表示使用官方权重</td>
</tr>
</tbody>
</table>

# Predict
<table>
<thead>
<tr>
<th>参数名</th>
<th>数据类型</th>
<th>描述</th>
<th>默认值</th>
</tr>
</thead>
<tbody>
<tr>
<td>batch_size</td>
<td>int</td>
<td>预测批大小</td>
<td>yaml文件中指定的预测批大小</td>
</tr>
<tr>
<td>model_dir</td>
<td>str</td>
<td>预测模型路径</td>
<td>默认训练产出的本地推理模型路径，当指定为None时，表示使用官方权重</td>
</tr>
<tr>
<td>input</td>
<td>str</td>
<td>预测输入路径</td>
<td>yaml文件中指定的预测输入路径</td>
</tr>
<tr>
<td>kernel_option.run_mode</td>
<td>str</td>
<td>推理引擎设置，如: "paddle"</td>
<td>paddle</td>
</tr>
<tr>
<td>use_hpip</td>
<td>bool</td>
<td>是否启用高性能推理插件</td>
<td></td>
</tr>
<tr>
<td>hpip_config</td>
<td>dict | None</td>
<td>高性能推理配置</td>
<td></td>
</tr>
</tbody>
</table>
