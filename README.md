# HPAL Project - Core Code Package

## 项目结构 / Project Structure

```
HPAL_Code_Package/
├── shared_code/                    # 共享核心代码 (所有实验共用)
│   ├── active_func.py              # 基础主动学习策略
│   ├── active_func_HLEU.py         # HLEU复合评分策略 (核心算法)
│   ├── s3dis_main.py               # 基础训练入口
│   ├── s3dis_main_HLEU.py          # HLEU训练入口 (主入口)
│   ├── s3dis_test_HLEU.py          # 测试脚本
│   ├── config.py                   # 配置文件
│   ├── data_base.py / data_base_HLEU.py  # 数据加载
│   ├── helper_tool.py / helper_utils.py  # 工具函数
│   ├── HierarchicalMatrixReader.py # 层次矩阵读取
│   ├── attention_model_trainer.py  # 注意力模型训练
│   ├── attention_model_test.py     # 注意力模型测试
│   ├── label_reader.py / learner_reader.py  # 标签/学习器读取
│   ├── run.sh / compile_op.sh / enviro.sh  # 运行脚本
│   ├── Mink/                       # MinkUNet 模型
│   │   ├── base_agent.py / base_agent_HLEU.py
│   │   ├── dataloader/             # 数据加载器
│   │   ├── models/                 # 网络模型 (MinkUNet)
│   │   └── utils/                  # mIoU 计算
│   ├── data_preparation/           # 数据预处理
│   ├── tools/                      # 消融实验分析工具
│   └── utils/                      # C++加速模块 (KNN, 子采样)
│
├── experiment_full_composite/      # 实验组 (Full Composite)
│   │   策略: HLEU_delete_dot
│   │   评分: composite = (MI + Rep + Edge) / 3
│   ├── config/                     # 实验配置
│   └── results/                    # 训练日志与结果
│
├── experiment_ablation/            # 消融组 (Ablation)
│   │   策略: HLEU_MI_only
│   │   评分: 仅使用 HLEU-conditioned MI 分支
│   ├── config/                     # 实验配置
│   └── results/                    # 训练日志与结果
│
└── experiment_equal_weight/        # 等权重组 (Equal Weight)
    │   策略: HLEU_delete_dot
    │   评分: (MI + Rep + Edge) / 3 (等权重)
    ├── config/                     # 实验配置
    └── results/                    # 训练日志与结果
```

## 三组实验说明 / Three Experiment Groups

### 1. 实验组 (Full Composite) - `experiment_full_composite/`
- **策略**: `HLEU_delete_dot`
- **评分公式**: `composite(x) = (MI(x) + Rep(x) + Edge(x)) / 3`
- **说明**: 完整的复合评分方法，使用互信息(MI)、代表性(Rep)和边缘性(Edge)三个分支的等权重融合
- **结果路径**: `/root/autodl-tmp/Training_results/Run_equal_value`

### 2. 消融组 (Ablation) - `experiment_ablation/`
- **策略**: `HLEU_MI_only`
- **评分公式**: 仅使用 HLEU-conditioned MI 分支
- **说明**: 消融实验，移除 Rep 和 Edge 分支，仅保留 MI 分支，用于验证完整复合评分的贡献
- **结果路径**: `/root/autodl-tmp/Training_results/Ablation_experiment`

### 3. 等权重组 (Equal Weight) - `experiment_equal_weight/`
- **策略**: `HLEU_delete_dot`
- **评分公式**: `composite(x) = (MI(x) + Rep(x) + Edge(x)) / 3`
- **说明**: 三个分支使用相同权重(1/3)的基准实验
- **结果路径**: `/root/autodl-tmp/Training_results/Run_equal_value`

## 核心算法 / Core Algorithm

核心评分逻辑在 `shared_code/active_func_HLEU.py` 中实现:
- **MI (Mutual Information)**: 基于 MC-Dropout 的互信息估计
- **Rep (Representativeness)**: 基于 KMeans 聚类的代表性评分
- **Edge (Edge-case)**: 基于密度的边缘稀有性评分
- **HLEU/RUP**: 层次不确定性条件化，作为全局不确定性参考

## 运行方式 / How to Run

```bash
# 激活环境
conda activate HPAL

# 运行实验 (修改 config.py 中的 active_strategy 和 base_path)
python s3dis_main_HLEU.py --test_area 5 --mode AL_train
```
