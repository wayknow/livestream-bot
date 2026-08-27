# 直播话术助手 — 项目决策日志

记录项目中的关键决策、原因和结果。

---

## 2026-08-27：项目初始化

### 决策 1：项目定位

**选择**：直播带货话术助手

**备选方案**：
- 客服话术生成器
- 营销文案助手
- 故事创作助手

**原因**：
- 用户有直播文案研究背景，效果直观
- 数据容易构造（品类+卖点+风格）
- 训完能直接用，验证快

---

### 决策 2：模型选择

**选择**：
- SFT：`Qwen/Qwen2.5-7B-Instruct`
- DPO：`Qwen/Qwen2.5-3B-Instruct`

**备选方案**：
- DeepSeek-R1-Distill-Qwen-7B（推理型）
- Qwen2.5-14B-Instruct（更大）

**原因**：
- Qwen 中文能力强，7B 在 24GB 上很舒服
- DPO 需同时加载 policy + reference，3B 刚好够
- 14B 可以但慢，先玩熟 7B

---

### 决策 3：训练框架

**选择**：
- SFT：mlx-lm（Apple Silicon 原生加速）
- DPO：TRL + PyTorch MPS

**备选方案**：
- 全用 TRL（transformers 生态）
- 全用 mlx-lm

**原因**：
- mlx-lm 对 Mac 优化最好，LoRA 训练快
- TRL 的 DPO 实现更成熟，mlx-lm 的 DPO 支持较新
- 混合使用取各自优势

---

### 决策 4：数据量

**选择**：
- SFT：200 条（50 手写 + 150 扩充）
- DPO：50-100 条

**备选方案**：
- 只写 50 条，不扩充
- 写 500+ 条

**原因**：
- 24GB 跑 600 iters 很快，200 条 × 3 epoch 合理
- 质量 > 数量，10 条精心设计的比 1000 条垃圾有效
- 扩充用模板替换，保持质量一致

---

### 决策 5：项目目录

**选择**：`/Users/xiaoxiao/work/finetuning/livestream-bot/`

**备选方案**：`~/ai-lab/livestream-bot/`

**原因**：
- 与用户其他项目（snapmark）在同一 work 目录下
- 路径更短，方便操作

---

## 待决策

- [ ] Week 1：先手写 50 条样本，还是先写扩充脚本？
- [ ] Week 2：是否需要先下载模型测试基座效果？
- [ ] Week 4：是否需要转 GGUF 格式用 llama.cpp 部署？
