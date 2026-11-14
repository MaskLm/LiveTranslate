# 更新日志

## 2026-11-14
- 新增日语振假名: 日语语音识别的原文可在汉字上方显示平假名读音, 支持悬浮窗与 OBS 字幕窗口 (样式标签页 → 振假名)。读音由 fugashi + unidic-lite 结合上下文分析得出, 正确切分送假名 (食べる → 食[た]べる); 可选汉字上方注音或括号内联、读音大小及生效位置。在 UI 线程之外计算, 译文流式更新时不会重复计算
- 扩充 ASR 模型列表: Whisper 引擎新增 large-v3-turbo、distil-large-v3.5 (英语)、kotoba-whisper v2.0 / v2.2 (日语, CTranslate2)、anime-whisper CT2 (fp16) 以及官方 kotoba-whisper-v2.2 (transformers); 通过 sherpa-onnx 新增 ONNX int8 模型 (SenseVoice、Paraformer-zh、Whisper small/turbo)
- 模型列表改为数据驱动 (`asr_registry.json`), 后台自动从 GitHub 仓库刷新 (本地缓存、离线可用、遵循下载代理); 新增「检查新模型」按钮, 在 Hugging Face 搜索新变体 — 点击「下载」前不会下载任何内容
- 新增「导入 HF 模型...」(粘贴链接或仓库 ID, 自动识别 CTranslate2 / transformers Whisper / SenseVoice FunASR) 和「导入 ONNX...」(本地 sherpa-onnx 文件夹或文件); 自定义条目保存在 user_settings.json
- 新增 ONNX 后端 (sherpa-onnx, CPU), 运行在 ASR 工作进程中; int8/fp32 文件版本跟随计算精度设置
- 新增「计算精度」设置 (auto/float16/bfloat16/int8*/float32): 加载时按 CTranslate2 实际支持校验, 计算能力 12.0+ 的 GPU (RTX 50 系) 上 int8 不可用并回退 float16; torch 模型 (anime-whisper、SenseVoice) 支持 fp16/bf16/fp32
- 原生支持 Claude (Anthropic) API: 模型新增「API 格式」(自动 / OpenAI 兼容 / Anthropic, 自动模式可识别 api.anthropic.com 与 `/anthropic` 网关), 通过 Messages API 支持流式与 JSON 输出, 每模型可设推理强度 (默认 low), 官方 Opus/Sonnet 支持服务端拒绝回退, 模型对话框新增「载入 Claude 预设」一键填入 (含价格); Anthropic 模型下不被 Claude 接受的采样参数 (temperature、top_p、惩罚项、seed) 会被禁用, 鉴权/限流/拒绝类错误只显示警告不再打印堆栈
- 模型默认下载源改为 Hugging Face (含中文界面), ModelScope 仍可手动选择; 已保存的选择不变

## 2026-08-17
- 增量 ASR 的分句库从 PySBD 切换到 yasbd-lib (#37): API 兼容无行为变化, 新增韩语等 17 种语言的原生分句规则 (此前韩语回退英语规则), 长文本分割速度大幅提升
- 修复 DeepSeek 默认开启思考模式导致翻译返回空、界面显示原文的问题 (#38): DeepSeek 需要 thinking.type=disabled 才能关闭思考, 此前发送的 enable_thinking=false 是 Qwen 风格参数对其无效; 模型编辑对话框新增「禁用思考」风格选择 (自动识别 / DeepSeek·火山方舟·GLM / Qwen·百炼·硅基流动 / vLLM·SGLang 自部署 / OpenAI·Grok reasoning_effort / 不发送), 自动识别对 OpenAI 等会拒绝未知参数的官方端点不再乱发参数; 思考耗尽 token 返回空时输出诊断日志
- 修复高 DPI 缩放 (150%+) 下设置窗口超出屏幕、确定/取消按钮点不到的问题 (#39): 模型编辑对话框与控制面板各标签页改为按需滚动, 窗口高度自动限制在屏幕可用范围内
- 安装与更新不再向系统盘写入下载缓存: 便携版的 uv 缓存和托管 Python 移入程序目录, install/update 脚本的 pip 缓存与临时文件移入项目目录, 装完自动清理

## 2026-07-11
- 修复 Fun-ASR-Nano 首次加载时 Qwen3-0.6B 权重下载被 180 秒超时打断的问题 (#32): 权重改为在模型下载阶段提前获取, ASR worker 启动不再等待大文件下载
- 修复 ModelScope 1.38+ 缓存位置变化导致模型漏检的问题 (#32, #33): 兼容各版本 SDK 的全部缓存目录布局, 升级 SDK 后不再重复下载已有模型
- Qwen3-0.6B 权重支持从 ModelScope 下载, ModelScope 源用户不再被迫走 HuggingFace (#33)

## 2026-06-20
- 新增「远程 Whisper」ASR 引擎: 把语音识别外包到带 GPU 的另一台机器 (附 `asr_server.py` 服务端), 无 GPU 的机器也能实时识别
- 新增「WebID 身份核验」翻译提示词预设: 针对视频核验场景调优用词
- ASR 改为子进程隔离运行: worker 崩溃/超时自动重启, 内存增长到阈值自动回收, 识别故障不再拖垮界面
- 字幕窗口支持鼠标穿透 (#28): 字幕设置里新增开关, 托盘菜单可「字幕鼠标穿透」快速切换, 开启后点击直达背后窗口 (此时中键拖动失效, 先关穿透再移动)

## 2026-05-10
- 新增「导出到文件」: 支持原文 / 译文 / 原文+译文 三种格式, 悬浮窗右键菜单与托盘菜单均可触发
- 新增「转录持久化」(默认开启): 每次会话自动在 `transcripts/` 下创建 3 份文件 (原文 / 译文 / 全部), 每段识别结果实时追加写入, 不再受悬浮窗 50 条上限丢失早期文本
- 设置面板「缓存」标签新增「转录持久化」分组: 开关 + 打开转录文件夹按钮
- 新增内存占用上限保护: RSS 超过 4096MB 时弹托盘通知提示重启 (ASR 后端存在 native 侧工作区/缓存, Python GC 与 `torch.cuda.empty_cache` 不一定能回收)
- 日志新增 `MEM[asr#/tick]` 行: 每次真实 ASR 调用后输出 RSS / GPU(alloc/reserved) / 音频时长 / overlay 消息数 / VAD buffer 长度, 便于排查内存问题

## 2026-04-20
- 移除 Qwen3-ASR 引擎 (ONNX + GGUF 方案兼容性较差, 相关模型文件和 llama.cpp 运行时依赖一并清理)
- 模型配置新增「高级参数」: `temperature` / `top_p` / `max_tokens` / `frequency_penalty` / `presence_penalty` / `seed`, 每项独立「覆盖」开关, 未勾选时使用服务端默认值
- 模型配置新增 `extra_body` (JSON): 供应商专有参数透传, 如 `thinking_budget`、`reasoning_effort` 等, 保存时自动校验 JSON 格式
- 修复 Anime-Whisper 模型未缓存时下载对话框静默无动作的问题
- 修复设置面板「更新日志」Tab 空白 (正则匹配 H3 但文件用 H2, 3 月加入功能起就失效)

## 2026-04-18
- 新增 ASR 引擎: Anime-Whisper (litagin/anime-whisper), 日语动画/Galgame 特化, 擅长识别喘息/叹息等非语言发声
- 修复 HF 缓存检测误判: 下载中断留下的空目录不再被认为"已缓存"

## 2026-03-31
- 管道线程拆分: 原 capture+VAD+ASR 单线程改为 capture 与 ASR 两条独立线程, 长句 ASR 不再阻塞 RMS/VAD 条实时更新
- ASR 调度新增有界队列 (16 段), 队列满时丢弃最旧的 interim 段, 避免堆积导致延迟

## 2026-03-26
- 默认翻译提示词优化: 新增 ASR 纠错规则 (识别错别字/同音字时尽量按上下文修正)、流畅性规则 (避免逐字直译)

## 2026-03-25
- 样式标签新增「重置窗口位置」按钮: 字幕窗口回到 (100,100), 主悬浮窗回到屏幕右下角
- 字幕窗口默认位置从屏幕底部居中改为 (100,100), 最小高度调整为 200px
- 窗口位置恢复时校验是否在可见屏幕范围内 (使用 availableGeometry 排除任务栏), 高度变化后 clamp 到屏幕边界

## 2026-03-24
- 字幕窗口: 超长文本自动换行显示(不再分段), 背景高度平滑动画, 文字渲染 Pixmap 缓存
- 主悬浮窗/字幕窗口: 位置和大小自动记忆, 重启后恢复
- 主悬浮窗: 精简模式切换动画
- 设置: 移除配置文件 key 白名单限制

## 2026-03-23
- 品牌更名 LiveTrans → LiveTranslate
- 模型配置新增: 流式传输、结构化输出、上下文数、禁用思考(默认开启)
- 翻译结果流式逐字显示
- 提示词优化: 禁止多候选翻译, 编辑即时生效
- 模型重复输出检测与提示
- ASR 引擎标注[准确]/[快速]
- 设置面板新增更新日志 Tab
