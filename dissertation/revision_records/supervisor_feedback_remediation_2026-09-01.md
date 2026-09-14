# 导师反馈整改记录（2026-09-01）

## 记录说明

- 整改对象：`dissertation/main.pdf` 及其 LaTeX 源文件。
- 行号均指本次整改完成后的源文件行号；若后续继续编辑，优先使用本记录中的“定位锚点”检索。
- 本记录区分“已完成”“已验证”和“等待学生材料”，避免把未完成项目写成已经完成。
- 本轮未重新训练模型，也未增加新的实验结果；所有新增实验说明均追溯到现有代码和结果文件。

## R-01：“lightweight”范围过宽

**导师意见**：应明确轻量化仅指项目训练的 compatibility head 和下游训练成本，不包括大型 SigLIP 编码器的原始预训练成本。

**整改前风险**：摘要、研究问题和结论中使用了 `lightweight model/system`，读者可能将其理解为整个技术栈都属于小模型。

**已完成整改**：

1. 在摘要首次定义轻量化边界。
   - 文件：`frontmatter/abstract.tex`
   - 位置：第 6--13 行
   - 锚点：`lightweight refers to the trainable head`
2. 在引言中加入全文统一定义。
   - 文件：`chapters/01_introduction.tex`
   - 位置：第 45--48 行
   - 锚点：`Throughout this dissertation`
3. 在方法章明确 379,000 个可训练参数只属于 compatibility head。
   - 文件：`chapters/03_methodology.tex`
   - 位置：第 61--69 行
   - 锚点：`The description \emph{lightweight}`
4. 在分析章将训练时间结论限定为 `lightweight downstream training`。
   - 文件：`chapters/05_analysis.tex`
   - 位置：第 47--51 行
   - 锚点：`narrower claim of lightweight downstream training`
5. 在讨论章保留 SigLIP 预训练成本和继承偏差的说明。
   - 文件：`chapters/06_discussion.tex`
   - 位置：第 37--45 行附近
   - 锚点：`original training cost and learned biases`
6. 在结论首段再次限定该术语，避免脱离正文阅读时产生误解。
   - 文件：`chapters/07_conclusion.tex`
   - 位置：第 3--9 行
   - 锚点：`Here, \emph{lightweight}`

**追加整改**：论文标题不再把整个系统笼统称为 `lightweight`，而明确写为 `a Lightweight Task-Specific Compatibility Head`；研究问题同步改为 `a small project-specific outfit-compatibility head`。

- 文件：`main.tex`；锚点：`\projecttitle`
- 文件：`chapters/01_introduction.tex`；锚点：`To what extent can a small project-specific`

**状态**：已完成；没有改变模型结果，只收紧了成果表述范围。

## R-02：泛化表述过强

**导师意见**：验证集和测试集接近只能支持所选 checkpoint 在当前 held-out split 上的一致性，不能证明一般可靠性。

**整改前表述**：`The model generalised consistently from validation to test.`

**已完成整改**：统一改为 `Held-out test performance closely matched validation performance`，并紧接说明该结果不代表跨数据域的一般可靠性。

- 分析章：`chapters/05_analysis.tex` 第 80--99 行；锚点 `Held-out test performance closely matched`。
- 讨论章：`chapters/06_discussion.tex` 第 104--111 行；锚点 `not general reliability`。
- 结论章：`chapters/07_conclusion.tex` 第 19--25 行；锚点 `not general reliability`。
- 章节标题同步由 `Generalisation, overfitting and calibration` 改为 `Held-out consistency, overfitting and calibration`。

**保留的限制证据**：train--test gap、Polyvore--Zara 域区分、FITB 覆盖率和 menswear 样本不足均继续保留。

**状态**：已完成；未声称模型在新零售图片或真实用户照片上可靠泛化。

## R-03：界面“user-friendly”缺少参与者证据

**导师意见**：没有用户可用性研究，不能把界面写成已经证明 user-friendly。

**已完成整改**：

- 将目标中的 `user-friendly interaction design` 改为 `an interaction design intended to support clear and accessible use`。
  - 文件：`chapters/01_introduction.tex`
  - 位置：第 170 行附近
- 方法章改为陈述可观察的设计属性：操作层级、主题颜色、服装范围胶囊和操作路径；随后明确这些不是 measured usability results。
  - 文件：`chapters/03_methodology.tex`
  - 位置：第 246--275 行
  - 锚点：`These features were designed to support clear, accessible interaction`
- 讨论章 Objective 3 改为 `designed to support clear and accessible interaction`，并保留没有 participant evaluation 的限定。
  - 文件：`chapters/06_discussion.tex`
  - 位置：第 188--200 行
- 结论只陈述 `by design`，不声称已证明 perceived usability、情绪改善或用户偏好。
  - 文件：`chapters/07_conclusion.tex`
  - 位置：第 34--43 行

**状态**：已完成；UI 功能与设计意图仍被呈现，但未被包装为用户实验结果。

## R-04：18 个 exact-search 查询的选择方式不透明

**导师意见**：需要说明 18 个查询如何选择，以增强 27.78% top-one recovery 结论的可信度。

**代码与结果追溯**：

- 正样本输入池构造：`recommendation_training/evaluate_v3_system.py` 第 180--187 行。
- 默认查询数 18：同文件第 238 行。
- 风格、位置和输入索引的确定性轮换：同文件第 255--265 行。
- quota 路径每个位置保留最多 10 个候选：同文件第 284--290 行。
- 结果元数据：`results/v3_system_comparison.json` 第 419--437 行。
- 已核对结果中的 18 行查询：九种风格各出现两次；输入位置分布为 inner top 5、outer layer 5、bottom 4、shoes 4。

**已完成整改**：方法章新增完整选择说明。

- 文件：`chapters/03_methodology.tex`
- 位置：第 186--198 行
- 锚点：`The 18 inputs were a deterministic coverage sample`
- 新增内容包括：
  - 来源为 `data/processed_disjoint/test.pt` 的正样本；
  - 按服装位置建立输入池，并按存储顺序取值；
  - 九种风格各覆盖两次；
  - 四个固定服装位置至少覆盖四次；
  - 颜色标签按固定顺序循环；
  - 记录实验使用 womenswear 候选范围；
  - held-out outfit 的真实服装没有被插入候选答案；
  - 18 个案例没有按结果进行事后筛选；
  - 该实验是 bounded engineering comparison，不是总体统计估计。

讨论章 RQ3 同步加入“小型确定性覆盖样本”的限制，防止将 27.78% 外推为所有用户查询的失败率。

- 文件：`chapters/06_discussion.tex`
- 位置：第 132--145 行附近
- 锚点：`The cases were generated deterministically`

**状态**：已完成；说明完全依据现有代码和 JSON 输出，没有补造随机抽样过程。

## R-05：Analysis、Discussion 和 Conclusion 重复

**导师意见**：最终章节应集中回答研究问题、贡献和主要限制，不需要再次复述全部数值。

**已完成整改**：

1. Discussion 的文献关系部分删除与 RQ1 重复的完整 AUC 数字清单，只保留与既有研究的概念联系。
   - 文件：`chapters/06_discussion.tex`
   - 位置：第 15--45 行
2. Conclusion 从原来的约 85 行压缩为 43 行，并排成一个正文页面。
   - 文件：`chapters/07_conclusion.tex`
   - 保留：研究问题答案、pairwise 消融的代表性证据、held-out 一致性边界、系统贡献和主要限制。
   - 删除：baseline 数值逐项复述、所有训练决策的再次罗列，以及与 Discussion 重复的长篇 future-work 清单。
3. 章节职责保持为：Chapter 4 报告结果；Chapter 5 分析结果；Chapter 6 回答 RQ/Objective 并讨论有效性；Chapter 7 给出集中结论。

**状态**：已完成；结论打印页为第 51 页，未再产生只有少量文字的第二页。

## R-06：目录编号与标题粘连

**发现的问题**：原目录中的 `2.10Synthesis`、`2.11Summary`、`6.10Prioritised` 和 `6.11Summary` 缺少可见间距。

**已完成整改**：扩大目录 section/subsection 的编号盒宽度。

- 文件：`main.tex`
- 位置：第 66--74 行
- 锚点：`The report-class section-number box is too narrow`
- 修改：重新定义 `\l@section` 和 `\l@subsection` 的编号宽度。

**视觉验证**：目录罗马页码 vi--vii 已渲染检查，2.10、2.11、6.10 和 6.11 均与标题正确分隔。

**状态**：已完成并视觉验证。

## R-07：页码、图表、引用和编译检查

**已执行检查**：

- LaTeX 全量重新编译，目录和交叉引用自动刷新。
- 检查 `Undefined control sequence`、undefined references 和 overfull box。
- 检查参考文献条目数与正文 citation keys。
- 将 Rendle et al. (2009) 的链接由会议论文列表页替换为该论文的官方 PDF 直达链接，便于读者直接核查原文。
  - 文件：`references.bib`
  - 位置：第 116 行附近
  - 锚点：`rendle2009bpr`
- 渲染并查看摘要、目录、18-query 方法说明、held-out 分析、Discussion RQ2、Conclusion 和 References 页面。
- 检查 A4 页面、空白页、外部链接和正文估算字数。

**状态**：本记录创建后执行最终自动检查；最终数值写入下方“最终验证结果”。

## R-08：仍等待学生提供的提交材料

以下内容无法根据模型代码或公开信息安全补写，因此保持未完成状态：

| 材料 | 源文件位置 | 当前状态 |
|---|---|---|
| Student name | `main.tex` 第 89 行 | 等待学生提供 |
| Student ID | `main.tex` 第 90 行 | 等待学生提供 |
| Submission month/year | `main.tex` 第 91 行 | 等待学生提供 |
| Qualifications obtained | `frontmatter/titlepage.tex` 第 17 行 | 等待学生提供 |
| Signature and date | `frontmatter/proforma.tex` 第 32 行 | 等待学生签署 |
| Degree project definition 原文 | `frontmatter/declaration.tex` 第 13--14 行 | 等待 Course Regulations 原文 |
| 项目与 degree definition 的关系 | `frontmatter/declaration.tex` 第 16--18 行 | 获得上项原文后撰写 |
| Ethics training evidence | `appendices/appendices.tex` 第 1--3 行 | 等待证明文件 |
| Ethical approval waiver email | `appendices/appendices.tex` 第 5--7 行 | 等待邮件或导出件 |
| Generative-AI 使用声明及课程许可确认 | `frontmatter/declaration.tex`，必要时连同提交系统声明 | 必须按 WMG assessment brief/handbook 如实说明使用了什么工具、为何使用、用于哪些工作；当前尚未写入最终论文 |
| Acknowledgements | `frontmatter/acknowledgements.tex` 第 4 行 | 学生决定填写或删除 |

这些占位符必须在提交前移除。为了避免虚构个人信息或学校规定，本轮没有代填。

## 最终验证结果

- **编译**：`dissertation/build.ps1` 全量编译成功；最终 PDF 为 72 页、A4、约 1.41 MB。
- **关键 LaTeX 问题**：未发现 overfull box、undefined control sequence、未定义引用、未定义交叉引用或重复定义。日志仍含少量 underfull box 排版提示，集中在窄列表格单元格；这些不造成文字遮挡或越界。
- **文稿完整性检查**：共检查 24 个 TeX 文件、22 条参考文献和 22 个正文 citation keys；自动检查通过，未发现未来章节占位符。
- **引用链接**：PDF 中检测到 22 个不同的可点击外部链接，与 22 条参考文献相对应。
- **字数估算**：在补充数据再分发伦理边界后，按 Abstract 加 Chapter 1 至 References 前正文的 PDF 可提取英文 token 估算为 15,846；处于 15,000 字目标的 ±10% 区间。该值用于内部控制，最终提交时仍应按学校要求报告字数。
- **页面完整性**：未检测到空白页；Introduction 从 PDF 第 12 个物理页面开始，References 从第 63 个物理页面开始。
- **视觉检查**：已检查标题页、目录第 vi--vii 页和 References 首页；标题完整、目录长编号不再粘连、参考文献可读。
- **差异复核**：在方法章界面说明中发现并修正 `In this dissertation, These features...` 的大小写/衔接错误；锚点为 `These features were designed to support clear`。
- **Overleaf 包**：已重新生成 `dissertation/Cove-dissertation-overleaf.zip`，内容与本轮最终 LaTeX 源文件一致。
- **仍未完成**：R-08 所列个人信息、Course Regulations 原文、签名以及两项伦理附件仍必须由学生提供。本轮没有用推测内容替代。
