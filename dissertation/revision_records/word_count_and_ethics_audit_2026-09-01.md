# 字数与学术伦理审计（2026-09-01）

## 1. 字数

计数对象为最终 PDF 中的 Abstract 与 Chapter 1--7。计入标题、正文、图注、表格/图中可提取文字和文内引用；排除封面、声明、致谢、目录、图表目录、References 和 Appendices。独立页码被排除。该方法贴近学生提供的 WMG 口径，但 PDF 分词仍是内部估算，提交时应以课程要求认可的最终计数方式为准。

| 部分 | 估算字数 |
|---|---:|
| Abstract | 327 |
| Chapter 1 | 1,993 |
| Chapter 2 | 2,795 |
| Chapter 3 | 4,063 |
| Chapter 4 | 1,548 |
| Chapter 5 | 1,697 |
| Chapter 6 | 2,508 |
| Chapter 7 | 304 |
| **合计** | **15,235** |

15,235 位于 15,000 字目标的 13,500--16,500 宽限区间内，距离上限约 1,265 字。后续正文发生变化时需要重新统计。计数由 `dissertation/tools/count_pdf_words.py` 从最终编译 PDF 重复生成。

## 2. 引用与近似复制检查

- 24 个 TeX 文件、26 条 bibliography entries 和 26 个实际使用的 citation keys 通过本地一致性检查。
- 引用数量只证明引用键存在且能解析，不自动证明每个观点的引文选择都正确；数据来源和软件 provenance 已另行按实际来源补入。
- 未发现 `\textquote`、`\blockquote` 或以引号呈现的长篇逐字引用。Introduction 中的 `quote` 环境是本研究自己的 research question，不是外部引文。
- 16 个带 DOI 的条目已于 2026-09-01 通过 Crossref 只读查询核对，DOI 均返回与 bibliography 相符的标题和年份。
- 6 个 URL-only 条目使用正式会议或出版平台链接。PMLR、AUAI 和 MLSys 页面/论文可访问；两个 CVF 页面在自动访问中返回 403，但 URL 结构和条目信息属于正式 CVF Open Access 页面。该访问限制不能被写成“文献不存在”。
- PDF 中检测到 22 个不同的外部链接。Rendle et al. (2009) 已改为论文 PDF 直达链接。
- 自动检查不能替代 Turnitin，也不能证明与所有未提供文献之间不存在文本重合。提交前应由学生逐段确认论证确实反映自己的理解，避免只替换词语的近似改写。

## 3. 研究伦理与数据边界

- 研究仅使用二手数据和离线系统实验；未招募参与者，也没有把用户照片纳入研究训练数据。
- 论文没有把离线 AUC/FITB 表现写成用户满意度，也明确说明没有参与者可用性证据。
- 用户照片和衣橱数据按当前实现保存在本地；系统不从图像自动推断性别，而由用户选择 menswear/womenswear candidate range。
- Polyvore 用于兼容性学习；Zara 数据仅用于识别与 feature-domain diagnostics。两者没有被写成同一标签来源。
- Appendix A 已嵌入三项真实培训证据：两页 `Research Integrity: Core` 证书、`Information Security Smart` 完成邮件，以及 `WMG Student Ethics Training` 徽章邮件。每份材料均保留姓名或 Warwick 收件人、日期和关键完成信息。
- Appendix B 已嵌入 2026 年 5 月 12 日的真实 waiver 确认，能够核对姓名、学号、导师和“不需要伦理审批”的结论。该确认沿用项目早期简短标题，附录正文已如实说明标题差异，没有把它改写成当前长标题。
- 数据公开可取得不等于允许再分发。Git 当前未跟踪第三方源数据图像、item-level embeddings 或 catalogue artifacts；仓库跟踪了两个约 1.5 MB 的兼容性头 checkpoint。检查显示它们只含 model tensors、configuration 和 training metadata，不含源图像或 product identifiers。即便如此，在公开发布前仍应确认数据源条款是否允许发布训练所得权重；如无法确认，最保守做法是不公开这些 checkpoint。

## 4. 最重要的学术诚信阻断项：生成式 AI 声明

Warwick 当前校级指引说明，是否允许 AI 取决于 assessment brief/course handbook；如使用 AI，通常必须说明 **why, where and how**，并按课程要求保留交互记录。官方页面：

- https://warwick.ac.uk/students/academic-essentials/academic-integrity/artificial-intelligence/
- https://warwick.ac.uk/services/ris/research-integrity/ethical-approval/

本项目确实使用了 OpenAI Codex 协助软件开发、调试、分析整理、LaTeX 制作、图表、引用审计，并生成和修改了部分论文草稿。不能把这些内容描述成完全没有 AI 辅助，也不能以“避免被检测”为目标。**在提交前，学生必须向导师或 WMG module team 确认该具体 assessment 允许的 AI 使用范围，并如实完成声明。** 如果课程不允许 AI 生成论文文字，仅增加声明也不能自动消除风险；学生需要按课程指示重写或采取其他补救措施。

可供导师审阅、但尚未写入提交稿的事实性声明草案：

> Generative AI disclosure: OpenAI Codex was used to support software development and debugging, LaTeX production, diagram preparation, reference auditing, and the generation and revision of draft prose. The student reviewed the outputs against the implemented code, recorded experimental results and cited sources, and remains responsible for the interpretation, final wording and submitted work. The permitted scope and required form of this disclosure will be confirmed against the WMG assessment brief before submission.

该草案必须根据 WMG 的具体规定调整，不能在未核对课程政策的情况下直接视为合规。

## 5. 当前结论

论文的引用结构、结果边界和研究伦理叙述目前较为谨慎，没有发现长篇逐字引用或伪造 DOI。真实伦理培训证明、waiver 确认、正式 project requirement 及其与本项目的关系说明均已补齐。Acknowledgements 页按学生后续要求恢复，仅简短感谢学术导师和朋友。仍不能把稿件称为“可直接提交”，因为签名以及经 WMG 确认的生成式 AI 使用声明/补救要求尚未最终完成。
