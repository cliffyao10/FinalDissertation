# Supervisor meeting progress update / 导师会议工作汇报稿

> This is a speaking and discussion draft, not dissertation prose. The English version is suitable for explaining the project orally; the Chinese version is for checking meaning. Values are taken from the current experiment artefacts and should not be replaced by rounded claims without checking the JSON results.
>
> 这是一份用于汇报和讨论的稿子，不是直接放入论文的正式正文。英文部分适合向导师口头介绍，中文部分用于核对含义。数值来自当前实验产物，未经核对 JSON 结果前不应进一步四舍五入或夸大。

## 1. Project summary / 项目概述

**English.** The project is a local, low-stakes complete-outfit recommendation prototype called Cove. Its research focus is a lightweight compatibility model combined with product-independent abstract garment search, explicit weather and clothing-range constraints, and a local wardrobe interface. The system is intended to help a user complete an outfit after selecting one garment; it is not presented as an authority on taste, identity, safety or purchasing.

**中文。** 本项目是一个名为 Cove 的本地、低风险完整穿搭推荐原型。研究重点是轻量兼容性模型、与具体零售产品无关的抽象服装搜索、明确的天气和服装范围约束，以及本地衣橱界面。系统的用途是在用户选定一件衣服后帮助补全穿搭；它不被描述为审美、身份、安全或购买决策的权威。

## 2. What has been used / 使用了什么

**English.** The main compatibility data are the published Polyvore Outfits disjoint splits. The deployed D2 model uses 18,243 training examples, 3,274 validation examples and 16,062 held-out test examples. Positive examples are observed outfits, and negative examples are same-type corrupted outfits supplied by the processed benchmark protocol. User photographs are kept locally and are not added to the research training data.

**中文。** 兼容性模型主要使用公开的 Polyvore Outfits disjoint 划分。当前部署的 D2 模型使用 18,243 条训练样本、3,274 条验证样本和 16,062 条独立测试样本。正样本是数据中观察到的完整穿搭，负样本是按照处理后的基准协议构造的同类型替换穿搭。用户照片只保留在本地，不加入研究训练数据。

**English.** A frozen `google/siglip-base-patch16-224` image encoder converts each garment image into a 768-dimensional vector. The encoder is reused rather than fine-tuned. The trainable part is a 379,265-parameter compatibility head that projects item representations, adds slot information, forms a masked outfit summary and pairwise slot differences, and predicts a compatibility score. This is a small deep-learning component, while the complete system also contains non-neural recognition, filtering, search and fallback code.

**中文。** 冻结的 `google/siglip-base-patch16-224` 图像编码器把每件服装图像转换为 768 维向量。项目复用了这个编码器，没有对它进行微调。真正训练的是一个包含 379,265 个参数的兼容性模型头：它投影单品表示、加入位置/槽位信息、形成带掩码的穿搭摘要和成对位置差异，最后预测兼容性分数。因此，项目包含一个小型深度学习组件，但完整系统还包括非神经的识别、过滤、搜索和回退代码。

## 3. What has been built / 已完成的功能和代码

**English.** The implemented pipeline accepts a selected garment from a photograph or the local wardrobe, analyses its broad type, colour and style, obtains its frozen image representation, constructs feasible abstract candidates and ranks complete four-position outfits. The supported positions are inner top, optional outer layer, bottom and shoes. The selected item remains fixed while the other positions are completed where feasible.

**中文。** 当前实现可以从照片或本地衣橱中接收用户选定的服装，分析其大类、颜色和风格，获取冻结图像表示，构造可行的抽象候选，并为四个服装位置组成的完整穿搭排序。支持的位置是内搭、可选外套、下装和鞋。用户选定的服装会保持固定，系统只在其他位置可行时进行补全。

**English.** Candidate states are derived from the Polyvore training split and are deliberately product-independent. They contain a slot, broad garment type, colour, primary style and clothing range, represented by an observed embedding medoid. The user interface displays these learned suggestions as filled garment icons. It does not display a source product title, brand, URL or price as if the system were recommending a specific retail item. A user's own wardrobe pieces retain their real photographs.

**中文。** 候选状态只从 Polyvore 训练部分派生，并且刻意不绑定具体产品。每个状态包含位置、大类、颜色、主要风格和服装范围，并由一个实际观测到的中心向量代表。用户界面把这些学习得到的建议显示为填色服装图标，而不是把来源商品的标题、品牌、网址或价格展示成具体零售推荐。用户自己的衣服则保留真实照片。

**English.** Weather, occupied slots, clothing range and unsupported combinations are handled as explicit constraints outside the compatibility score. The exact search path enumerates feasible abstract combinations in bounded batches and keeps the global top-k results. If model artefacts are unavailable, a deterministic rule-based fallback keeps the application usable and records that the result did not come from the learned model.

**中文。** 天气、已经占用的位置、服装范围和不支持的组合由兼容性分数之外的明确规则处理。精确搜索路径分批枚举可行的抽象组合，并保留全局 top-k 结果。如果模型产物不可用，系统会使用确定性的规则回退以维持可用性，并记录该结果不是由学习模型产生的。

## 4. Model and evaluation work / 模型与评价工作

**English.** The selected checkpoint achieved a test ROC-AUC of 0.8583, with a 95% bootstrap interval of 0.8524–0.8641, on 16,062 held-out compatibility examples. FITB accuracy on the retained four-slot subset was 0.6025, with a 95% interval of 0.5873–0.6175. The retained FITB coverage was 4,468 of 15,145 official questions, or 29.50%, because the prototype supports four clothing positions. The FITB number is therefore not presented as performance on the complete official benchmark.

**中文。** 在 16,062 条独立兼容性测试样本上，最终检查点的测试 ROC-AUC 为 0.8583，bootstrap 95% 区间为 0.8524–0.8641。在保留的四位置 FITB 子集上，准确率为 0.6025，95% 区间为 0.5873–0.6175。由于原型只支持四个服装位置，FITB 只保留了官方 15,145 个问题中的 4,468 个，即 29.50%。因此，论文不会把该 FITB 数值描述为完整官方基准上的成绩。

**English.** The controlled three-seed ablation used seeds 42, 7 and 123 under the same questions and protocol. The full model had mean FITB accuracy 0.5994. Removing the pairwise relationship module reduced mean accuracy by 0.1028, with a paired 95% interval of 0.0896–0.1157. Removing slot embeddings reduced the point estimate by only 0.0047, and its paired interval crossed zero. The evidence therefore supports retaining the pairwise module, but does not justify claiming a clear independent benefit from slot embeddings.

**中文。** 受控三随机种子消融实验使用 42、7 和 123，并保持相同问题和实验协议。完整模型的 FITB 平均准确率为 0.5994。移除成对关系模块后，平均准确率下降 0.1028，配对 95% 区间为 0.0896–0.1157。移除槽位嵌入只带来 0.0047 的点估计下降，而且配对区间跨过零。因此，证据支持保留成对关系模块，但不能据此声称槽位嵌入具有明确的独立增益。

**English.** The overfitting audit found a training AUC of approximately 0.9699, while validation and test AUC differed by only 0.00145 after validation-based early stopping. This supports the selected checkpoint procedure but does not support an absence-of-overfitting claim. The project also reports calibration, outfit-length and changed-slot imbalance, domain shift, candidate-search completeness, runtime and operational fallback behaviour.

**中文。** 过拟合审计显示训练 AUC 约为 0.9699，而采用基于验证集的早停后，验证与测试 AUC 只相差 0.00145。这支持当前的检查点选择流程，但不能据此声称不存在过拟合。项目还报告了校准、穿搭长度与被替换位置的不平衡、领域偏移、候选搜索完整性、运行时间和运行时回退行为。

## 5. Main findings and honest boundaries / 主要发现与边界

**English.** The evidence suggests that a small relational compatibility head can extract useful compatibility signals from frozen visual representations under the project's controlled protocol. The pairwise component is the strongest supported architectural contribution. Exact abstract search also recovers solutions that the earlier quota-limited path often missed. These are offline and system-level findings, not evidence of universal taste, user satisfaction, purchase intention or commercial performance.

**中文。** 现有证据表明，在本项目的受控协议下，小型关系兼容性模型头能够从冻结视觉表示中提取有用的搭配信号。成对关系组件是目前证据支持最强的架构贡献。抽象空间中的精确搜索也能找回旧配额路径经常遗漏的解。这些是离线实验和系统层面的发现，不是普遍审美、用户满意度、购买意愿或商业表现的证据。

**English.** The principal limitations are the 29.50% FITB coverage, sparse pure-menswear evidence, unresolved Polyvore-to-catalogue domain shift, possible false negatives created by item replacement and the absence of a participant study. The interface therefore lets users choose the clothing range explicitly and does not infer gender from an image. Dress-specific multi-slot modelling remains outside the current scope. The compatibility head is shared across ranges because the available pure menswear training evidence is too small for a defensible separate menswear model.

**中文。** 主要局限包括 FITB 只有 29.50% 的覆盖率、纯男装证据稀少、Polyvore 到其他目录的领域偏移尚未解决、服装替换可能产生错误负样本，以及没有参与者研究。因此，界面让用户主动选择服装范围，不从图片推断性别。连衣裙所需的多位置建模仍不在当前范围内。由于现有纯男装训练证据太少，无法合理支持一个独立男装模型，所以当前使用共享兼容性模型头。

## 6. What I would like feedback on / 希望导师重点反馈的问题

**English.** I would like feedback on whether the research question is appropriately scoped as an applied-AI recommendation-system study, whether the distinction between learned compatibility scoring and explicit system constraints is clear enough, and whether the evidence is sufficient for the claims made. I would also like advice on how prominently to discuss the limited FITB coverage, sparse menswear evidence and domain shift in the final discussion.

**中文。** 我希望请导师重点反馈三个方面：目前研究问题是否适合作为 Applied AI 的推荐系统研究；学习式兼容性评分与系统明确约束之间的区别是否表达清楚；以及现有证据是否足以支持当前结论。另外，也希望得到关于如何在最终讨论中突出 FITB 覆盖率有限、男装证据稀少和领域偏移问题的建议。

## 7. Immediate next steps / 下一步计划

**English.** The immediate work is to finish auditing the dissertation wording and references, remove unnecessary product-specific emphasis from the abstract-candidate explanation, and keep the product-independent formulation consistent throughout. The implementation and test artefacts are complete enough for report rewriting: the current release check passes, 98 automated tests pass, and the LaTeX manuscript has been compiled and checked. Further model changes should only be made if the supervisor identifies a research question or evidence gap that justifies them.

**中文。** 近期工作是继续审计论文措辞和引用，删除抽象候选说明中不必要的具体产品强调，并在全文统一使用“与具体产品无关的抽象服装状态”这一表述。当前实现和测试产物已经足以开始重写论文：发布检查通过，98 个自动化测试通过，LaTeX 稿件也已经编译和检查完成。除非导师指出需要新研究问题或新证据，否则不应为了追求更高分数而随意继续改模型。

## Wording note recorded for revision / 已记录的措辞修改

**English.** Replace wording such as “abstract garment states rather than as Zara” with “abstract garment states that are independent of specific retail products”. Zara should only be mentioned where its separately documented dataset role is necessary, namely recognition or feature-domain diagnostics; it should not be presented as the conceptual alternative to the abstract recommendation space.

**中文。** 将类似 “abstract garment states rather than as Zara” 的表述改为“与具体零售产品无关的抽象服装状态”。只有在说明 Zara 二手数据被用于服装识别或特征领域诊断时才提及它，不要把 Zara 写成抽象推荐空间的概念对立面。
