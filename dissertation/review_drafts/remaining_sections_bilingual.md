# Remaining dissertation sections — bilingual review draft

> Status: for student review. The English text may be transferred to the submission LaTeX only after approval. The Chinese text is an aligned explanation and is not intended for submission.

## Proposed addition to Chapter 3: Research risks and mitigation

### English

The investigation involved methodological, data, ethical and operational risks. These risks were considered during system development rather than being treated only as limitations after the experiments. The first risk concerned data provenance and permitted use. The work therefore used downloaded secondary datasets, recorded their stated sources and purposes, and did not scrape live retailer pages. Polyvore data were used for compatibility learning, while the separately obtained Zara data were restricted to recognition and feature-domain diagnostics. User photographs remained local and were not incorporated into the research dataset. This boundary reduces privacy exposure, but it does not establish that every image has the same provenance or representation quality. Any public release must consequently exclude third-party images and large derived artefacts unless their redistribution terms have been checked separately.

The second risk was data leakage. Product overlap between training and test data could allow a model to recognise previously seen garments instead of learning relationships that transfer to unseen items. The published disjoint split was therefore used, and candidate concepts were derived from the training division rather than from the test division. Model selection and temperature scaling used validation data; the test set was reserved for final evaluation. These controls reduce leakage, although repeated interpretation of one test set can still influence later development decisions. The test results are therefore treated as final evidence for the present model rather than as a target for further tuning.

The third risk concerned misleading labels and incomplete coverage. Positive labels came from observed Polyvore outfits, whereas negatives were constructed by replacing an item within a broad category. A constructed negative is not proof that an outfit is inherently unacceptable; it is an evaluation device for learning distinctions in this dataset. The official FITB evaluation was also filtered to questions representable by the four supported clothing positions and available embeddings. Coverage is reported alongside accuracy so that the retained result is not presented as performance on the complete benchmark. These choices follow the wider recommendation-system principle that a metric is meaningful only in relation to its task and sampling protocol (Herlocker et al., 2004).

The fourth risk was overfitting and unstable model selection. Training, validation and test behaviour were recorded separately, three random seeds were used for the controlled architecture comparison, and the best checkpoint was selected from validation AUC. Confidence intervals were estimated for the principal test and paired comparisons. Early stopping limits continued fitting after validation performance ceases to improve, but it cannot remove the generalisation gap (Prechelt, 1998). For that reason, both the close validation–test result and the larger train–test difference are reported.

The fifth risk involved unequal representation. The retained FITB questions and training examples were heavily concentrated in womenswear. Reporting a small menswear subgroup as if it supported a reliable fairness conclusion would be misleading. The interface consequently asks the user to select a clothing range and does not infer gender from an image. Menswear evidence is reported as sparse coverage, not as proof of equivalent performance. This response does not solve representational bias; it makes the boundary visible and avoids automated gender classification, which can reproduce harmful assumptions about identity (Keyes, 2018; Mehrabi et al., 2021).

Finally, the deployed recommendation process could fail because model files, embeddings or candidate records are unavailable, or because an allowed candidate combination cannot be produced. Release checks, deterministic seeds and recorded artefact paths support reproducibility. Explicit slot, weather and clothing-range constraints reject infeasible combinations, while a deterministic rule-based fallback allows the application to continue when learned artefacts cannot be loaded. The fallback improves operational resilience but is not evidence that the resulting recommendation is stylistically correct. Across these mitigations, the purpose was to make failure conditions observable and bounded rather than to claim that technical testing removed all social or external-validity risks.

### 中文对照

本研究包含方法、数据、伦理和运行方面的风险。这些风险在系统开发过程中就被考虑，而不是等实验结束后才统一列为局限。第一项风险涉及数据来源与允许用途。因此，项目只使用已下载的二手数据集，记录其声明的来源和用途，并且不爬取实时零售网站。Polyvore 数据用于学习穿搭兼容性；另行获得的 Zara 数据仅用于服装识别和特征域诊断。用户照片保留在本地，也不进入研究数据集。这样的边界能够降低隐私风险，但不能证明每张图像都拥有相同的来源可靠性或代表性。因此，若将来公开项目，不应直接发布第三方图片或大型衍生产物，除非已单独核实其再分发条款。

第二项风险是数据泄漏。训练集与测试集若包含相同商品，模型可能只是认出了以前见过的衣服，而不是学会能够迁移到未见商品的搭配关系。因此，本研究采用公开的 disjoint 划分，并且只从训练部分生成候选概念，不使用测试部分生成候选。模型选择和温度缩放仅使用验证集，测试集保留用于最终评价。这些控制降低了泄漏风险，但如果不断根据同一测试集解释和修改系统，测试信息仍可能间接影响开发。因此，本研究把当前测试结果视为现有模型的最终证据，而不是继续调参的目标。

第三项风险来自标签含义和覆盖不完整。正标签来自 Polyvore 中观察到的真实穿搭，负标签则通过在同一大类中替换一件衣服构造。构造出的负样本不能证明这套衣服在现实中一定不能穿；它只是用于学习该数据集内部区分的一种评价方式。官方 FITB 评价也只保留了能够映射到四个服装位置且拥有可用向量的问题。因此，论文把覆盖率与准确率一起报告，避免把过滤后的结果描述成完整官方基准成绩。这符合推荐系统评价中的基本原则：任何指标都必须结合任务与采样协议解释（Herlocker et al., 2004）。

第四项风险是过拟合和模型选择不稳定。本研究分别记录训练、验证和测试行为，在受控架构比较中使用三个随机种子，并根据验证集 AUC 保存最佳检查点。主要测试结果和配对比较同时给出置信区间。早停能够限制模型在验证表现不再改善后继续拟合，但不能消除泛化差距（Prechelt, 1998）。因此，论文同时报告验证集与测试集接近的结果，以及更大的训练集与测试集差距。

第五项风险涉及代表性不均衡。保留的 FITB 问题和训练样本明显集中于女装。如果把很小的男装子组描述成可靠的公平性结论，会产生误导。因此，界面由用户主动选择服装范围，而不是从人物或服装图片中推断性别。男装结果被报告为样本稀少的覆盖局限，而不是两个范围表现相同的证据。这种做法并没有解决代表性偏差，但它使问题可见，也避免了可能强化身份刻板假设的自动性别识别（Keyes, 2018; Mehrabi et al., 2021）。

最后，部署后的推荐流程可能因为模型文件、向量或候选记录不可用而失败，也可能在约束下无法生成有效组合。发布检查、固定随机种子和已记录的产物路径用于支持可复现性。服装位置、天气和服装范围约束负责排除不可行组合；当学习模型无法载入时，确定性的规则回退可以维持应用运行。回退机制提高了运行韧性，但不能证明回退结果在风格上一定正确。总体而言，这些措施的目的不是声称技术测试消除了所有社会风险和外部有效性问题，而是让失败条件能够被观察并限制在明确范围内。

---

## Chapter 6: Discussion

### 6.1 Introduction — English

This chapter brings together the literature, experimental findings and system evidence to answer the research questions. The discussion separates three levels of claim. The first concerns discrimination and completion ranking on the processed Polyvore data. The second concerns the contribution of the model components under controlled ablation. The third concerns the behaviour of the complete prototype, including candidate search, constraints and fallback. Keeping these levels separate is important because offline compatibility accuracy does not by itself demonstrate that users will prefer, trust or purchase the resulting outfits.

### 6.1 中文对照

本章把文献、实验结果和系统证据结合起来回答研究问题。讨论区分三个层次的结论：第一层是模型在处理后的 Polyvore 数据上的区分和补全排序能力；第二层是受控消融实验中各模型组件的作用；第三层是完整原型的运行表现，包括候选搜索、约束和回退机制。区分这些层次十分重要，因为离线兼容性准确率本身不能证明用户会喜欢、信任或购买生成的穿搭。

### 6.2 Relation to previous outfit-compatibility research — English

Previous work has represented an outfit as a sequence, as type-specific pair relationships, or as a set processed by attention (Han et al., 2017; Vasileva et al., 2018; Sarkar et al., 2023). The present results support the shared premise behind these approaches: compatibility is not adequately represented by isolated visual similarity. Mean pairwise SigLIP cosine similarity reached AUC 0.7012 in the fixed-seed comparison, whereas the learned relational head reached 0.8573 in that experiment. The mean-pooled logistic baseline was weaker still, with test AUC 0.5532 on the disjoint data. Frozen visual features therefore contained useful information, but a simple average of those features did not express the relationships required by the task.

The proposed model produced mean AUC 0.8596 across three seeds, above the controlled bidirectional LSTM, type-aware pairwise and set-Transformer implementations. This is relevant evidence for the selected lightweight design, but it is not a claim that it outperforms the original published systems. The comparison models were adapted to the same four-slot tensors, frozen embeddings, processed labels and training conditions. Their purpose was to compare architecture families while controlling the surrounding pipeline. Published headline scores use different inputs, splits, negative construction and implementation details, so directly ranking the project against those values would not be valid. The finding is narrower: under this project’s protocol, the comparatively small relational head was more effective and more stable than the three controlled alternatives.

The frozen SigLIP encoder also changes how the result should be interpreted. SigLIP was trained previously on large-scale image–text data and supplied a transferable representation (Zhai et al., 2023). The project did not reproduce or fine-tune that representation learning. Its original training cost and learned biases therefore remain part of the dependency, even though the project-specific head contains only about 379,000 trainable parameters and trains quickly. The short head-training time is consistent with reusing precomputed embeddings; it should not be presented as the total computational cost of creating the visual representation from first principles.

### 6.2 中文对照

既有研究分别把穿搭表示为序列、带类别的成对关系，或由注意力处理的集合（Han et al., 2017; Vasileva et al., 2018; Sarkar et al., 2023）。本研究的结果支持这些方法共有的前提：单独的视觉相似度不足以表示穿搭兼容性。在固定随机种子的比较中，SigLIP 平均成对余弦相似度的 AUC 为 0.7012，而学习得到的关系模型为 0.8573。更简单的平均池化逻辑回归在 disjoint 数据上的测试 AUC 只有 0.5532。因此，冻结视觉特征确实包含有用信息，但简单地平均这些特征不能表达任务所需要的服装关系。

本研究模型在三个随机种子上的平均 AUC 为 0.8596，高于受控实现的双向 LSTM、类别感知成对模型和集合 Transformer。该结果为本项目选择轻量模型提供了相关证据，但不能被描述为超过原论文中的完整系统。用于比较的模型都被调整为使用相同的四位置张量、冻结向量、处理后标签和训练条件，其作用是在控制其他流程的情况下比较架构方向。公开论文中的主要成绩依赖不同输入、数据划分、负样本构造和实现细节，不能直接与本项目数值排序。能够支持的结论更窄：在本项目协议下，这个较小的关系模型比三个受控替代模型更有效且更稳定。

冻结 SigLIP 编码器也改变了对结果的解释。SigLIP 已经在大规模图文数据上完成预训练，并提供可迁移的表示（Zhai et al., 2023）。本项目没有复现或微调这一表示学习过程。因此，即使项目专用模型只有约 37.9 万个可训练参数且训练很快，SigLIP 原始训练成本和潜在偏差仍然属于系统依赖。较短的模型头训练时间来自复用预先计算的向量，不能被描述为从零创造视觉表示所需的全部计算成本。

### 6.3 Answer to RQ1: comparative performance — English

RQ1 asked how the proposed model compared with non-learned and controlled learned baselines under consistent protocols. The evidence gives a positive but bounded answer. The selected checkpoint achieved test AUC 0.8583 with a 95% interval of 0.8524–0.8641. It was clearly above deterministic random scores and mean pairwise cosine similarity. Across three seeds, its mean AUC exceeded each controlled learned architecture; paired bootstrap intervals for the AUC differences did not include zero. Its standard deviation of 0.0009 was also lower than those of the comparison models.

These results indicate that the model learned a repeatable distinction between observed and synthetically corrupted outfits within the processed disjoint setting. AUC measures the probability that a randomly selected positive example is ranked above a randomly selected negative example (Fawcett, 2006). It does not mean that 85.83% of proposed outfits would be judged fashionable, nor does it measure individual taste. The use of same-category replacement negatives makes the task more meaningful than arbitrary cross-category corruption, but it still defines compatibility through the available dataset and negative generator.

The FITB accuracy of 0.6025 provides a second form of evidence because it requires selection of one answer among four alternatives. This is above the chance level of 0.25 and its confidence interval does not approach chance. However, only 29.50% of the official questions were retained. The result therefore applies to the four supported positions with available embeddings, not to the complete Polyvore disjoint benchmark. Together, AUC and FITB show useful offline discrimination and completion ranking, while the coverage figure prevents the latter from being overstated.

### 6.3 中文对照

RQ1 询问本研究模型在一致协议下与非学习和受控学习基线相比表现如何。现有证据给出正面但有边界的答案。所选检查点的测试 AUC 为 0.8583，95% 区间为 0.8524–0.8641，明显高于确定性随机分数和平均成对余弦相似度。在三个随机种子下，其平均 AUC 高于每一种受控学习架构；AUC 差异的配对 bootstrap 区间均不包含零。其 0.0009 的标准差也低于这些比较模型。

这些结果说明，在处理后的 disjoint 场景内，模型能够较稳定地区分观察到的穿搭与人工破坏的穿搭。AUC 表示随机抽取一个正例时，其分数高于随机抽取负例的概率（Fawcett, 2006）。它并不表示 85.83% 的推荐会被认为时尚，也不衡量个人审美。使用同类替换来构造负例，比任意跨类别替换更有意义，但兼容性的定义仍受到数据集和负样本生成方式限制。

0.6025 的 FITB 准确率提供了第二种证据，因为模型需要从四个选项中选出一个答案。该值高于 0.25 的随机水平，且其置信区间没有接近随机水平。然而，官方问题中只有 29.50% 被保留。因此，这一结果只适用于具有可用向量的四个支持位置，而不是完整的 Polyvore disjoint 基准。AUC 与 FITB 共同支持模型具有实用的离线区分和补全排序能力；同时报告覆盖率则避免夸大后者。

### 6.4 Answer to RQ2: architecture, training and uncertainty — English

RQ2 asked which design and training choices were supported by the combined evidence. The strongest structural result concerns explicit pairwise comparison. Removing the pairwise module reduced mean FITB accuracy by 10.28 percentage points, with a paired 95% interval from 8.96 to 11.57 points. The corresponding classification AUC also fell. This supports the interpretation that compatibility depends on interactions among garments rather than only an outfit-level average.

The evidence for clothing-position embeddings was weaker. Removing them reduced mean FITB accuracy by 0.47 percentage points, but the paired interval ranged from –0.13 to 1.06 points. Since this interval crosses zero, the experiment does not establish a reliable improvement. Retaining the component is defensible because its cost is small and the representation has a clear semantic role, but the dissertation should not call it necessary. A larger or more position-diverse evaluation would be needed to resolve the effect.

The selected model was trained with binary cross-entropy on positive and negative outfits, then used as a scorer for candidate ranking. This is coherent because a scalar compatibility estimate can order candidates, but it is not equivalent to directly optimising FITB or user preference. A previously tested pairwise ranking objective did not improve the retained evaluation and was not selected. The negative outcome is informative: changing the loss alone did not overcome the limitations of candidate construction and representation.

Generalisation evidence is mixed. Validation and test AUC were close, differing by approximately 0.00145, which supports the use of validation performance for checkpoint selection. In contrast, training AUC exceeded test AUC by 0.1116, and validation performance declined after epoch 6. The model therefore overfit the training distribution even though early stopping selected a checkpoint that transferred consistently to the held-out test split. Early stopping controlled the observed deterioration; it did not eliminate overfitting (Prechelt, 1998).

Temperature scaling reduced validation negative log-likelihood and expected calibration error without changing candidate ordering, consistent with its role as a post-hoc calibration method (Guo et al., 2017). These improvements apply to the validation distribution and do not convert compatibility scores into probabilities of human approval. Similarly, outfit-length weighting was rejected because the small gain in overall AUC coincided with weaker performance for the lowest-performing length group and a larger group gap. This decision reflects the predefined acceptance gate rather than a general claim that weighting is ineffective.

### 6.4 中文对照

RQ2 询问综合证据支持哪些设计与训练选择。最强的结构性结果来自显式成对比较。移除成对关系模块后，平均 FITB 准确率下降 10.28 个百分点，配对 95% 区间为 8.96 至 11.57 个百分点；对应的分类 AUC 也下降。这支持兼容性依赖服装间互动，而不只是整套穿搭平均表示的解释。

服装位置嵌入的证据较弱。移除该模块后，平均 FITB 准确率下降 0.47 个百分点，但配对区间为 –0.13 至 1.06 个百分点。由于区间跨过零，实验不能确认它带来了可靠提升。这个组件成本较低且具有明确语义，因此保留是可以解释的，但论文不能把它称为必要组件。要判断其作用，需要更大或位置类型更丰富的评价。

所选模型通过正负穿搭上的二元交叉熵训练，然后作为候选排序的评分器。标量兼容性分数可以用于排列候选，因此逻辑上是一致的，但它不等同于直接优化 FITB 或用户偏好。此前测试的配对排序目标没有改善保留的评价，因此没有被选中。这个负结果同样有意义：只更换损失函数并不能解决候选构造和表示的限制。

泛化证据是混合的。验证 AUC 与测试 AUC 很接近，相差约 0.00145，这支持使用验证表现选择检查点。另一方面，训练 AUC 比测试 AUC 高 0.1116，而且验证表现从第 6 个 epoch 后开始下降。因此，模型确实对训练分布产生了过拟合，只是早停选出的检查点能较稳定地迁移到保留测试集。早停控制了观察到的进一步退化，但没有消除过拟合（Prechelt, 1998）。

温度缩放降低了验证集负对数似然和期望校准误差，同时不改变候选排序，这符合其后处理校准方法的作用（Guo et al., 2017）。这些改善只针对验证分布，不能把兼容性分数变成“用户认可概率”。同样，穿搭长度加权被拒绝，是因为整体 AUC 的轻微提升伴随最弱长度组表现下降和组间差距扩大。该决定来自预先设定的接受条件，而不是宣称所有加权方法都无效。

### 6.5 Answer to RQ3: candidate search and system control — English

RQ3 concerned the trade-off between quota-limited retrieval and exact constrained search over abstract garment states. In the 18-query comparison, the quota path examined only 3.41% of combinations on average. This reduced median CPU time from 0.474 seconds for exact search to 0.028 seconds, but it recovered the exact highest-scoring result in only 27.78% of queries and achieved 24.44% mean recall of the exact top ten. The mean score loss was positive, with a bootstrap interval that did not include zero.

For the measured candidate space, exact enumeration therefore provided a meaningful improvement in completeness according to the deployed model objective, while remaining within a recorded median of half a second and a maximum of 3.206 seconds. This supports its use in the current prototype. It does not prove that exact search will remain practical for an unrestricted catalogue. The number of complete combinations grows multiplicatively with candidates per slot, so later expansion may require pruning or approximate retrieval. Methods for large-scale similarity search are well established (Johnson et al., 2021), but adopting one would create a new recall–latency trade-off that should be measured rather than assumed.

The abstract candidate representation also changed the role of recommendation. Polyvore did not become a catalogue from which users were instructed to buy individual archived products. Instead, training-split evidence was aggregated into product-independent combinations of position, broad garment type, colour, style and clothing range. This matched the interface, which displays an abstract coloured icon unless an item belongs to the user’s local wardrobe. Compared with the earlier retailer-derived space, the artefact increased visible type–colour combinations from 122 to 271. This is evidence of broader representational coverage, not proof that every combination exists in shops or will be preferred by a user.

The search remained subordinate to explicit feasibility rules. The user’s selected item was fixed, weather logic could mask the outer-layer position, and clothing range was chosen manually. All 36 weather–style coverage cases retained the four logical positions. These checks support functional completeness within the defined representation. They do not assess whether every recommendation is seasonally or culturally appropriate, and the deterministic fallback ensures continuity rather than stylistic correctness.

### 6.5 中文对照

RQ3 关注在抽象服装状态上使用限额检索与精确约束搜索的取舍。在 18 个查询的比较中，限额路径平均只检查 3.41% 的组合，使 CPU 中位时间从精确搜索的 0.474 秒降到 0.028 秒；但它只在 27.78% 的查询中找回精确最高分结果，对精确前十的平均召回率为 24.44%。平均分数损失为正，其 bootstrap 区间不包含零。

因此，对当前测量的候选空间而言，精确枚举按照部署模型的评分目标带来了有意义的完整性提升，同时中位时间保持在半秒以内，最大时间为 3.206 秒。这支持在当前原型中采用精确搜索，但不能证明它对无限扩大的商品库仍然可行。完整组合数量会随每个位置候选数相乘增长，未来扩展可能需要剪枝或近似检索。大规模相似度搜索已有成熟方法（Johnson et al., 2021），但采用这些方法会产生新的召回率与延迟取舍，仍需实际测量，不能直接假定。

抽象候选表示也改变了推荐的含义。Polyvore 并未变成一个要求用户购买历史商品的目录；相反，训练部分的证据被整理为与具体产品无关的位置、服装大类、颜色、风格和服装范围组合。这与界面一致：除非衣服属于用户本地衣橱，否则推荐只显示抽象填色图标。与早期零售商候选空间相比，该产物把可见类型—颜色组合从 122 扩展到 271。这说明表示覆盖更广，但不能证明每一种组合都能在商店购买或会受到用户喜欢。

搜索仍然服从显式可行性规则。用户选中的衣服保持固定；天气逻辑可以屏蔽外套位置；服装范围由用户主动选择。36 个天气—风格覆盖案例都保留了四个逻辑位置。这些检查支持系统在既定表示范围内具有功能完整性，但不能评价每条推荐是否在季节或文化上都合适；确定性回退保证的是流程连续，而不是风格正确。

### 6.6 Achievement of objectives — English

The seven objectives were achieved to different degrees. O1 was addressed through the critical review of fashion compatibility, transferable visual representations, evaluation, search and responsible recommendation. O2 was achieved at the implemented research boundary: the pipeline uses secondary data, no live scraping and no participant data, although final submission still requires the ethics-waiver and training evidence to be inserted in the reserved appendices. O3 was achieved through the frozen-SigLIP compatibility pipeline and saved lightweight head.

O4 was achieved through random and cosine controls, a mean-pooled logistic model, and controlled recurrent, type-aware and attention-based models. These are architecture-family comparisons rather than exact published reproductions. O5 was substantially achieved through discrimination metrics, FITB, calibration, ablation, three seeds, confidence intervals, overfitting, length imbalance, runtime and domain diagnostics. It was not achieved for demographic fairness or user-centred outcomes because those variables were neither available nor ethically collected. O6 was achieved within the bounded prototype through abstract candidates, exact search, explicit constraints and fallback. O7 is addressed by the present interpretation, which distinguishes demonstrated system behaviour from unsupported claims about users and markets.

The objectives therefore support completion of the stated project, but not completion of every problem in fashion recommendation. The most important partial areas are benchmark coverage, menswear representation, target-domain validation and human evaluation. Reporting these as partial rather than silently redefining the objectives preserves the connection between the research plan and the evidence produced.

### 6.6 中文对照

七项目标的完成程度不同。O1 通过对穿搭兼容性、可迁移视觉表示、评价、搜索和负责任推荐的批判性综述得到落实。O2 在已实现的研究边界内完成：流程只使用二手数据，不实时爬取，也不使用参与者数据；但最终提交前仍需把伦理豁免和培训证明放入预留附录。O3 通过冻结 SigLIP 的兼容性流程和已保存的轻量模型头完成。

O4 通过随机与余弦控制、平均池化逻辑回归，以及受控的循环、类别感知和注意力模型完成。这些属于架构方向比较，而不是对原论文的精确复现。O5 通过区分指标、FITB、校准、消融、三个随机种子、置信区间、过拟合、长度不平衡、运行时间和域诊断得到较充分完成；但由于没有相应变量，也没有进行伦理许可下的数据收集，人口统计公平性和用户中心结果并未完成。O6 在限定原型范围内通过抽象候选、精确搜索、显式约束和回退完成。O7 由本章的解释完成，即把已经证明的系统行为与没有证据支持的用户和市场结论分开。

因此，这些目标能够支持既定项目的完成，但不代表解决了服装推荐中的所有问题。完成度较低的部分主要是基准覆盖、男装代表性、目标域验证和真人评价。如实将这些内容列为部分完成，而不是悄悄重新定义目标，有助于保持研究计划与实际证据之间的一致性。

### 6.7 Validity and confidence in the findings — English

Construct validity is limited by the operational definition of compatibility. The model learned from observed Polyvore outfits and same-category replacements. These labels are reproducible, but they cannot capture all reasons why a person might combine garments. FITB is closer to candidate completion than binary classification, yet it is still an offline choice among four supplied answers. Confidence should therefore be placed in dataset-relative ranking, not in a universal judgement of style.

Internal validity was strengthened by common splits, frozen features, fixed comparison settings, validation-based checkpointing and paired analysis. The disjoint split reduces product memorisation across the main divisions. Nevertheless, comparison architectures were adaptations, and implementation choices may favour some models. Three seeds expose part of training variation but are not a complete characterisation of all hyperparameter uncertainty. Bootstrap intervals describe uncertainty conditional on the retained samples and scoring procedure; they do not correct sampling bias in the original dataset (Efron and Tibshirani, 1993; Bouthillier et al., 2021).

External validity is the most restricted area. The feature-domain classifier separated Polyvore and Zara embeddings with AUC 1.000. This shows that the two sampled sources were readily distinguishable in the chosen representation, but it does not quantify how much compatibility performance would fall on new retail or user photographs. The four-slot representation excludes dresses and other multi-slot garments, and only 29.50% of official FITB questions were evaluable. Menswear evidence is particularly sparse. The findings should therefore be generalised only to similar processed data and supported garment structures.

Ecological validity is also untested because no participant study was conducted. The application demonstrates that the components can be integrated and that tested actions function, not that the experience is useful in daily life. This distinction is consistent with the wider warning that recommender evaluation requires metrics matched to the intended task and that offline accuracy alone is incomplete (Herlocker et al., 2004; Deldjoo et al., 2023).

### 6.7 中文对照

构念有效性受到兼容性操作定义的限制。模型从观察到的 Polyvore 穿搭和同类替换样本中学习。这些标签可以复现，但不能涵盖一个人组合衣服的所有原因。FITB 比二分类更接近候选补全，但仍然只是从四个给定答案中进行离线选择。因此，应当对“相对于该数据集的排序能力”给予信心，而不是把模型视为普遍风格裁判。

一致的数据划分、冻结特征、固定比较设置、基于验证集的检查点选择和配对分析增强了内部有效性。Disjoint 划分降低了模型在主要数据部分之间记忆相同商品的风险。但比较架构仍然是适配实现，某些实现选择可能更有利于部分模型。三个随机种子揭示了一部分训练波动，却不能完整覆盖所有超参数不确定性。Bootstrap 区间描述的是在现有样本和评分流程条件下的不确定性，不能修正原始数据集的采样偏差（Efron and Tibshirani, 1993; Bouthillier et al., 2021）。

外部有效性是限制最明显的部分。特征域分类器以 AUC 1.000 区分 Polyvore 与 Zara 向量，说明在所选表示中，这两个抽样来源很容易区分；但它不能量化模型在新零售图片或用户照片上的兼容性表现会下降多少。四位置表示不支持连衣裙等跨位置服装，官方 FITB 只有 29.50% 可评价，男装证据尤其稀少。因此，结果只应推广到相近的处理后数据和受支持服装结构。

由于没有参与者研究，生态有效性也没有得到检验。应用证明了组件能够整合、测试过的操作能够运行，但不能证明日常使用体验有效。这与推荐系统领域的普遍提醒一致：评价指标必须匹配目标任务，离线准确率本身并不完整（Herlocker et al., 2004; Deldjoo et al., 2023）。

### 6.8 Contribution, benefit and responsible use — English

The principal contribution is not a new foundation model. It is an auditable combination of reused visual representations, a small relational compatibility head, controlled evaluation and constrained abstract search. The implementation demonstrates how a resource-limited project can investigate compatibility without training a large image encoder end to end. It also separates learned scoring from rules that express feasibility, which makes it possible to identify whether a failure came from model ranking, candidate availability or a contextual constraint.

The product-independent output provides a practical design benefit. A recommendation can describe a broad type and colour without claiming that one brand owns the suitable product. The selected garment remains visible and fixed; abstract icons distinguish generated suggestions from garments that the user actually saved. Local wardrobe storage and the absence of automatic gender inference further align the prototype with its privacy boundary. These are implemented properties, not measured user benefits.

Industrial or commercial worth remains prospective. The system could support wardrobe organisation, style exploration or later links to retailers and independent designers, but no conversion, retention, willingness-to-pay or designer outcome was measured. A commercial deployment would also require current product data, verified image licences, accessibility work, security review, service monitoring and evidence that recommendations do not systematically disadvantage clothing ranges or styles. The dissertation can therefore claim a working and extensible research artefact, not a validated business.

### 6.8 中文对照

本研究的主要贡献并不是创造新的基础模型，而是把复用的视觉表示、小型关系兼容性模型、受控评价和受约束抽象搜索组合成可审计流程。实现结果说明，在资源有限的项目中，可以不端到端训练大型图片编码器，仍然研究服装兼容性。系统还把学习评分与表达可行性的规则分离，因此能够判断问题来自模型排序、候选可用性还是情境约束。

与产品无关的输出带来实际设计价值。推荐可以描述服装大类和颜色，而不声称合适商品只属于某个品牌。用户选中的衣服保持显示和固定；抽象图标则把生成建议与用户实际保存的衣服区分开。本地衣橱存储和不自动推断性别也符合原型的隐私边界。这些是已经实现的系统属性，不是经过测量的用户收益。

产业或商业价值仍然只是潜在可能。该系统未来可以支持衣橱整理、风格探索，或连接零售商和独立设计师，但本研究没有测量转化率、留存、付费意愿或设计师收益。商业部署还需要实时商品数据、经过核实的图像许可、无障碍改进、安全审查、服务监控，以及不会系统性损害某些服装范围或风格的证据。因此，论文可以声称形成了可运行、可扩展的研究产物，但不能声称商业模式已经得到验证。

### 6.9 Prioritised future work — English

Future work should address evidence gaps in order of risk rather than add features solely for visual complexity. The first priority is licensed data expansion. A new dataset should contain broader menswear coverage, supported provenance and explicit structures for dresses and other one-piece garments. This would require data mapping, new slot-occupancy rules, regenerated frozen embeddings, retraining and repetition of the full AUC, FITB, ablation and subgroup evaluation. Dresses should not simply be relabelled as inner tops because doing so would falsely permit a separate bottom.

The second priority is target-domain validation. A modest, legally reusable set of complete outfits from a contemporary target domain should be labelled under a documented protocol. It could then test whether the compatibility ordering transfers beyond Polyvore and whether calibration remains valid. If performance falls, carefully bounded encoder adaptation or domain-robust training could be compared with the current frozen baseline. Such work requires additional GPU time and licensing review, but the main cost is reliable data rather than model size.

The third priority is a participant evaluation only after ethics approval. A study could compare recommendations from the learned model, the rule fallback and a controlled baseline while measuring perceived usefulness, choice diversity and trust. Recruitment, informed consent, withdrawal, secure data handling and a preregistered analysis plan would be required. Until this is completed, the dissertation should not infer satisfaction from interface completeness.

The fourth priority is scaling candidate search. Exact enumeration should remain the reference while the current candidate space is bounded. If the catalogue grows enough for latency to become unacceptable, structured pruning or approximate search should be introduced and evaluated against exact top-k recall, score loss and runtime. This preserves a measurable accuracy–efficiency trade-off instead of replacing exact search without a reference.

Long-term local personalisation is lower priority than these validation tasks. Learning from clicks can create feedback loops in which early exposure shapes later data (Chaney et al., 2018). Any future personal model would need explicit consent, reset and deletion controls, exploration, exposure logging and an evaluation that distinguishes repeated exposure from genuine preference. Local execution reduces data transfer but does not remove these methodological obligations.

### 6.9 中文对照

未来工作应按证据风险排序，而不是只为了视觉复杂度继续增加功能。第一优先级是扩大具有许可的数据。新数据应包含更充分的男装覆盖、可核实来源，以及连衣裙和其他一件式服装的明确结构。这需要进行数据映射、设计新的位置占用规则、重新生成冻结向量、重新训练，并重复完整的 AUC、FITB、消融和子组评价。不能简单把连衣裙重标为内搭，因为这样会错误地允许系统同时推荐下装。

第二优先级是目标域验证。应建立一个规模适中、可以合法复用、来自当代目标场景的完整穿搭数据集，并使用有记录的标注协议。它可用于检验兼容性排序能否迁移到 Polyvore 之外，以及校准是否仍然有效。如果表现下降，可以把谨慎限定的编码器适配或域稳健训练与当前冻结基线进行比较。这需要额外 GPU 时间和许可审查，但主要成本是可靠数据，而不是模型规模。

第三优先级是在获得伦理批准后进行参与者评价。研究可以比较学习模型、规则回退和受控基线的推荐，并测量感知有用性、选择多样性和信任。需要完成招募、知情同意、退出机制、安全数据处理和预先登记的分析计划。在此之前，论文不能从界面功能完整推断用户满意度。

第四优先级是扩展候选搜索。在当前候选空间受限时，精确枚举应继续作为参照。如果目录增长到延迟不可接受，可以引入结构化剪枝或近似搜索，并针对精确 top-k 召回率、分数损失和时间进行评价。这样能够保留可测量的准确性—效率取舍，而不是在没有参照的情况下替换精确搜索。

长期本地个性化的优先级低于上述验证工作。从点击学习可能产生反馈循环，使早期曝光影响后续数据（Chaney et al., 2018）。任何未来个人模型都需要明确同意、重置和删除控制、探索机制、曝光记录，以及能够区分重复曝光和真实偏好的评价。本地运行能够减少数据传输，但不能消除这些方法责任。

### 6.10 Summary — English

The findings answer the primary question to a meaningful but restricted extent. A lightweight relational head on frozen SigLIP embeddings produced stronger offline discrimination than the controlled alternatives, and the pairwise component contributed clearly to completion ranking. Exact constrained search improved completeness according to the model score at acceptable latency for the measured candidate space. Confidence is highest for these within-dataset and bounded-system claims. It is lower for unsupported garment structures, menswear, new image domains and all claims involving human preference. The project’s value lies in the controlled evidence and transparent boundary between what was implemented, what was measured and what remains untested.

### 6.10 中文对照

研究结果在有意义但受限制的范围内回答了主要研究问题。建立在冻结 SigLIP 向量上的轻量关系模型，在离线区分方面优于受控替代方法；成对关系组件也对补全排序作出了明确贡献。对当前测量的候选空间而言，精确约束搜索以可接受延迟提升了模型评分意义上的完整性。最可信的是这些数据集内部和有边界的系统结论；对不支持的服装结构、男装、新图片域以及所有涉及真人偏好的结论，信心较低。本项目的价值来自受控证据，以及对“已经实现、已经测量、尚未测试”三者边界的透明说明。

---

## Chapter 7: Conclusion

### 7.1 Conclusion — English

This dissertation investigated whether a lightweight outfit-compatibility model, built on frozen vision–language embeddings and combined with constrained search over abstract garment states, could support accurate, practical and auditable complete-outfit recommendation using secondary data. The result is a working research prototype and a controlled body of offline evidence. It is not a universal fashion model, a complete reproduction of published systems or a validated measure of human taste.

Within the processed Polyvore disjoint setting, the selected model achieved test AUC 0.8583 and FITB accuracy 0.6025 on the retained four-position subset. Across three seeds, its mean AUC of 0.8596 exceeded the controlled recurrent, type-aware and attention-based alternatives. The structural ablation identified the pairwise module as the clearest useful component: removing it reduced mean FITB accuracy by 10.28 percentage points. The effect of clothing-position embeddings remained inconclusive because its paired interval crossed zero.

The model generalised consistently from validation to test, but the larger train–test gap showed that overfitting remained present. Early stopping selected the epoch-six checkpoint, temperature scaling improved validation calibration, and an outfit-length weighting alternative was rejected because it worsened the weakest group and increased the group gap. These decisions illustrate the central methodological conclusion of the project: model selection should depend on several aligned measures and predefined acceptance conditions, not on one favourable aggregate score.

The system contribution extended beyond classification. A training-derived artefact represented 1,670 product-independent garment concepts rather than restricting recommendations to Zara products. Exact constrained search recovered solutions that the quota-limited path frequently missed, with median measured CPU latency of 0.474 seconds. Weather, clothing range, fixed-item and slot constraints remained explicit, and a deterministic fallback supported operation when learned artefacts were unavailable. This separation between learned compatibility, feasible candidate generation and interface state made the system easier to inspect and test.

The primary research question can therefore be answered as follows: the proposed lightweight approach provides useful and comparatively strong offline compatibility ranking, together with practical exact search for the present bounded candidate space, but only for the garment structures and data distributions that were evaluated. Confidence is reduced by 29.50% FITB coverage, sparse menswear evidence, the strong Polyvore–Zara domain distinction, the exclusion of one-piece garments and the absence of participant evaluation. No conclusion is drawn about user satisfaction, demographic fairness, retail conversion or universal style quality.

The seven objectives were addressed sufficiently to complete the stated investigation. The literature, secondary-data pipeline, lightweight model, controlled baselines, multifaceted evaluation and integrated prototype were delivered. The objectives relating to bias and practical benefit were achieved only as critical audits: demographic performance and real user benefit remain unmeasured. This is an important distinction rather than an implementation failure, because ethically unavailable evidence should not be replaced by speculation.

The most valuable next step is not automatic personalisation or a larger model. It is a better licensed evaluation base: broader clothing-range representation, explicit one-piece structures and a contemporary target-domain outfit set. Retraining and repeating the controlled evaluation on that base would show whether the current relational design transfers. A participant study could then be considered after ethics approval. Approximate search and consent-based local preference learning should be introduced only if catalogue scale and validated user needs justify their additional risks.

Overall, the project demonstrates a proportionate Applied AI approach. It reuses a powerful frozen representation, trains only the component required for the local research question, compares credible alternatives, records negative as well as positive results, and integrates the selected model without hiding deterministic constraints. Its contribution is the evidence-backed combination of model, search and transparent system boundaries, rather than a claim that computational compatibility resolves the social and personal nature of clothing choice.

### 7.1 中文对照

本论文研究了这样一个问题：建立在冻结视觉—语言向量上的轻量穿搭兼容性模型，与抽象服装状态上的约束搜索结合后，能否使用二手数据支持准确、实用且可审计的完整穿搭推荐。最终成果包括一个可运行的研究原型和一套受控离线证据。它不是普遍适用的时尚模型，也不是对公开系统的完整复现，更不是衡量真人审美的有效工具。

在处理后的 Polyvore disjoint 场景内，所选模型的测试 AUC 为 0.8583，在保留的四位置子集上 FITB 准确率为 0.6025。在三个随机种子下，0.8596 的平均 AUC 高于受控的循环、类别感知和注意力替代模型。结构消融把成对关系模块确定为最明确的有效组件：移除后，平均 FITB 准确率下降 10.28 个百分点。服装位置嵌入的作用仍不能确定，因为其配对区间跨过零。

模型从验证集到测试集表现一致，但更大的训练—测试差距说明过拟合仍然存在。早停选择了第 6 个 epoch 的检查点；温度缩放改善了验证校准；穿搭长度加权方案则因最弱组下降和组间差距扩大而被拒绝。这些决定体现了项目的核心方法结论：模型选择应依靠多个相互配合的指标和预先设定的接受条件，而不是单一有利的总体分数。

系统贡献不止于分类。由训练数据生成的产物表示 1,670 个与产品无关的服装概念，没有把推荐限制在 Zara 商品中。精确约束搜索找回了限额路径经常错过的解，其记录的 CPU 中位延迟为 0.474 秒。天气、服装范围、固定用户衣服和位置约束仍由显式规则处理；学习产物不可用时，确定性回退维持系统运行。把学习兼容性、可行候选生成和界面状态分开，使系统更容易检查和测试。

因此，主要研究问题可以这样回答：所提出的轻量方法提供了有用且相对较强的离线兼容性排序，并在当前有限候选空间内实现了实用的精确搜索；但结论只适用于实际评价过的服装结构和数据分布。29.50% 的 FITB 覆盖、稀少的男装证据、明显的 Polyvore—Zara 域差异、不支持一件式服装以及缺少参与者评价，都降低了结论可信范围。论文不对用户满意度、人口统计公平性、零售转化或普遍风格质量作出结论。

七项目标已经得到足够落实，可以完成既定研究。项目完成了文献分析、二手数据流程、轻量模型、受控基线、多角度评价和整合原型。涉及偏差和实践收益的目标只完成到批判性审计层面：人口统计表现和真实用户收益仍未测量。这是一项重要区分，而不是用猜测替代伦理上无法取得的证据。

最有价值的下一步不是立即加入自动个性化，也不是单纯扩大模型，而是建立更好的许可评价基础，包括更充分的服装范围代表性、明确的一件式服装结构，以及来自当代目标域的穿搭数据。在该基础上重新训练并重复受控评价，才能判断当前关系设计能否迁移。之后可在获得伦理批准后考虑参与者研究。只有当目录规模和已验证的用户需求确实需要时，才应引入近似搜索和基于同意的本地偏好学习。

总体而言，本项目体现了一种与 Applied AI 相称的研究方式：复用强大的冻结表示，只训练本地研究问题真正需要的组件；比较可信替代方法；同时记录正面和负面结果；并在整合模型时不隐藏确定性约束。它的贡献是有证据支持的模型、搜索和透明系统边界组合，而不是声称计算得到的兼容性能够解决服装选择中的社会性和个人性问题。

## Citation audit for this review draft

All sources cited above already exist in `dissertation/references.bib` with DOI or accessible publication URLs: Herlocker et al. (2004), Han et al. (2017), Vasileva et al. (2018), Sarkar et al. (2023), Zhai et al. (2023), Fawcett (2006), Prechelt (1998), Guo et al. (2017), Johnson et al. (2021), Keyes (2018), Mehrabi et al. (2021), Efron and Tibshirani (1993), Bouthillier et al. (2021), Deldjoo et al. (2023), and Chaney et al. (2018). No new bibliographic facts have been invented in this draft.
