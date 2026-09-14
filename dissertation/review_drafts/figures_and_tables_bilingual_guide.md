# Cove dissertation figures and tables — bilingual reading guide

> Purpose: this file is a reading companion for reviewing the dissertation's visual and tabular artefacts. It does not replace the formal English dissertation. Each English paragraph is followed immediately by its Chinese counterpart.
>
> 用途：本文件是论文图表的双语通读说明，用于帮助检查图表是否准确表达项目。它不替代正式英文论文。每段英文之后均紧跟对应中文。

## 1. Functional scope of the Cove system / Cove 系统功能范围图

**English.** The functional-scope diagram defines what the implemented prototype accepts, processes and returns. Its left-hand column contains user and context inputs: a selected garment, city weather, style, clothing range and locally saved wardrobe content. The centre column separates garment recognition, candidate generation, complete-outfit compatibility ranking and wardrobe management. The right-hand column records the visible outcomes, including the main recommendation, a diverse alternative, model-ranked colour alternatives, weather-aware cues and saved outfits.

**中文。** 功能范围图界定了当前原型接收什么、处理什么以及返回什么。左列是用户和情境输入，包括选中的服装、城市天气、风格、服装范围以及保存在本地的衣橱内容；中间一列把服装识别、候选生成、完整穿搭兼容性排序和衣橱管理分开；右列记录用户能够看到的结果，包括主推荐、差异化备选、由模型排序的颜色替代方案、天气提示和保存的穿搭。

**English.** The figure is important because it prevents the dissertation from attributing every function to the learned model. The model scores complete candidate outfits, whereas recognition, weather validation, storage and explicit feasibility rules are separate system responsibilities. The arrows show the main dependencies rather than every user-interface event.

**中文。** 这张图的重要意义在于，它避免论文把所有功能都归因于学习模型。模型负责给完整候选穿搭评分，而识别、天气校验、存储和明确的可行性规则属于其他系统模块。图中的箭头表示主要依赖关系，而不是穷举每一个界面事件。

## 2. Technical route / 技术路线图

**English.** The technical-route diagram follows the research from secondary fashion data to the evaluated Cove prototype. Polyvore disjoint images and metadata are mapped to four supported clothing positions and used to construct positive and corrupted negative outfits. A frozen SigLIP encoder converts garments into 768-dimensional representations. These processed tensors are then used to train the compatibility head, select a checkpoint with validation data and calibrate its output.

**中文。** 技术路线图从二手时尚数据开始，一直追踪到经过评价的 Cove 原型。Polyvore disjoint 的图片和元数据首先被映射到四个支持的服装位置，并用于构造正穿搭和人工破坏的负穿搭；冻结的 SigLIP 编码器把服装转换成 768 维表示；处理后的张量用于训练兼容性模型头、依据验证集选择检查点，并对输出进行校准。

**English.** The lower part of the diagram connects model development to baselines, FITB evaluation, ablation, bias and generalisation checks. The selected checkpoint and abstract clothing prototypes then enter constraint filtering and exact candidate search. This figure functions as a concise map of the methodology and evidence chain; it also makes clear that Zara is not the source of compatibility labels or the only recommendation catalogue.

**中文。** 图的下半部分把模型开发与基线比较、FITB 评价、消融、偏差和泛化检查连接起来。所选模型检查点与抽象服装原型随后进入约束过滤和精确候选搜索。这张图相当于方法与证据链的简明地图，同时明确说明 Zara 既不是兼容性标签来源，也不是唯一的推荐候选库。

## 3. Lightweight compatibility-model architecture / 轻量兼容性模型架构图

**English.** The architecture diagram shows the trainable task-specific part of the recommendation model. Up to four frozen 768-dimensional SigLIP embeddings and a position mask enter layer normalisation, linear projection and GELU activation. Learned position embeddings are concatenated to the projected item representations. The model then forms a masked global outfit summary and six absolute pairwise-difference representations, corresponding to all pairs among four positions.

**中文。** 模型架构图展示推荐模型中针对本任务训练的部分。最多四个冻结的 768 维 SigLIP 向量和一个位置掩码首先经过层归一化、线性投影与 GELU 激活；学习得到的位置嵌入随后与投影后的单品表示拼接。模型再形成带掩码的全局穿搭摘要，并构造四个位置之间全部六组绝对成对差异表示。

**English.** The global and pairwise summaries are concatenated and passed through a small multilayer perceptron to produce a compatibility logit, which is converted into a calibrated score for runtime ranking. The pairwise branch is central because compatibility concerns relationships between garments rather than only the average appearance of an outfit. The diagram also communicates the project's lightweight boundary: SigLIP representation learning is reused and frozen; only the smaller compatibility head is trained.

**中文。** 全局摘要与成对摘要拼接后进入小型多层感知机，产生兼容性 logit，并转换成运行时排序使用的校准分数。成对关系分支很关键，因为穿搭兼容性关注衣服之间的关系，而不只是整套衣服外观的平均值。该图还说明了项目的轻量化边界：SigLIP 的表示学习成果被复用并冻结，真正训练的是较小的兼容性模型头。

## 4. Evaluation framework / 评价框架图

**English.** The evaluation-framework diagram begins with disjoint training, validation and held-out test data and applies the same data protocol to the selected model and controlled alternatives. Evidence is divided into four groups: binary discrimination, completion ranking, robustness and complete-system behaviour. Corresponding measures include AUC and threshold metrics; FITB, MRR and NDCG; multiple seeds, ablation and calibration; and coverage, exact search and runtime.

**中文。** 评价框架图从相互独立的训练集、验证集和保留测试集出发，并把一致的数据协议用于最终模型和受控替代模型。证据被划分为四组：二分类区分、穿搭补全排序、稳健性以及完整系统行为。相应指标包括 AUC 与阈值指标；FITB、MRR 和 NDCG；多随机种子、消融与校准；以及覆盖率、精确搜索和运行时间。

**English.** Its purpose is to show that no single score is sufficient for the dissertation's claims. Confidence intervals, retained benchmark coverage and stated limitations accompany the metrics. Model promotion and deployment are therefore based on several forms of evidence rather than the highest observed AUC alone.

**中文。** 这张图的作用是说明，单一分数不足以支持论文结论。指标需要同时配合置信区间、保留的基准覆盖率和明确的局限解释。因此，模型晋级和系统部署依据多种证据，而不是只选择观察到的最高 AUC。

## 5. Runtime recommendation and fallback flow / 运行时推荐与回退流程图

**English.** The runtime-flow diagram starts when a user selects a garment from a photograph or the local wardrobe. The system analyses its broad type, colour and style and obtains a SigLIP representation. It then checks whether the compatible model checkpoint and candidate artefacts are available. When they are available, the selected item is fixed, infeasible states are filtered by position, weather, style and clothing range, and complete candidates are scored in batches through exact enumeration.

**中文。** 运行时流程图从用户在照片或本地衣橱中选中一件服装开始。系统分析它的大类、颜色和风格，并取得 SigLIP 表示；随后检查兼容性模型检查点和候选产物是否可用。如果这些文件可用，系统就固定用户选择的服装，根据位置、天气、风格和服装范围过滤不可行状态，再通过精确枚举分批评价完整候选组合。

**English.** If learned artefacts are unavailable or incompatible, the system follows a deterministic content-based fallback rather than stopping. Both paths produce a primary outfit, a different alternative and feasible position-level replacements. This flow documents operational resilience while keeping the fallback claim bounded: continued operation does not mean that a rule-ranked output is equivalent to a learned compatibility score.

**中文。** 如果学习模型产物缺失或不兼容，系统会使用确定性的内容规则回退，而不是直接停止。两条路径最终都生成主穿搭、具有明显差异的备选，以及可行的位置级替代方案。该流程体现了系统的运行韧性，但同时限制了结论边界：系统能够继续运行，不代表规则排序结果与学习得到的兼容性分数等价。

## 6. Logical local-data relationships / 本地数据逻辑关系图

**English.** This diagram presents how user state and research artefacts relate at runtime. `wardrobe.json` contains arrays of pieces, outfits and outfit groups. A saved piece may reference a local PNG file, while a saved outfit contains values or absences for four supported positions. `user_preferences.json` stores up to six saved cities. The application loads these local records together with the compatibility checkpoint and abstract prototype catalogue.

**中文。** 这张图展示用户状态与研究产物在运行时如何关联。`wardrobe.json` 包含单品、穿搭和穿搭分组数组；保存的单品可以引用本地 PNG 文件，保存的穿搭则为四个支持位置保存服装值或空缺状态；`user_preferences.json` 最多保存六个城市。应用运行时会同时载入这些本地记录、兼容性模型检查点以及抽象服装原型目录。

**English.** Evaluation JSON files, manifests, hashes and training histories form a separate research-evidence layer and are not user profiles. The figure deliberately describes logical relationships instead of database tables because the prototype does not implement a relational database. Its value is accuracy and auditability: readers can see which information is personal, which files support inference and which outputs document the experiments.

**中文。** 评价 JSON、清单、哈希和训练历史构成单独的研究证据层，并不属于用户画像。由于原型没有实现关系型数据库，这张图有意描述逻辑关系，而不是虚构数据库表。它的价值在于准确性和可审计性：读者能够分辨哪些是个人信息、哪些文件支持模型推理，以及哪些输出用于记录实验。

## 7. UML use-case view / UML 用例图

**English.** The use-case diagram records the interactions available to a single local wardrobe user. The user can set or restore a city, upload and circle a garment, choose style and clothing range, request a complete outfit, inspect alternatives, save results and manage the wardrobe. Garment analysis is included by image selection and recommendation. Inspecting alternatives extends the main recommendation journey rather than being required for every result.

**中文。** 用例图记录单个本地衣橱用户可以进行的交互。用户能够设置或恢复城市、上传并圈选服装、选择风格与服装范围、请求完整穿搭、查看备选、保存结果以及管理衣橱。服装分析是图片选择和推荐流程所包含的步骤；查看备选则是对主推荐流程的扩展，并非每一次结果都必须执行。

**English.** Open-Meteo appears outside the Cove system boundary as the only external service in this view. It receives a city query for geocoding and weather retrieval but does not receive wardrobe images. The diagram is useful for defining actors, privacy boundaries and implemented user goals without introducing administrator roles or login flows that do not exist in the prototype.

**中文。** Open-Meteo 位于 Cove 系统边界之外，是该图中的外部服务。它只接收城市查询以进行地理编码和天气获取，不接收衣橱图片。这张图用于明确参与者、隐私边界和已经实现的用户目标，同时避免加入原型中并不存在的管理员角色或登录流程。

## 8. Overall UML-style component view / 整体 UML 风格组件图

**English.** The overall component diagram places the Streamlit presentation and session-state orchestration at the top. It calls garment recognition, colour analysis, weather, recommendation and wardrobe modules. The recommendation module can use either the trained-model adapter and candidate ranker or the deterministic fallback. Separate stores provide the model checkpoint, abstract prototypes, local JSON documents and wardrobe images, while Open-Meteo supplies weather data.

**中文。** 整体组件图把 Streamlit 展示层和会话状态编排放在顶端。它调用服装识别、颜色分析、天气、推荐和衣橱模块。推荐模块既可以经过训练模型适配器和候选排序器，也可以进入确定性回退。模型检查点、抽象原型、本地 JSON 文档和衣橱图片由不同存储提供，天气数据则来自 Open-Meteo。

**English.** This is a component view rather than a conventional overall class diagram because most of the application is function-oriented Python. Inventing service, repository and data-transfer-object classes would misrepresent the code. The diagram's function is to communicate real software boundaries and dependencies at a readable scale.

**中文。** 这是一张组件图，而不是传统意义上的整体类图，因为应用的大部分 Python 代码采用函数式组织。如果凭空加入 Service、Repository 或 DTO 类，就会错误描述实际代码。该图的功能是在容易阅读的尺度上表达真实的软件边界与依赖关系。

## 9. Recommendation-module class and dependency view / 推荐模块类与依赖图

**English.** This module-level diagram focuses on the object-oriented structures that genuinely exist in the recommendation path. `ModelConfig` supplies the embedding, hidden, position and dropout settings used to construct `CompatibilityRanker`. The model exposes forward scoring, preprojected scoring and probability conversion. `OutfitCandidateRanker` holds the trained model and performs exact or bounded candidate ranking while recording search statistics.

**中文。** 这张模块级图只聚焦推荐路径中确实存在的面向对象结构。`ModelConfig` 提供构造 `CompatibilityRanker` 所需的向量维度、隐藏层维度、位置维度和 dropout 设置；模型提供前向评分、预投影评分和概率转换；`OutfitCandidateRanker` 持有训练模型，执行精确或有限候选排序，并记录搜索统计。

**English.** The lower row adds the function-based trained-model adapter and recommendation facade, plus the deterministic `LightweightOutfitRanker`. Solid arrows indicate construction or ownership relationships; dashed arrows indicate software dependencies. Its purpose is to connect the architecture described in the methodology to identifiable code units and to show where learned and fallback paths diverge.

**中文。** 下方一行补充了函数式训练模型适配器、推荐门面以及确定性的 `LightweightOutfitRanker`。实线箭头表示构造或持有关系，虚线箭头表示软件依赖。该图的意义在于把方法章节描述的架构对应到可以识别的代码单元，并明确学习路径与回退路径在哪里分开。

## 10. Logical entity-relationship view / 逻辑实体关系图

**English.** The entity-relationship view describes the logical schema serialised to local files. One wardrobe store can contain zero or more pieces and outfits and at least the default outfit groups. Each outfit contains exactly four slot entries; each slot contains either a snapshot of one piece or a null value. A piece may reference one image file, and a group may contain multiple outfits. Saved city preferences are stored separately.

**中文。** 实体关系图描述序列化到本地文件中的逻辑模式。一个衣橱存储可以包含零个或多个单品与穿搭，并至少包含默认穿搭分组；每套穿搭固定含有四个位置条目，每个位置保存一件服装的快照或空值；一件单品可以引用一个图片文件，一个分组可以包含多套穿搭；收藏城市则单独保存。

**English.** Cardinalities make behaviours such as partial outfits, piece reuse and grouping explicit. The figure must not be interpreted as evidence of SQL tables, primary keys or foreign-key enforcement. Its function is to document persistence rules and support reasoning about deletion, reuse and privacy in the implemented file-based design.

**中文。** 图中的基数关系明确表达了未完成穿搭、单品复用和分组等行为。不能把这张图理解为系统存在 SQL 表、主键或外键约束的证据。它的功能是记录持久化规则，并支持对当前文件式设计中的删除、复用和隐私问题进行分析。

## 11. Core application-functions table / 核心应用功能表

**English.** The core-functions table assigns a principal input, output and responsibility to garment analysis, candidate control, compatibility ranking, wardrobe management and fallback. It is a compact responsibility matrix rather than an experiment table. The separation shows that the compatibility head evaluates relational coherence only after candidates have been made feasible by explicit controls.

**中文。** 核心功能表为服装分析、候选控制、兼容性排序、衣橱管理和回退分别指定主要输入、输出与职责。它是一张紧凑的职责矩阵，而不是实验结果表。这样的划分说明，候选首先由明确规则保证可行，之后兼容性模型头才负责评价服装之间的关系协调性。

**English.** This table supports traceability between the product functions and the implementation. It also makes testing easier to explain: recognition accuracy, constraint correctness, model ranking and local persistence are different concerns and should not be represented by one metric.

**中文。** 该表支持产品功能与代码实现之间的可追踪性，也使测试设计更容易说明：识别准确性、约束正确性、模型排序和本地持久化属于不同问题，不能用同一个指标概括。

## 12. Local persistence and research-artefacts table / 本地持久化与研究产物表

**English.** This table lists the concrete files behind the logical data diagrams. It distinguishes the local wardrobe store, local garment images and saved-city preferences from the compatibility checkpoint, abstract candidate prototypes and experimental evidence. Each row states the principal contents and explains how the artefact relates to the running application or the research audit trail.

**中文。** 这张表列出逻辑数据图背后的具体文件，把本地衣橱记录、用户服装图片和收藏城市偏好，与兼容性模型检查点、抽象候选原型及实验依据区分开来。每一行都说明文件的主要内容，以及它与运行中的应用或研究审计链之间的关系。

**English.** The distinction is meaningful for privacy and reproducibility. User files remain local and are Git-ignored, whereas model and evaluation artefacts document which system version produced the reported results. The table therefore complements the relationship diagrams with implementation-level file names.

**中文。** 这种区分对于隐私和可复现性很重要。用户文件保留在本地并被 Git 忽略；模型和评价产物则记录哪一个系统版本产生了论文中报告的结果。因此，该表用实现层面的文件名补充了关系图。

## 13. User stories and acceptance criteria / 用户故事与验收标准表

**English.** The user-story table translates twelve implemented interaction paths into testable requirements. The stories cover location and saved cities, image selection and recognition, context controls, outfit completion, alternatives, fallback, saving, editing, the reusable piece library and local persistence. Each story contains an observable acceptance criterion and is assigned a Must or Should priority.

**中文。** 用户故事表把十二条已实现的交互路径转换成可测试需求，覆盖地点与收藏城市、图片选择与识别、情境控制、穿搭补全、备选、回退、保存、编辑、可复用单品库和本地持久化。每条故事都有可以观察的验收条件，并被标记为 Must 或 Should 优先级。

**English.** The table's purpose is requirements traceability rather than claiming that a formal industrial agile process was followed. It helps a reader check whether the implementation meets the intended user journey and identifies which supporting conveniences are less critical than the core recommendation and privacy requirements.

**中文。** 该表的目的在于需求可追踪性，而不是声称项目完整采用了正式工业敏捷流程。它帮助读者检查实现是否覆盖预期用户旅程，并区分哪些便利功能的优先级低于核心推荐与隐私要求。

## 14. How the diagrams work together / 图表之间如何配合

**English.** The figures operate at different levels and should not be read as duplicates. The functional-scope and use-case views describe what the user can do. The technical route and evaluation framework describe how the research was conducted and judged. The model-architecture and recommendation-class views explain the learned scorer and its code dependencies. The overall component and runtime-flow views describe deployment behaviour. Finally, the local-data, ER and persistence views document storage and privacy boundaries.

**中文。** 这些图位于不同层次，不应被视为重复内容。功能范围图和用例图说明用户能够做什么；技术路线图和评价框架图说明研究怎样开展、怎样判断；模型架构图和推荐类图解释学习式评分器及其代码依赖；整体组件图和运行时流程图描述部署行为；最后，本地数据图、ER 图和持久化表记录存储方式与隐私边界。

**English.** Together they support three dissertation needs: communicating the product clearly, demonstrating that the software was actually engineered, and connecting implementation decisions to research evidence. They do not prove recommendation quality by themselves; quality claims remain grounded in the reported experiments, confidence intervals, coverage and limitations.

**中文。** 这些图表共同支持论文的三项需要：清楚介绍产品、证明软件确实经过工程实现，以及把实现决策与研究证据连接起来。图表本身不能证明推荐质量；关于质量的结论仍必须建立在论文报告的实验、置信区间、覆盖率和局限之上。

## Review note / 审阅提示

**English.** During review, check whether every label matches the current code, whether any arrow implies a function the system does not perform, and whether the distinction between learned ranking and explicit constraints remains clear. The most important conceptual boundary is that the model supplies a compatibility score, while the full recommender also depends on recognition, candidate construction, weather and clothing-range rules, exact search, storage and fallback.

**中文。** 通读时应重点检查：每个标签是否与当前代码一致、是否有箭头暗示系统并未执行的功能，以及学习式排序与明确约束之间的区别是否清楚。最重要的概念边界是：模型只提供兼容性分数，而完整推荐系统还依赖识别、候选构造、天气与服装范围规则、精确搜索、存储和回退机制。
