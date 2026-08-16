Experiment A：FSLR 全频率 Case Study —— A1–A5 结果
总体结论
Experiment A 以 FSLR 作为全频率 case study，使用 Hourly、Half-Day、Daily 和 Weekly 四类输入频率。A1–A5 的目的，是检验 PathFormer 风格的多尺度划分、固定多尺度聚合、静态可学习尺度权重、自适应 router 尺度选择，以及多频率融合，是否能够在单股票场景下带来稳定的预测增益。
总体来看，A1–A5 的结果是“有启发性但不完全支持复杂模型”的。A1 显示，不同输入频率和不同预测 horizon 的最优 patch size 并不一致，这说明时间尺度选择本身是一个有意义的问题，也支持多尺度建模的研究动机。但是，A2–A5 进一步显示，在 FSLR 单股票设定下，模型复杂度的提升并没有稳定转化为预测性能的提升。尤其是从固定多尺度聚合，到静态可学习权重，再到自适应 router，结果并没有形成一致优势。全频率融合设置尤其不稳定，在多个实验中出现了明显的预测尺度爆炸和 MSE 异常增大。
因此，A1–A5 的主要结论并不是“多尺度建模没有意义”，而是：时间尺度选择确实是一个有意义的问题，但在当前 FSLR 单股票样本下，全频率自适应融合尚未达到稳定状态。FSLR 更适合作为一个 full-frequency case study 和 stress test，而 adaptive multi-scale framework 仍需要在更广泛的多股票 panel 数据上进一步验证。
A1：Patch Size Search
A1 对每个输入频率和每个预测 horizon 分别进行了 patch size 搜索。候选 patch size 如下：
Hourly：2, 4, 8, 12
Half-Day：2, 4, 8, 10, 12
Daily：5, 10, 20, 30
Weekly：2, 4, 8, 12, 13, 14
按照 test MSE 选出的最优 patch size 如下：
Frequency	5d	10d	20d
Daily	5	20	5
Half-Day	10	12	4
Hourly	2	12	12
Weekly	2	8	13
这个结果说明，不同 frequency 和不同 horizon 对 patch size 的偏好并不一致，不存在一个对所有任务都最优的统一 patch size。这支持多尺度建模的基本动机：不同预测期限可能依赖不同的局部时间结构。
从金融解释上看，小 patch 可能更对应短期波动、局部量价异动和短期交易行为；大 patch 可能更对应中期趋势形成、波动聚集和较长期市场状态。因此，A1 说明 patch-size selection 本身就是一个值得研究的建模问题。
A2：Single-Scale vs Fixed Multi-Scale
A2 比较了 A1 中选出的最优单一 patch size 与固定多尺度组合。固定多尺度版本使用所有候选 patch size，并采用等权重聚合。
在 12 个 frequency-horizon 组合中，fixed multi-scale 相比 single-scale 的表现如下：
Metric	Fixed Multi-Scale 更优
MSE	8 / 12
Corr	6 / 12
Rank Corr	7 / 12
Direction Accuracy	5 / 12
结果说明，固定多尺度建模在部分配置下能够改善误差指标，尤其是 MSE。但这种改善并不稳定，也没有一致转化为更好的方向预测能力。Direction Accuracy 只在 5/12 个组合中改善，说明误差下降不一定意味着模型对涨跌方向或趋势判断更准确。
因此，A2 支持一个相对谨慎的结论：fixed multi-scale division 具有一定价值，但它本身不是一个稳定的完整解决方案。多尺度聚合有时能够带来增益，但这种增益依赖具体 frequency 和 horizon。
A3：Fixed Multi-Scale vs Static Learned Scale Weight
A3 比较了固定等权重多尺度聚合与静态可学习 softmax 尺度权重。这里的 static weight 是全局学习得到的，不随样本或市场状态变化。
在 12 个 frequency-horizon 组合中，static learned weight 相比 fixed multi-scale 的表现如下：
Metric	Static Weight 更优
MSE	4 / 12
Corr	3 / 12
Rank Corr	5 / 12
Direction Accuracy	8 / 12
结果显示，静态可学习尺度权重并没有稳定改善回归误差或相关性指标。它在 Direction Accuracy 上有更多改善，但这种改善没有同时反映在 MSE、Corr 或 Rank Corr 上。
这说明，学习一个全局固定的尺度偏好可能过于受限。金融时间序列中的有效时间尺度往往会随着市场状态变化而变化，因此单一全局权重向量未必能够捕捉这种动态变化。
因此，A3 并不能强有力地支持 static learned scale weighting 是 fixed multi-scale aggregation 的稳健改进。
A4：Static Learned Weight vs Adaptive Router
A4 是验证 adaptive scale selection 的核心实验。它比较了静态可学习权重模型和原生 adaptive router。adaptive router 的尺度权重是 sample-dependent、input-driven 的，即不同样本可以有不同的尺度权重。
在 12 个 frequency-horizon 组合中，adaptive router 相比 static learned weight 的表现如下：
Metric	Adaptive Router 更优
MSE	4 / 12
Corr	7 / 12
Rank Corr	5 / 12
Direction Accuracy	3 / 12
这个结果比较混合。Adaptive router 在 Corr 上有 7/12 个组合优于 static weight，但在 MSE、Rank Corr 和 Direction Accuracy 上没有形成一致优势。更重要的是，adaptive router 在部分配置中引入了严重的不稳定性。
一个典型失败案例是：
Weekly-20d adaptive router：
otest MSE = 4659.7
otest prediction standard deviation = 60.31
otest true standard deviation = 0.145
这说明模型发生了严重的预测尺度爆炸。在这个配置下，adaptive router 不只是没有提升表现，而是产生了与真实收益率尺度完全不匹配的预测值。
因此，A4 不能支持“adaptive router 在当前 FSLR 单股票场景下稳定优于 static weighting”这一结论。相反，结果显示，adaptive router 可能需要更丰富的训练样本和市场状态变化，例如多股票 panel 数据，才能学习到稳定的 input-dependent scale selection 机制。
A5：Frequency Ablation
A5 检验了使用更多输入频率是否能够改善预测表现。它比较了单频、双频和四频输入组合。
按 test MSE 看，各 horizon 的最佳组合为：
Horizon	最佳频率组合	Test MSE
5d	Hourly + Daily	1.481
10d	Single Half-Day	0.143
20d	Single Hourly	0.888
按 test Corr 看，各 horizon 的最佳组合为：
Horizon	最佳频率组合	Test Corr
5d	Single Weekly	0.051
10d	Daily + Weekly	0.095
20d	Daily + Weekly	0.199
按频率数量分组的平均 test MSE 如下：
Frequency Count	Average Test MSE
Single-frequency	91.19
Double-frequency	292.44
Four-frequency	2249.78
全频率设置尤其不稳定：
Setting	Test MSE
Full-frequency, 5d	41.0
Full-frequency, 10d	560.5
Full-frequency, 20d	6147.8
这些结果说明，增加输入频率并不会自动改善预测表现。在当前 FSLR 设定下，四频融合显著增加了模型不稳定性。极大的 MSE 数值说明，这里的问题已经不是普通预测误差，而是预测幅度发生了爆炸。
因此，A5 更应该被理解为一个稳定性诊断实验。结果显示，在 FSLR 单股票设定下，naive four-frequency fusion 并不是一个稳定的实验基座。相比全频率融合，单频或少量双频组合反而更稳定。
A1–A5 综合解读
A1–A5 可以得出四个主要结论。
第一，patch size 确实重要。A1 显示，不同频率和不同预测 horizon 的最优 patch size 不一致，说明金融时间序列中存在任务依赖的时间尺度结构。
第二，固定多尺度建模有一定帮助，但作用有限。A2 显示，fixed multi-scale 在 8/12 个组合中改善了 MSE，但这种改善没有稳定体现在相关性、排序能力和方向预测上。
第三，静态可学习尺度权重和 adaptive router 在当前 FSLR 单股票设定下尚未带来稳定增益。A3 和 A4 显示，增加尺度权重机制的灵活性并没有稳定改善表现，adaptive router 甚至在部分配置下造成严重的尺度失配和预测爆炸。
第四，全频率融合目前不稳定。A5 显示，使用 Hourly、Half-Day、Daily、Weekly 四频同时输入后，MSE 明显大于单频或双频设置，说明当前 full-frequency architecture 容易出现输出尺度失控。
总体而言，这些结果说明，时间尺度选择这个问题本身是成立的，但当前 FSLR 单股票全频率设定不足以支撑一个稳定的 adaptive multi-scale router。FSLR 实验更适合作为 full-frequency case study 和 stress test。它为后续研究提供了两个方向：一是在相对稳定的单频或双频设置上继续做组件级消融；二是将 adaptive multi-scale framework 扩展到多股票 panel 数据中，让模型从更丰富的横截面和市场状态中学习稳定的尺度选择机制。
A6 a6_dual_attention_ablation.py —— 双重注意力机制消融实验。该实验比较 full dual attention、intra-only 和 inter-only 三种设置；虽然结果展示中有 5 个 display names，但实际对应的是 3 组唯一训练配置。
A7 a7_router_interpretation.py —— 仅用于分析，不重新训练模型。该脚本读取 A4 中保存的 router-weight CSV 文件，并基于 daily window 构造市场状态变量：首先使用 rolling daily log-return standard deviation 的中位数切分高/低波动状态；其次使用线性拟合的 R² 阈值划分趋势/震荡状态。随后，按照 horizon × regime × patch size 统计平均 gate weight，用于分析 router 在不同预测期限和不同市场状态下对各类尺度的激活情况。
但是A6 现在默认建立在 full-frequency 多分支结构上，但 A5 已经证明 full-frequency 本身就是不稳基座。在坏基座上做 dual-attention ablation，解释价值很弱，容易被质疑“你比较的是坏模型内部的坏组件”。A7 也不建议现在跑。原因：A4 的 adaptive router 没有稳定优势，A5 又说明 full-frequency 路线本身不成立，这时去做 router interpretability，叙事上会很虚，因为对象本身不是有效模型。