# 已执行实验总览与失败／未完成事项

本目录按现存代码、正式汇总、消费记录和历史交接整理，**不是一次重跑或全周期独立审查**。完整精度统计与来源见 [formal_results.json](evidence/formal_results.json)，逐任务CSV见 [formal_results.csv](evidence/formal_results.csv)，全部已提取政策表见 [EXPERIMENT_RESULTS.md](EXPERIMENT_RESULTS.md)，现存目录索引见 [local_inventory.json](evidence/local_inventory.json)。

证据索引覆盖31个现存results顶层目录、43份汇总源、126条政策摘要；其中含复用参照和旧五／八任务结果，**不等于126个新独立实验**。日期与采集时间以JSON为准；目录数和metadata文件数也不能相加为训练预算。已删除、未留记录或未独立核实的旧单元不伪造补齐。

## 1. 当前结论与主要结果

已有同一当前七任务口径下，r10全局预测融合macro R²为0.796，UA_2D为0.788；0.816是r10 validation，不是正式TEST。论文七项逐任务最佳值包络为0.810，不是某个模型的七任务整体分数。所有新GATE正式结果仍缺失；P2 r7训练中。

### 早期工程与正式路线

|周期／位置|实际执行内容|观测结果／预算|证据状态与后续意义|
|---|---|---|---|
|2026-09-16 cache_optimization_repair、cleanup|缓存等价性／性能、部署读取工程|parity、benchmark、recovery等保存记录|工程证据；没有据此宣称新模型SOTA|
|glt_dual_static|direct/static输入缓存构建与pilot核对|pilot1k/10k、PI1M和downstream构建记录|数据准备，不是性能实验；不重新生成|
|glt_engineering_20260918|CPU/worker/输入路径及预训练ABBA工程比较|summary记录执行完成待代码审查；正式5k/20k和full FT未执行|不能把吞吐／探针当正式模型效果|
|glt_dual_static_finetune_formal_grid|concat、kfuse，八任务五折|源macro8分别0.788、0.770；std为ddof=1|旧协议；七任务摘录另表，不和当前macro7混排为公平排名|
|glt_pred_20260918/r3、s0—s5|旧GLT-PRED多阶段准备、速度门槛及FP有／无对照|s5有80个正式单元，macro8 B_FP 0.792、B_NONE 0.794|Xc保护门未过，保留B_FP；路线退役，完整中间stage列于现存目录索引|
|MCL-PH r10—r12／repair|PH/角度/输入来源返修、旧预跑与取消队列|p0/p1/p2r2等smoke、benchmark、失败／取消现场保留|历史入口多已删除；不能仅靠目录名称认定完成或合计预算|
|MCL-PH r13-PAPER5|GLT_REF、O8_ONLY、CAT、GATE、XATTN，五任务五折|125FT；macro5依次0.837/0.836/0.829/0.829/0.837|角度键偏移与旧PH域有已知限制，非当前七任务基线；Egc/Xc缺失|
|2026-09-29 route consolidation、mcl_ph_repair、next/p0|活动路线收敛、split和direct来源修复|工程／来源语义工作，不产生新性能结论|旧SHA合同已不适用；当前不得新增哈希门槛|
|MCL-PH r4（10-01/02）|GLT与M_XATTN新5k PT；D1/D2/D3开发及七任务复评|两完整PT共10k更新，另有取消/旧smoke消费；57个新完整FT、3940epoch|七任务冻结路线macro 0.756；Eea固定M_XATTN，其余GLT；审查记录PASS，不是PH单因素收益|
|MCL-PH r5（10-04）|Xc仅改train-only目标标准化；Nc新增确认折|8个完整FT＋1个smoke；Xc正式五折|Xc R²从0.226到0.319；保留路线macro 0.770；Nc未晋级|
|MCL-PH r6（10-04）|Xc joint backbone LR降低|2个开发FT＋1个smoke，无候选新正式预测|开发门槛拒绝，七任务复用r5，不能称新增正式提升|
|GP-SOTA r2/r3|GLT＋PH多尺度专家；GP/GU及GP_A ALIGN事故|GP/GU各5k，GP_A在800中断；累计13完整FT，Nc确认5个新增单元|Nc后3折确认失败，无新正式七任务预测；GP_A事故不能当完成或当前活跃|
|SOTA-FT 10-05 A|EPS/Nc上F0/SP及single/E5开发|保存ledger为10个单元、566epoch、6226更新，含smoke|EPS无候选；Nc SP-single锁定后续；这里不是完整新SOTA评估|
|MCL-PH-OPT 10-06 R1/R2/r3|SP确认／M0对Mmix的专家几何监督组合|R2两个5k PT、14FT/980epoch/10080更新；旧A包消费另计|Eea初筛收益未通过后三折确认，EPS未过，0新增正式test；收口自检不等于独立审查|

上表早期数字来自原保存汇总或PROJECT_HISTORY；不能将旧宏8、宏5与当前宏7直接比较。旧路线保留I是预登记按任务保留的历史结果集，不是一个新统一encoder。

### GLT结构、目标与微调系列

|周期|受控变化／条件|实际范围|正式macro R²或结论|
|---|---|---|---|
|GLT-DUAL-OPT 10-06/r1|R_INCIDENT、B_FP、A_ALIGN、F_TOKEN、M_TOKEN_CE、G_SEAM_GEO|5个PT各5k＋210FT，14400epoch、210正式单元|完整逐任务见结果表；旧FP监督与现无FP目标须区分；候选均未过原validation门槛，保留I|
|GLT-FUSION 10-07/r1|B0、T_RAW、T0、T0_A、K0、K_MD、K_MD_A、TK_MD、TK_MD_A、TK_MD_L、G_NORM、G_FGR|12组完整七任务汇总；各目标/MD/融合组合|B0 0.768；T0 0.785；全部政策均完整列于结果表，未证明稳定优势|
|FT-GENERALIZATION 10-08|PARTIAL、LOW_LR、REG，相对旧B0|105FT/7200epoch/154350更新；12个PT端点事后补评|0.769、0.780、0.778；原validation门槛均未过；事后评估不改写原PT时序|
|PERIODIC-SEAM 10-09|F_RAW/F_CE/P_RAW/P_CE的有限/周期模板及CE路径2×2|4PT各5k，140逻辑FT；资源中断后累计143启动|宏0.767/0.786/0.765/0.773；原validation选择保留B0/I；F_CE 0.786是后续研究参考|
|QUALITY 10-10/r1、r2|严格几何收敛筛选，Q/Q_A/Q_G/Q_GA|原PT train911391降到9623、下游3D覆盖仅319/3655|宏0.768/0.757/0.762/0.762；分布/覆盖混杂严重；工程返修不等于全科学周期审查通过|
|FT-ROBUSTNESS 10-11|D/G/DG分路径dropout|105FT/7200epoch/154350更新|0.789/0.777/0.775；独立审查未关闭|
|S1|F/R路径dropout|70FT/4800epoch/102900更新|0.779/0.783；未达标，历史审查通过|
|S2|L_FT入射限制、B_FT入射ln2 bias|70FT/4800epoch/102900更新|0.769/0.777；审查通过来自旧用户报告，本次未重做|
|S3|M_G全图、M_L入射、M_A入射＋端点输入，同新增端点模块|3PT共15k、105FT|0.770/0.774/0.778；未关闭独立审查，M_G含新容量不能冒充原F_CE|
|U1|P_ALIGN；U0_2D、UA_2D、U0_FUSED、UA_FUSED|1个5k PT、140FT/9800epoch/224000更新|0.778/0.788/0.787/0.771；UA_2D是当前二维参考，ALIGN提升只在固定配方下观察|
|F1|rank64图级乘性交互残差，零末层|35FT/2450epoch/56000更新|0.779；不能泛化否定所有融合|
|N1|F_CE及UA_2D只取消标签缩放|70FT/4900epoch/112000更新|0.765/0.779；均低于匹配缩放条件；旧独立审查未关闭|
|S4|有限Trimer torsion输入，新增33473参数|1个两rank5k PT＋35FT|0.781；与S5差有容量及结果集中度限制|
|S5|与S4匹配两rank、无torsion参考|1个5k PT＋35FT|0.767；与旧四rankparent也有RNG混杂|
|旧架构r10/E0|独立纯二维恢复、CPU输入缓存工程|CPU 8passed/2failed，无新PT/FT|部分纯二维输出恢复可用；E0整体需返修，无提速结论|
|旧架构r10/E1|token fusion工程比较|保存CPU失败；另有36对GPU比较、72次forward/backward且0更新的PASS文件|早期consumption仍写GPU0，与后存比较材料不一致；未核实完整关闭，不删证据不宣称整周期通过|
|旧架构r10/A1|RAWCE_2D对FUSEDCE_2D的ALIGN预训练候选|实际ledger PT启动2、attempted5093/completed5092；FT0；signal15停止|比旧文档“仅初始化PASS”有更多实际消费，必须以新证据更正；无正式结果|
|FCE-UA2D path ablation p1|R0_RAW2D/R0_FUSED2D/R0_FUSED的validation-only诊断|发现19份FT指标记录，完整预算／矩阵未核实|formal_test明确FORBIDDEN，无完整正式七任务；不把它与U1或G2混称|

PERIODIC-SEAM行的macro按保存的未舍入七任务均值计算；其后的任务配方/新parent结果仍需绑定各自来源，不混称同一F_CE。所有完整逐项均值、std、RMSE、论文差保存在本次结果表和JSON中。

### F_CE/G05调优与FP融合系列

|周期|科学变化|实际新消费|正式macro R²／状态|
|---|---|---|---|
|参数r2|R0/H_LR/B_LR/S_COS，validation选择|140FT/9600epoch/205800更新；正式R0与B_LR各35|R0 0.780、B_LR 0.772；旧CPU启动2次，不能写成一次|
|T1|head dropout、backbone WD四配方每fold选|140FT；62新正式单元，另复用|TUNED 0.778，未晋级|
|T2|只将Galformer joint LR乘0.3|70FT、70正式单元|0.786；较R0约+0.006，未达+0.010|
|T3a/r4|相同两rankstep0，schedule20k对5k，均停5k|2PT/10000更新、70FT|REF 0.759、COS5K 0.764；未晋级|
|T3b/r5|相对REF仅geometry权重1→0.5|1PT/5000更新、35FT|G05 0.782；本家族后续共同parent|
|r6|G05上Galformer低LR|35FT、35新正式|0.785；未按任务拼赢家|
|r7/Xc|radius2 Tanimoto KRR＋0.5输出融合|15fit，5正式单元|仅Xc 0.430，高于论文数值0.423；没有七任务macro|
|r8/Ei|相同核政策迁移Ei|累计15fit、5正式单元，事故接续记录保留|仅Ei 0.801；没有七任务macro|
|r9|扩展七任务0.5融合，α从0.1/1/10按fold选|新增75fit，25新正式＋10复用|融合0.794；核单分支0.741是该选择政策下诊断|
|r10|固定r9模型/α，validation选一个全局w|0FT/0fit/0推理，35数组算术|w=0.75；正式0.796、validation0.816；当前主要完整系统参考|
|r11|α选择目标对齐w=0.75|8新fit／8新正式，27复用|0.796；原精度略低于r10，不用显示同分判胜|
|r12|固定α/w，Morgan radius2→3|35fit、35正式|0.795，无改善|
|r13|2×Huber(delta1)替代标准化MSE|35FT、35正式|神经0.772／融合0.787，无改善|
|r14|joint只训encoder LayerNorm affine和head|35FT、35正式|神经0.742／融合0.764，无改善|
|r15|G05匹配续训对图级teacher蒸馏|2PT各新增5k、70FT、70神经/140数组|REF与DISTILL融合均0.790；未晋级；原测试1failed，指定返修1passed，旧独立审查通过记录保留|
|r16|有限Trimer三键端点常数/真实距离，同容量34368参数|70FT、70神经/140数组|CONST神经/融合0.768/0.785，REAL0.766/0.781；待独立审查|
|FP-GRAPH f1|NULL_KFUSE、REAL_MEAN、REAL_KFUSE，三候选图级注意力|105FT、105神经正式；新增2033152参数|0.778/0.776/0.739；本周期不新做KRR后融合，待审查|

除表中明确已关闭者，后续FCE周期多为执行完成、实施后复核通过而未完成独立审查；详见原PROJECT_HISTORY的状态，不把本次资料整理变成补签PASS。r10预测融合、旧GLT架构r10和当前UA计划r7是不同命名空间。

### UA迁移、故障与当前运行

|周期|实际状态|实耗／结论|
|---|---|---|
|A预测迁移|只有部分辅助实现，完整执行未核实|没有A_GLOBAL正式结果；G2不以其为前置依赖|
|G1初次|并发PID视图归属错误后停止|8启动、0完整FT；更新已知24/21，保守32/32；图unknown按2818占用、FP586|
|G1 recovery1|同类PID问题再次停止|8启动、0完整FT；更新26/26、图768、新FP0；无正式预测／CPU suite|
|G1 recovery2|出生握手修复后运行，后遇NVML CSV/设备错误停止|27启动、19旧parent完整FT、更新30790/30790、图20549、新FP2828；无正式结果／CPU suite|
|G1旧累计|上述三次合计|43启动、19完成；更新保守30848/30848，图24135，FP3414；初次unknown仍保留；旧19不能拼入20k新parent矩阵|
|P2 r5|GPU2句柄错误，定向reset exit6；训练未spawn|新PT/FT0；保存配置/step0与失败现场|
|GPU3单卡草案|随后被最新两卡指令替代|仅代码草案，0训练消费；不列成完成实验|
|P2 r7|新20k ALIGN，GPU2/3各rank，最多2路PT|训练中；每5k保存；固定20k用于G2，当前进度看带时间证据快照|
|G2 r7|两臂×七task×五fold，自动接续|计划70FT、4900epoch、112000更新、70正式神经/140数组、至多35参照数组；FT总并发4，尚无新正式结果|

## 2. 计划与证据的冲突处理

本次实际发现两类旧文档过期：A1不止初始化，已有5092持久化PT更新；E1早期零GPU账本与后存GPU比较材料并存。已如实记录为“现有文件不一致／需补证”，没有重跑实验或反推缺失消费。当前G2的20k恢复参数已作静态修正，当前预训练保持原方案；其最终一次性CPU suite尚未运行。

科学目标不是继续堆模块。现有证据反复显示：更复杂FP图级注意力、Huber、LayerNorm-only、路径/局部/非键/蒸馏并未获得明确整体收益；这些只是否定已执行配方，不是证明机制永远无用。下一轮先审查P2/G2实际正式预测和失败记录，再提出最多一个最小受控实验，不能从已有test中挑任务/折/seed赢家。

## 3. 完整性边界

本文件覆盖已归档周期、未关闭报告和现存目录中的已执行/准备/失败项；逐fold实耗不能在来源缺失时精确重建。所有现存roots/subruns均在JSON索引列出；早期删除/无证据的实验仅能标“未核实”。既有模型权重和预测没有重新加载或评估，本次没有新增训练、核拟合、模型forward、正式预测或CPU suite。
