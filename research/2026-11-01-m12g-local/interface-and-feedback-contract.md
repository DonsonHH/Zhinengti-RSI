# G1a 接口与反馈契约

状态：软件接口和真实 G0 记录适配已完成；六次合成大模型预检因 G0 门关闭而未执行；正式 Q0/Q1/Q2 未执行。

## 补丁

使用 g_timing_patch.patch_view / compile_patch / parse_patch_reply。父内容使用完整 SHA-256；稳定 r0.pN 等 ID 指原父节点，不因较早删除而漂移。每条目标编辑带原节点 expected_hash；新增节点不作为同一补丁的旧目标。1—2 项编辑按顺序作用在副本上，最终 TimingSkill 原子验证；失败不改父内容。不自动裁剪、补阈值或填分支。

replace_threshold 限于数值谓词和 evidence_margin；布尔值、kind 和 is_missing 不作为数值阈值替换。其他操作为 add_predicate、delete_predicate、add_rule、delete_rule。新谓词／分支仍受原 v1 字段、运算符、值域、0—3 OR 和1—6 AND 约束。

三个编译行为测试检查父版本、稳定定位、原子失败、缺测独立分支与越界。真实 G0 的23个 TimingSkill 策略（含H02）由该编译器生成并执行，H01为外部人工排序桥接。真实策略运行不等于大模型已掌握补丁语法。

## 决策与测量

Gv1 执行时在子策略当前公开状态上计算父准入与团队影子选择，记录第一差异。它不是父策略另一条路径上的实测结果；真正父、子轨迹另行用两个实际运行的动作日志比较。

shadow_counts.team_choice_differences 的单位是候选行，同一个团队决策时点可贡献多行；不是不同派单时点数。后处理中的 model_visible_sample 仅指为未来提示准备的样本，配套 no_model_called=True，本轮没有实际模型曝光。

公开卡保留读数／来源／年龄／未知状态、准入与实际派单、预计合法完成／Tour到达／顺延、后继合格测量及时间和报警转移。原接受位可在未来 Q1/Q2 中同权使用；隐藏漏检／等待评分、故障起止和 Oracle 不进入提示。

v1 中 selected_optional 描述早先决策。旧选择器用掉额度后不再刷新最近候选快照，因此某些卡的后继测量属于更晚的 Tour。新的 g_feedback_results.OptionalResultIndex 通过任务、执行者、实际候选完成时刻精确关联直接结果：

- direct_optional_measurement：该次 CHECK 的实际合格结果。
- actual_follow_up：这张卡的实际后继测量，可以是晚些时候的 Tour。
- follow_up_is_optional_completion：是否同一个完成事件。
- is_direct_optional_positive_card：只有直接完成且测得阳性才为真。

一次 CHECK 的结果按决策／完成身份去重，不按卡片张数计成功。220秒正常、255秒阳性的合成回归用例通过；不能把255秒值回填为220秒真值。本期实际数据中“直接正常而晚期阳性”计数为0，不把合成用例写成实测发现。

select_refined_cards 先覆盖四专业，再提供最多四张变化／直接阳性／正常／未选中反例；不足显式缺失。没有隐藏标签筛选。每卡属于明确轨迹；同输入影子、父实际运行和子实际运行的证据不混用。

## 真实适配与源码版本

本期处理647460张卡。796张关联早先 CHECK 决策，其中132张是直接完成、664张是后来 Tour；另有 H01 的4次额外检查没有 TimingSkill 卡，所以本期全部额外检查为136次，而非132次或796次。

本期实际后处理脚本按部署原字节保存在 source/derive-feedback-deployed-v1.py，SHA256为7a52b984e7ee267f55019e52abeda3942040b62e4c644e463118f1bd3bc47cdc；不改 Gv1 原始记录、动作或评分。反馈模块的部署前清单、结束哈希与现有文件另作核对。

交付版 derive_m12g_feedback.py 另补强了入口哈希捕获与结束再核对；没有回填声称旧运行已经具备该新增护栏，也未因此重复480次策略评价。未来执行须重新冻结实际使用的后处理工具身份。软件契约具备不等于差异反馈已证明提高模型收益。
