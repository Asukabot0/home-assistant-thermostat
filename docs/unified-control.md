# 区域闭环恒温与证据归档

## 运行结构

```mermaid
flowchart LR
    A[HA 独立测温、辅助点、环境、实时功率] --> B[来源与真实上报时间校验]
    C[Recorder 与持久功率账本] --> B
    D[拓扑及用户目标] --> E[区域状态与持续偏差]
    B --> E
    E --> F[设备分配与双区风险检查]
    F --> G[许可、保护、实时复核及串行回执]
    G --> H[伴侣及设备]
    A --> I[温度响应及覆盖电量]
    G --> I
    I --> E
    E --> J[实机状态与逐轮审计]
    G --> J
    J --> K[只读证据采集]
    C --> K
    K --> L[GitHub telemetry Release 与索引]
```

```mermaid
stateDiagram-v2
    [*] --> Observe
    Observe --> Recover: 持续偏差经独立上报证实
    Recover --> Observe: 已发令或有可信回归
    Observe --> Maintain: 回到目标附近
    Maintain --> Observe: 新的持续偏差
    Maintain --> Weaken: 预测或实测过冲
    Recover --> Weaken: 风险或需退出相反方向
    Weaken --> Observe: 等待温度或证实减载
    Weaken --> Parked: 无需该方向且通过停机门禁
    Parked --> Recover: 需求及停机证据通过
    Observe --> Limited: 暂停、证据不足、冲突或失败锁
    Recover --> Limited: 暂停、证据不足、冲突或失败锁
    Limited --> Observe: 条件恢复并重新取证
```

执行账本单独保存 `prepared → accepted → proxy_echo`。不明确的传输结果进入失败锁。确认开机后按当前目标处理 `initial`；目标、观测定义或需求变化时丢弃未发送意图，必要时仅做安全归位。关闭后在 `parked` 保留模式停机证据，下一次恢复重新计算。回显代表伴侣确认，温度与功率作为单独的观测证据。

## 区域与设备配置

从 `home-assistant/shadow/topology.example.json` 建立本地 `topology.json`。`rooms.json` 继续绑定已有设备、硬件软限及调试见证。`sensors` 支持 `primary`、`auxiliary`、`constraint` 和 `environment`，显式禁用的点无需存在。每个区域具有独立主点，`primary` 直接使用主点，`weighted` 按配置把新鲜辅助点加入融合，独立证据仍采用主点的上报时间。`constraint` 独立保护危险边界，环境点使用配置的温度属性。

`zones.target` 可映射某台虚拟恒温器、指定 HA 数值实体及可选属性，或设静态目标。`pause_entity` 可接绑定的 `input_boolean`。许可的 `paused_zones` 暂停一个区域，`isolated_rooms` 和伴侣手动状态限制对应设备。暂停区域仍保留温度约束。`actuators.influences` 显式描述设备可作用的多个区域。`connections` 记录区域耦合，`opposite_operation.enabled` 默认关闭。开启后还要求明确分区、低跨区影响、真实输出证据以及对侧匹配需求。新增室内主点后建立新区域，修改设备影响关系并提升 `observation_version`。变更定义、实际来源或目标会废弃受影响的暂态证据；设备基线及发令历史照常继承。

## 策略与保护

初始跟踪带为 ±0.3℃。区域偏差、持续时间和独立上报共同资格化，轻重偏差保持同一段记录。偏差持续至少 20 分钟并有跨 10 分钟的独立报告时可以恢复，可信趋势已在回归时暂缓。加强后使用本区新温度和阶段等待，邻区以新鲜值和风险边界约束，单台暂停不会消除该区域。设备短运行、停机、反向保护和数据新鲜度独立记录。停机重启继续要求连续低耗的实测报告，报告断流仍以 120 秒为限。

维持阶段保留有效设温。稳态温度和真实活动时段共同资格化维持 setpoint，恢复先用当前目标或同观测定义下的可靠维持值。修正受硬件范围、模式软限及步长限制。对无方向需求的有效反向负荷先减载，满足停机保护后允许让已经验证低荷的相反模式退出。两区域都需要相反方向时先按区域风险、偏差和等待排序；无法满足的需求保留在审计中。总电量依据真实报告之间的有覆盖积分。排序只有在至少两种方案各积累足够可比、成功的响应与电量样本后才比较 Wh/温度改善；覆盖不足继续依照区域与作用范围分配。

`zone_control.py` 产生状态、候选和区域作用范围。`continuous_pilot.py` 把同一决策写入状态输出和逐轮日志，用第二次 HA 观测复核后再串行执行。`live_evidence.py` 合并真上报与同源同代账本，避免分钟状态等待遮断实时功率。`service.py` 保留历史功率模型和 SQLite 分钟观测，可用 `shadow-health.json` 和状态 `published_at`／`stale_after` 查看更新时间。

## 部署与回退

运行全量单测、`work/verify_*.py` 和历史逻辑回放。使用 `work/release_local.py --production <home-assistant> --commit <完整远端 main SHA> --topology <私有配置>`。脚本备份旧源码、有效配置、Compose 和状态证据。在途指令结算后停服，按 Git 对象部署，启用 32 MiB `/tmp`，恢复旧运行时可用同脚本的 `--rollback <备份路径>`。回退只处理源码、配置和 Compose，运行中新产生的账本、回执和 SQLite 历史继续使用。

## 每小时证据

`work/install_telemetry.py` 在 macOS 安装 `local.codex.home-assistant.thermostat-telemetry`。每小时第 5 分钟运行，时间窗用 UTC 计算并映射北京时间。安装后从同一入口立即上传状态包。该用户服务随 Docker Desktop 登录环境运行，睡眠或关机漏跑后以游标补齐。任务状态和日志位于 `~/Library/Application Support/HomeAssistantThermostat/telemetry/`。停止可执行 `launchctl bootout gui/$(id -u)/local.codex.home-assistant.thermostat-telemetry`，并用 `launchctl disable` 持久停用。

采集器只读，独立 SQLite 查询事务包含已提交的 WAL，跨库分别记录采集时刻和水位。游标在下载验证 ZIP、逐文件 SHA 和远端索引完成后推进。失败在队列留下包和窗口，使用互斥锁、限时及有限退避。最近已上传窗口重新对账接收晚到的数据；修订作为不同附件链接。状态包与小时包的范围在 manifest 明列。只在确认上传后清理本地任务副本。GitHub 用 `telemetry-` 日期预发布件及 `telemetry-index.json` 索引，遇不可变 Release 使用独立日期窗口附件。认证由安装用户已有 GitHub CLI 凭据提供，白名单及本地 HMAC 对来源、设备和区域匿名化。首包及小时包均不以短窗口推断长期节电或恒温改善。
