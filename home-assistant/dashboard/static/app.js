(() => {
  "use strict";

  const HEALTH_META = {
    healthy: { label: "运行正常", cls: "badge-ok" },
    paused: { label: "控制已暂停", cls: "badge-warn" },
    stale: { label: "数据过期", cls: "badge-warn" },
    fault_locked: { label: "存在故障锁", cls: "badge-danger" },
    offline: { label: "未检测到数据", cls: "badge-danger" },
  };

  const STAGE_BADGE = {
    in_flight: "badge-ok",
    selected: "badge-ok",
    queued_transaction: "badge-accent",
    gate_blocked: "badge-warn",
    handoff_waiting: "badge-warn",
    projected: "badge-purple",
    idle_maintain: "badge-muted",
    paused_or_locked: "badge-danger",
  };

  const STATE_INFO = {
    observe: {
      title: "Observe · 观测",
      color: "#38bdf8",
      desc: "正在采集并累计独立测温上报；偏差需持续 ≥20 分钟且具备跨 10 分钟独立上报才会转入 Recover；回归带内后需 10 分钟稳定确认才转入 Maintain。",
    },
    recover: {
      title: "Recover · 恢复",
      color: "#fb923c",
      desc: "持续偏差已由独立测温证实且温度未在可信回归；作为主区 (owner) 可在动作后观察窗与门禁通过后执行按需启机或小步加码。",
    },
    maintain: {
      title: "Maintain · 维持",
      color: "#34d399",
      desc: "室温稳定处于目标 ±0.3℃ 跟踪带内；保持当前有效设温并在满足稳态时长与负载证据后学习维持设温。",
    },
    weaken: {
      title: "Weaken · 减载",
      color: "#fbbf24",
      desc: "预测过冲、持续高功率越过理想线、邻区出现舒适风险或存在无需求反向负荷；按 3 分钟反馈周期逐步减载或满足门禁后退出。",
    },
    parked: {
      title: "Parked · 停机驻留",
      color: "#a78bfa",
      desc: "空调经由受控停机退出后保留模式与停机时间证据；重新启机需满足 ≥5 分钟停机与连续低耗实测，冷热换向还受 60 分钟保护。",
    },
    limited: {
      title: "Limited · 受限",
      color: "#f87171",
      desc: "区域被暂停、测温缺失/过期、双区双向冲突让权、影响本区的设备存在未确认指令锁，或暂无可避免邻区越界的安全设备。",
    },
  };

  const POLICY_LABELS = {
    tracking_band: ["目标跟踪带半宽", "℃"],
    ideal_band: ["理想带半宽", "℃"],
    comfort_band: ["舒适带半宽", "℃"],
    ordinary_step: ["常规单步调温", "℃"],
    large_step: ["大偏差单步上限", "℃"],
    large_deviation: ["大偏差阈值", "℃"],
    tracking_dwell_seconds: ["持续偏差资格时长", "s"],
    tracking_report_span_seconds: ["独立上报跨度要求", "s"],
    recovery_settle_seconds: ["回归稳定确认跨度", "s"],
    feedback_wait_seconds: ["加码动作后反馈等待", "s"],
    reduction_wait_seconds: ["减载后反馈等待", "s"],
    min_run_seconds: ["设备最短运行时间", "s"],
    min_off_seconds: ["停机最短等待 / 低耗证明", "s"],
    reversal_run_seconds: ["冷热换向保护间隔", "s"],
    lower_output_observe_seconds: ["交接低耗连续观察", "s"],
    peer_fairness_wait_seconds: ["双区冲突公平轮换窗", "s"],
    maintain_confirmation_seconds: ["维持设温学习确认时长", "s"],
  };

  const state = {
    snapshot: null,
    receivedAtMs: 0,
    scenario: null,
    selectedNode: null,
    auditFilter: "all",
    activeTab: "commands",
    eventSource: null,
    pollTimer: null,
  };

  function escapeHtml(val) {
    return String(val ?? "")
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#39;");
  }

  function el(tag, attrs = {}, children = []) {
    const node = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (v === null || v === undefined || v === false) continue;
      if (k === "class") node.className = v;
      else if (k === "text") node.textContent = String(v);
      else if (k.startsWith("data-")) node.setAttribute(k, String(v));
      else if (k === "title") node.title = String(v);
      else if (k === "type") node.type = String(v);
      else node.setAttribute(k, String(v));
    }
    const list = Array.isArray(children) ? children : [children];
    for (const child of list) {
      if (child === null || child === undefined || child === false) continue;
      if (typeof child === "string" || typeof child === "number") {
        node.appendChild(document.createTextNode(String(child)));
      } else {
        node.appendChild(child);
      }
    }
    return node;
  }

  function fmtDur(seconds) {
    if (seconds === null || seconds === undefined || Number.isNaN(seconds)) return "—";
    const s = Math.max(0, Math.round(seconds));
    if (s < 60) return `${s}秒`;
    const m = Math.floor(s / 60);
    const rem = s % 60;
    if (m < 60) return `${m}分${String(rem).padStart(2, "0")}秒`;
    const h = Math.floor(m / 60);
    const rm = m % 60;
    return `${h}小时${rm}分`;
  }

  function fmtClock(ts) {
    if (!ts) return "—";
    const d = new Date(ts * 1000);
    if (Number.isNaN(d.getTime())) return "—";
    return d.toLocaleTimeString("zh-CN", { hour12: false });
  }

  function fmtDateTime(ts) {
    if (!ts) return "—";
    const d = new Date(ts * 1000);
    if (Number.isNaN(d.getTime())) return "—";
    return d.toLocaleString("zh-CN", {
      month: "2-digit",
      day: "2-digit",
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
      hour12: false,
    });
  }

  function fmtTemp(v, digits = 2) {
    if (v === null || v === undefined || Number.isNaN(v)) return "—";
    return `${Number(v).toFixed(digits)}℃`;
  }

  function fmtSignedTemp(v) {
    if (v === null || v === undefined || Number.isNaN(v)) return "—";
    const n = Number(v);
    return `${n >= 0 ? "+" : ""}${n.toFixed(2)}℃`;
  }

  function fmtWatts(v) {
    if (v === null || v === undefined || Number.isNaN(v)) return "—";
    return `${Math.round(Number(v))} W`;
  }

  function modeBadgeClass(mode) {
    if (mode === "heat") return "badge-heat";
    if (mode === "cool") return "badge-cool";
    if (mode === "conflict") return "badge-danger";
    if (mode === "hold") return "badge-ok";
    return "badge-muted";
  }

  function intentBadgeClass(intent) {
    return (
      {
        observe: "badge-accent",
        recover: "badge-heat",
        maintain: "badge-ok",
        weaken: "badge-warn",
        parked: "badge-purple",
        limited: "badge-danger",
      }[intent] || "badge-muted"
    );
  }

  function statusBadgeClass(status) {
    return (
      {
        proxy_echo: "badge-ok",
        accepted: "badge-accent",
        prepared: "badge-purple",
        unconfirmed: "badge-danger",
        rejected: "badge-danger",
        canceled: "badge-muted",
      }[status] || "badge-muted"
    );
  }

  function currentCountdown(baseSeconds) {
    if (baseSeconds === null || baseSeconds === undefined) return null;
    const elapsed = (Date.now() - state.receivedAtMs) / 1000;
    return Math.max(0, baseSeconds - elapsed);
  }

  // ---------- Rendering ----------

  function renderDemoBanner(snap) {
    const banner = document.getElementById("demo-banner");
    const pills = document.getElementById("scenario-pills");
    if (snap.mode !== "demo" || !Array.isArray(snap.available_scenarios) || !snap.available_scenarios.length) {
      banner.classList.add("hidden");
      return;
    }
    banner.classList.remove("hidden");
    pills.replaceChildren();
    for (const sc of snap.available_scenarios) {
      const btn = el("button", {
        type: "button",
        class: `scenario-btn ${snap.scenario === sc.id ? "active" : ""}`,
        title: sc.subtitle,
        text: sc.title,
      });
      btn.addEventListener("click", () => {
        if (state.scenario === sc.id) return;
        state.scenario = sc.id;
        connectStream();
        fetchSnapshot();
      });
      pills.appendChild(btn);
    }
  }

  function renderTopbar(snap) {
    const ctrl = snap.controller;
    const dot = document.getElementById("health-dot");
    dot.className = `brand-dot ${ctrl.health || "offline"}`;

    document.getElementById("controller-status-text").textContent =
      `${ctrl.status_text || "—"} · ${ctrl.status_age_s !== null ? `状态更新于 ${fmtDur(ctrl.status_age_s)}前` : "等待状态文件"}`;

    const hMeta = HEALTH_META[ctrl.health] || HEALTH_META.offline;
    const kpiHealth = document.getElementById("kpi-health");
    kpiHealth.replaceChildren(el("span", { class: `badge ${hMeta.cls}`, text: hMeta.label }));

    const kpiNeed = document.getElementById("kpi-need");
    kpiNeed.replaceChildren(
      el("span", {
        class: `badge ${modeBadgeClass(ctrl.conflict ? "conflict" : ctrl.house_need)}`,
        text: ctrl.conflict ? "双向冲突" : (ctrl.house_need_label || "未知"),
      })
    );

    document.getElementById("kpi-owner").textContent = ctrl.owner ? `${ctrl.owner}${ctrl.lead ? ` / ${ctrl.lead}` : ""}` : "无（维持）";

    const envEntries = Object.values(ctrl.environment || {});
    const freshEnv = envEntries.find((e) => e && e.value !== null && e.quality === "fresh") || envEntries[0];
    document.getElementById("kpi-outdoor").textContent = freshEnv && freshEnv.value !== null ? fmtTemp(freshEnv.value, 1) : "—";

    const kpiNext = document.getElementById("kpi-next-cycle");
    if (ctrl.next_cycle_in_s !== null && ctrl.next_cycle_in_s !== undefined) {
      kpiNext.setAttribute("data-countdown", String(ctrl.next_cycle_in_s));
      kpiNext.setAttribute("data-countdown-loop", String(ctrl.cycle_interval_s || 20));
      kpiNext.textContent = `${Math.ceil(currentCountdown(ctrl.next_cycle_in_s))}s`;
    } else {
      kpiNext.removeAttribute("data-countdown");
      kpiNext.textContent = ctrl.stale ? "已过期" : "—";
    }

    const modeBadge = document.getElementById("ctrl-mode-badge");
    modeBadge.textContent =
      ctrl.mode === "unified_topology"
        ? `统一区域闭环 · ${ctrl.release?.observation_version || "topology"}`
        : ctrl.mode === "legacy_gentle"
        ? "传统全屋 / 温和模式"
        : "未识别模式";

    // Alert banner for read errors / offline / stale / fault lock
    const notice = document.getElementById("offline-notice");
    const locks = snap.state_machines?.ledger?.locks || {};
    const holdKeys = Object.keys(locks.holds || {});
    const readErrors = ctrl.read_errors || [];
    if (readErrors.length > 0) {
      notice.className = "alert-banner danger";
      const list = readErrors.map((e) => `${e.file} (${e.error})`).join("、");
      notice.textContent = `部分控制器运行文件读取失败或权限不足：${list}。请检查只读挂载与容器权限（compose.dashboard.yaml 已配置 DAC_READ_SEARCH 以读取 0600 账本与配置）。`;
    } else if (!ctrl.data_present) {
      notice.className = "alert-banner";
      notice.textContent = "未在挂载目录检测到实机控制器输出（pilot-data/status.json）。若要在本地预览各状态机效果，可使用 `--demo` 参数启动演示模式。";
    } else if (holdKeys.length > 0) {
      notice.className = "alert-banner danger";
      const msgs = holdKeys.map((k) => `${k}: ${locks.holds[k]}`).join("；");
      notice.textContent = `检测到控制器故障/回报锁：${msgs}。在人工核对并解除锁定前，受影响设备组已暂停发令。`;
    } else if (ctrl.stale) {
      notice.className = "alert-banner";
      notice.textContent = `控制器状态已超过 90 秒未刷新（${ctrl.status_age_s !== null ? fmtDur(ctrl.status_age_s) : "未知"}前），请检查 heating-pilot 容器是否正常运行。`;
    } else if (!ctrl.permit?.active) {
      notice.className = "alert-banner";
      notice.textContent = `控制许可未生效：${ctrl.permit?.reason || "已暂停"}。控制器当前不会向设备下发任何指令。`;
    } else {
      notice.className = "alert-banner hidden";
    }
  }

  function renderNextCommand(snap) {
    const nc = snap.next_command;
    const card = document.getElementById("next-command-card");
    card.setAttribute("data-stage", nc.stage || "idle_maintain");

    const pill = document.getElementById("next-stage-badge");
    pill.className = `stage-pill ${STAGE_BADGE[nc.stage] || "badge-muted"}`;
    pill.textContent = nc.stage_label || nc.stage;

    const body = document.getElementById("next-command-body");
    body.replaceChildren();

    const leftBox = el("div", { class: "next-action-main" }, [
      el("div", { class: "next-headline", text: nc.headline || "—" }),
      el("div", { class: "next-summary", text: nc.summary || "" }),
    ]);

    if (nc.action) {
      const metaRow = el("div", { style: "margin-top:8px; display:flex; flex-wrap:wrap; gap:6px; align-items:center;" }, [
        el("span", { class: "badge badge-accent", text: nc.action.purpose_label }),
        nc.action.zone ? el("span", { class: "badge badge-muted", text: `目标区域: ${nc.action.zone}` }) : null,
        nc.waiting_for_new_report ? el("span", { class: "badge badge-warn", text: "需收到新的独立测温上报" }) : null,
      ]);
      leftBox.appendChild(metaRow);
      leftBox.appendChild(el("code", { class: "next-service-call mono", text: nc.action.service_call }));
    } else if (nc.waiting_for_new_report) {
      leftBox.appendChild(
        el("div", { style: "margin-top:8px;" }, [
          el("span", { class: "badge badge-warn", text: "需收到新的独立测温上报" }),
        ])
      );
    }

    const etaBox = el("div", { class: "eta-box" });
    etaBox.appendChild(el("div", { class: "eta-label", text: nc.stage === "in_flight" ? "回报超时倒计时" : "预计时间 (ETA)" }));
    const etaTime = el("div", { class: "eta-time mono" });
    if (nc.eta_seconds !== null && nc.eta_seconds !== undefined) {
      etaTime.setAttribute("data-countdown", String(nc.eta_seconds));
      etaTime.textContent = fmtDur(currentCountdown(nc.eta_seconds));
    } else {
      etaTime.textContent = nc.stage === "paused_or_locked" ? "已暂停" : "待定";
    }
    etaBox.appendChild(etaTime);
    etaBox.appendChild(el("div", { class: "eta-note", text: nc.eta_note || "" }));

    body.appendChild(el("div", { class: "next-action-box" }, [leftBox, etaBox]));

    // Blockers / prerequisites checklist
    if (Array.isArray(nc.blockers) && nc.blockers.length > 0) {
      const list = el("div", { class: "blocker-list" });
      for (const b of nc.blockers) {
        const rightTag = b.met
          ? el("span", { class: "badge badge-ok", text: "已满足" })
          : b.remaining_s !== null && b.remaining_s !== undefined
          ? el("span", {
              class: "badge badge-warn mono",
              "data-countdown": String(b.remaining_s),
              text: `剩 ${fmtDur(currentCountdown(b.remaining_s))}`,
            })
          : el("span", { class: "badge badge-danger", text: "未满足" });

        const item = el("div", { class: "blocker-item" }, [
          el("div", { class: "blocker-top" }, [
            el("strong", { text: b.label }),
            rightTag,
          ]),
          el("div", { class: "blocker-detail", text: b.detail }),
        ]);

        if (b.progress_pct !== null && b.progress_pct !== undefined) {
          const fillCls = b.met ? "met" : b.remaining_s !== null ? "warn" : "danger";
          item.appendChild(
            el("div", { class: "progress-track" }, [
              el("div", { class: `progress-fill ${fillCls}`, style: `width:${Math.max(0, Math.min(100, b.progress_pct))}%` }),
            ])
          );
        }
        list.appendChild(item);
      }
      body.appendChild(list);
    }

    // Candidates / Rejections from latest cycle
    if (Array.isArray(nc.candidates) && nc.candidates.length > 0) {
      const cbox = el("div", { class: "candidates-box" }, [
        el("div", { class: "candidates-title", text: `本轮候选队列（共 ${nc.candidates.length} 条）` }),
      ]);
      for (const c of nc.candidates) {
        cbox.appendChild(
          el("div", { class: "candidate-row" }, [
            el("div", {}, [
              el("strong", { text: c.display }),
              el("span", { class: "badge badge-muted", style: "margin-left:6px;", text: c.purpose_label }),
            ]),
            c.rejected
              ? el("span", { class: "badge badge-warn", text: `门禁拦截: ${c.rejection_reason}` })
              : el("span", { class: "badge badge-ok", text: "门禁通过" }),
          ])
        );
      }
      body.appendChild(cbox);
    }
  }

  function renderZoneStateMachineSvg(snap) {
    const wrap = document.getElementById("zone-sm-svg-wrap");
    const zones = snap.state_machines?.zone_states || [];
    const parked = snap.state_machines?.ledger?.parked || [];

    const byState = { observe: [], recover: [], maintain: [], weaken: [], parked: [], limited: [] };
    for (const z of zones) {
      if (byState[z.state]) byState[z.state].push(z.zone);
    }
    for (const p of parked) {
      byState.parked.push(p.room);
    }

    const nodes = [
      { id: "observe", x: 22, y: 24, w: 176, h: 82, title: "Observe", sub: "观测 / 累计证据" },
      { id: "recover", x: 252, y: 24, w: 176, h: 82, title: "Recover", sub: "持续偏差恢复" },
      { id: "maintain", x: 482, y: 24, w: 176, h: 82, title: "Maintain", sub: "±0.3℃ 稳定维持" },
      { id: "limited", x: 22, y: 158, w: 176, h: 82, title: "Limited", sub: "暂停 / 冲突 / 锁" },
      { id: "weaken", x: 252, y: 158, w: 176, h: 82, title: "Weaken", sub: "过冲 / 风险减载" },
      { id: "parked", x: 482, y: 158, w: 176, h: 82, title: "Parked", sub: "停机驻留保护" },
    ];

    const edges = [
      { d: "M 198 54 L 250 54", label: "偏差证实", lx: 224, ly: 46 },
      { d: "M 250 76 L 198 76", label: "发令/回归", lx: 224, ly: 92 },
      { d: "M 110 24 Q 340 -8 566 22", label: "稳定回归", lx: 340, ly: 13 },
      { d: "M 566 106 L 566 130 Q 566 138 552 138 L 430 138 Q 370 138 370 156", label: "过冲", lx: 468, ly: 133 },
      { d: "M 340 106 L 340 156", label: "风险/退反向", lx: 382, ly: 133 },
      { d: "M 252 184 L 222 184 Q 204 184 204 166 L 204 128 Q 204 114 184 114 L 160 108", label: "", lx: 0, ly: 0 },
      { d: "M 428 199 L 480 199", label: "停机门禁", lx: 454, ly: 190 },
      { d: "M 530 158 Q 470 118 428 84", label: "低耗启机", lx: 468, ly: 112 },
      { d: "M 110 106 L 110 156", label: "锁/暂停", lx: 78, ly: 135 },
      { d: "M 134 156 L 134 106", label: "恢复", lx: 154, ly: 135 },
    ];

    let svg = `<svg class="sm-svg" viewBox="0 -10 680 268" role="img" aria-label="区域闭环状态机图">
      <defs>
        <marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
          <path d="M 0 1 L 10 5 L 0 9 z" fill="var(--svg-edge)" />
        </marker>
      </defs>`;

    for (const e of edges) {
      svg += `<path d="${e.d}" fill="none" stroke="var(--svg-edge)" stroke-width="1.5" marker-end="url(#arrow)" opacity="0.75" />`;
      if (e.label) {
        svg += `<text x="${e.lx}" y="${e.ly}" text-anchor="middle" font-size="10" fill="var(--text-dim)">${escapeHtml(e.label)}</text>`;
      }
    }

    for (const n of nodes) {
      const occupants = byState[n.id] || [];
      const active = occupants.length > 0;
      const selected = state.selectedNode === n.id;
      const info = STATE_INFO[n.id];
      const stroke = active || selected ? info.color : "var(--svg-node-stroke)";
      const strokeW = active ? 2.5 : selected ? 2 : 1.3;
      const fill = active ? `${info.color}22` : "var(--svg-node-bg)";

      svg += `<g class="sm-node" data-node="${n.id}">
        <rect x="${n.x}" y="${n.y}" width="${n.w}" height="${n.h}" rx="11" fill="${fill}" stroke="${stroke}" stroke-width="${strokeW}" />
        <circle cx="${n.x + 16}" cy="${n.y + 21}" r="5" fill="${active ? info.color : "var(--text-dim)"}" />
        <text x="${n.x + 28}" y="${n.y + 25}" font-size="13.5" font-weight="700" fill="var(--text)">${escapeHtml(n.title)}</text>
        <text x="${n.x + 14}" y="${n.y + 44}" font-size="11" fill="var(--text-muted)">${escapeHtml(n.sub)}</text>`;

      if (active) {
        let px = n.x + 12;
        const py = n.y + 54;
        for (const tag of occupants.slice(0, 3)) {
          const tw = Math.max(46, tag.length * 7.5 + 16);
          svg += `<rect x="${px}" y="${py}" width="${tw}" height="19" rx="9" fill="${info.color}" />
            <text x="${px + tw / 2}" y="${py + 13}" text-anchor="middle" font-size="10.5" font-weight="700" fill="#08101c">${escapeHtml(tag)}</text>`;
          px += tw + 5;
        }
      } else {
        svg += `<text x="${n.x + 14}" y="${n.y + 68}" font-size="10.5" fill="var(--text-dim)">空闲</text>`;
      }
      svg += `</g>`;
    }

    svg += `</svg>`;
    wrap.innerHTML = svg;

    wrap.querySelectorAll(".sm-node").forEach((g) => {
      g.addEventListener("click", () => {
        state.selectedNode = g.getAttribute("data-node");
        renderZoneStateMachineSvg(state.snapshot);
        renderZoneInspector(state.snapshot);
      });
    });

    renderZoneInspector(snap);
  }

  function renderZoneInspector(snap) {
    const box = document.getElementById("zone-sm-inspector");
    const nodeKey = state.selectedNode;
    if (!nodeKey || !STATE_INFO[nodeKey]) {
      const zones = snap.state_machines?.zone_states || [];
      const summary = zones.length
        ? zones.map((z) => `${z.label}: ${z.state_label}（${z.reason_label}）`).join(" ｜ ")
        : "点击上方任意状态节点，可查看该状态的进入与退出条件。";
      box.textContent = summary;
      return;
    }
    const info = STATE_INFO[nodeKey];
    const matched = (snap.state_machines?.zone_states || []).filter((z) => z.state === nodeKey);
    const activeTxt = matched.length ? `当前命中区域：${matched.map((z) => z.label).join("、")}。` : "当前无区域处于此状态。";
    box.replaceChildren(
      el("strong", { style: `color:${info.color}`, text: info.title }),
      el("span", { text: ` — ${info.desc} ${activeTxt}` })
    );
  }

  function renderHandoffStateMachine(snap) {
    const ho = snap.state_machines?.handoff || { active: false, phase: "idle", phase_label: "空闲" };
    const badge = document.getElementById("handoff-phase-badge");
    badge.className = `badge ${ho.active ? (ho.phase === "limited" ? "badge-danger" : "badge-warn") : "badge-muted"}`;
    badge.textContent = ho.phase_label || ho.phase;

    const wrap = document.getElementById("handoff-sm-svg-wrap");
    const steps = [
      { id: "elect", x: 14, y: 20, w: 94, h: 52, t: "Elect", s: "仲裁选主" },
      { id: "reduce", x: 126, y: 20, w: 94, h: 52, t: "Reduce", s: "对侧减载" },
      { id: "verify_reduction", x: 238, y: 20, w: 108, h: 52, t: "VerifyRed", s: "核验减载" },
      { id: "exit", x: 364, y: 20, w: 86, h: 52, t: "Exit", s: "对侧停机" },
      { id: "verify_exit", x: 468, y: 20, w: 102, h: 52, t: "VerifyExit", s: "核验低耗" },
      { id: "serve", x: 588, y: 20, w: 78, h: 52, t: "Serve", s: "服务主区" },
    ];

    let svg = `<svg class="sm-svg" viewBox="0 -12 680 156" role="img" aria-label="双向需求交接状态机">
      <defs>
        <marker id="harrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
          <path d="M 0 1 L 10 5 L 0 9 z" fill="var(--svg-edge)" />
        </marker>
      </defs>`;

    for (let i = 0; i < steps.length - 1; i++) {
      const a = steps[i];
      const b = steps[i + 1];
      svg += `<line x1="${a.x + a.w}" y1="46" x2="${b.x - 2}" y2="46" stroke="var(--svg-edge)" stroke-width="1.5" marker-end="url(#harrow)" />`;
    }
    // Branch Elect -> Limited
    svg += `<line x1="61" y1="72" x2="61" y2="96" stroke="var(--svg-edge)" stroke-width="1.5" marker-end="url(#harrow)" />`;
    // Shortcut Elect -> Exit (proven dormant)
    svg += `<path d="M 85 20 Q 246 -4 404 18" fill="none" stroke="var(--svg-edge)" stroke-width="1.2" stroke-dasharray="4 3" marker-end="url(#harrow)" />
      <text x="246" y="11" text-anchor="middle" font-size="9.5" fill="var(--text-dim)">已证实低耗或达软限边界</text>`;

    for (const s of steps) {
      const isCur = ho.active && ho.phase === s.id;
      const color = s.id === "serve" ? "#34d399" : "#fbbf24";
      svg += `<g>
        <rect x="${s.x}" y="${s.y}" width="${s.w}" height="${s.h}" rx="9"
          fill="${isCur ? `${color}26` : "var(--svg-node-bg)"}"
          stroke="${isCur ? color : "var(--svg-node-stroke)"}"
          stroke-width="${isCur ? 2.4 : 1.2}" />
        <text x="${s.x + s.w / 2}" y="${s.y + 22}" text-anchor="middle" font-size="11.5" font-weight="700" fill="var(--text)">${s.t}</text>
        <text x="${s.x + s.w / 2}" y="${s.y + 39}" text-anchor="middle" font-size="10" fill="var(--text-muted)">${s.s}</text>
      </g>`;
    }

    const limCur = ho.active && ho.phase === "limited";
    svg += `<g>
      <rect x="14" y="98" width="168" height="36" rx="9"
        fill="${limCur ? "rgba(248,113,113,0.2)" : "var(--svg-node-bg)"}"
        stroke="${limCur ? "#f87171" : "var(--svg-node-stroke)"}"
        stroke-width="${limCur ? 2.4 : 1.2}" />
      <text x="98" y="120" text-anchor="middle" font-size="11" font-weight="700" fill="var(--text)">Limited · 交接受限/余量不足</text>
    </g></svg>`;
    wrap.innerHTML = svg;

    const detail = document.getElementById("handoff-detail");
    detail.replaceChildren();
    if (!ho.active) {
      detail.appendChild(el("span", { style: "color:var(--text-muted)", text: "当前双区无冷热反向冲突，双向交接状态机处于空闲待命状态。" }));
      return;
    }
    detail.appendChild(el("span", { class: "badge badge-warn", text: `主区: ${ho.owner || "—"}` }));
    detail.appendChild(el("span", { class: "badge badge-accent", text: `接收机: ${ho.receiver || "—"}` }));
    detail.appendChild(el("span", { class: "badge badge-danger", text: `待退出对侧机: ${ho.blocker || "—"}` }));
    if (ho.elapsed_s !== null && ho.elapsed_s !== undefined) {
      detail.appendChild(el("span", { class: "mono", text: `交接已历时 ${fmtDur(ho.elapsed_s)}` }));
    }
    if (ho.reason_label) {
      detail.appendChild(el("span", { style: "color:var(--text-muted)", text: `原因：${ho.reason_label}` }));
    }
  }

  function renderLedgerAndLocks(snap) {
    const ledger = snap.state_machines?.ledger || {};
    const badge = document.getElementById("ledger-phase-badge");
    const phaseMap = {
      idle: ["空闲", "badge-muted"],
      prepared: ["已落盘待发", "badge-purple"],
      accepted: ["在途待回报", "badge-accent"],
      proxy_echo: ["回执正常", "badge-ok"],
      unconfirmed: ["触发故障锁", "badge-danger"],
    };
    const [pLabel, pCls] = phaseMap[ledger.active_phase] || ["—", "badge-muted"];
    badge.className = `badge ${pCls}`;
    badge.textContent = pLabel;

    const pipe = document.getElementById("ledger-pipeline");
    pipe.replaceChildren();
    const pipeDefs = [
      { key: "prepared", title: "Prepared", sub: "指令落盘" },
      { key: "accepted", title: "Accepted", sub: `在途 ${ledger.pending_count || 0} 条` },
      { key: "proxy_echo", title: "ProxyEcho", sub: `24h 回报 ${ledger.echoed_24h_count || 0} 条` },
      { key: "unconfirmed", title: "Unconfirmed", sub: `异常锁 ${Object.keys(ledger.locks?.holds || {}).length} 项` },
    ];
    for (const step of pipeDefs) {
      let cls = "pipe-step";
      if (ledger.active_phase === step.key) {
        cls += step.key === "proxy_echo" ? " active-ok" : step.key === "unconfirmed" ? " active-danger" : " active";
      }
      pipe.appendChild(
        el("div", { class: cls }, [
          el("div", { class: "pipe-name", text: step.title }),
          el("div", { class: "pipe-sub", text: step.sub }),
        ])
      );
    }

    const body = document.getElementById("locks-and-tx-body");
    body.replaceChildren();

    // Locks panel
    const locks = ledger.locks || {};
    const holds = Object.entries(locks.holds || {});
    const manuals = Object.entries(locks.manual_holds || {});
    const isolated = locks.isolated_rooms || [];
    const pausedZones = locks.paused_zones || [];
    const lockPanel = el("div", { class: "mini-panel" }, [
      el("div", { class: "mini-panel-title" }, [
        el("span", { text: "保护锁与暂停状态" }),
        holds.length || manuals.length
          ? el("span", { class: "badge badge-danger", text: `${holds.length + manuals.length} 项锁定` })
          : el("span", { class: "badge badge-ok", text: "无异常锁定" }),
      ]),
    ]);
    if (!holds.length && !manuals.length && !isolated.length && !pausedZones.length) {
      lockPanel.appendChild(el("div", { style: "color:var(--text-muted)", text: "所有可控设备组运行正常，无故障锁、无外部手动接管、无手动隔离。" }));
    } else {
      const tags = el("div", { style: "display:flex; flex-direction:column; gap:5px;" });
      for (const [grp, msg] of holds) {
        tags.appendChild(el("div", {}, [el("span", { class: "badge badge-danger", text: `故障锁 ${grp}` }), " ", msg]));
      }
      for (const [room, msg] of manuals) {
        tags.appendChild(el("div", {}, [el("span", { class: "badge badge-warn", text: `人工暂停 ${room}` }), " ", msg]));
      }
      if (isolated.length) {
        tags.appendChild(el("div", {}, [el("span", { class: "badge badge-muted", text: "手动隔离设备" }), " ", isolated.join(", ")]));
      }
      if (pausedZones.length) {
        tags.appendChild(el("div", {}, [el("span", { class: "badge badge-muted", text: "暂停控制区域" }), " ", pausedZones.join(", ")]));
      }
      lockPanel.appendChild(tags);
    }
    body.appendChild(lockPanel);

    // Active transactions
    const txs = ledger.active_transactions || [];
    if (txs.length > 0) {
      const txPanel = el("div", { class: "mini-panel" }, [
        el("div", { class: "mini-panel-title" }, [
          el("span", { text: "进行中的多步事务 (transitions)" }),
          el("span", { class: "badge badge-accent", text: `${txs.length} 组` }),
        ]),
      ]);
      for (const tx of txs) {
        for (const m of tx.members || []) {
          txPanel.appendChild(
            el("div", { style: "display:flex; justify-content:space-between; gap:8px; padding:3px 0;" }, [
              el("span", { text: `${m.room} (${tx.mode}) · 阶段 ${m.phase}${m.initial !== null ? ` · 初值 ${m.initial}℃` : ""}` }),
              el("span", {
                class: "mono",
                style: "color:var(--accent);",
                text: tx.ready_in_s > 0 ? `间隔剩 ${fmtDur(tx.ready_in_s)}` : "就绪",
              }),
            ])
          );
        }
      }
      body.appendChild(txPanel);
    }

    // Parked actuators timers
    const parked = ledger.parked || [];
    if (parked.length > 0) {
      const pPanel = el("div", { class: "mini-panel" }, [
        el("div", { class: "mini-panel-title" }, [
          el("span", { text: "停机驻留设备保护计时 (Parked)" }),
          el("span", { class: "badge badge-purple", text: `${parked.length} 台` }),
        ]),
      ]);
      for (const p of parked) {
        const minOffTxt = p.min_off_remaining_s > 0 ? `停机剩 ${fmtDur(p.min_off_remaining_s)}` : "最短停机已满";
        const revTxt = p.reversal_remaining_s > 0 ? `换向保护剩 ${fmtDur(p.reversal_remaining_s)}` : "换向保护已满";
        pPanel.appendChild(
          el("div", { style: "display:flex; flex-wrap:wrap; justify-content:space-between; gap:6px; padding:4px 0; border-top:1px solid var(--border);" }, [
            el("span", { text: `${p.room_name || p.room}（原模式 ${p.mode || "—"} · 已停 ${fmtDur(p.off_elapsed_s)}）` }),
            el("div", { style: "display:flex; gap:6px;" }, [
              el("span", { class: `badge ${p.min_off_remaining_s > 0 ? "badge-warn" : "badge-ok"}`, text: minOffTxt }),
              el("span", { class: `badge ${p.reversal_remaining_s > 0 ? "badge-muted" : "badge-ok"}`, text: revTxt }),
            ]),
          ])
        );
      }
      body.appendChild(pPanel);
    }
  }

  function renderBandGauge(zone, policy) {
    const maxRange = 1.0; // ±1.0℃ visual span
    const toPct = (delta) => Math.max(2, Math.min(98, 50 + (delta / maxRange) * 50));
    const trackHalf = Number(policy?.tracking_band ?? 0.3);
    const idealHalf = Number(policy?.ideal_band ?? 0.5);
    const comfortHalf = Number(policy?.comfort_band ?? 0.7);

    const gauge = el("div", { class: "band-gauge", title: "以目标温度为中心的偏差标尺（±1.0℃）" });
    gauge.appendChild(el("div", { class: "band-zone band-comfort", style: `left:${toPct(-comfortHalf)}%; width:${(comfortHalf / maxRange) * 100}%` }));
    gauge.appendChild(el("div", { class: "band-zone band-ideal", style: `left:${toPct(-idealHalf)}%; width:${(idealHalf / maxRange) * 100}%` }));
    gauge.appendChild(el("div", { class: "band-zone band-track", style: `left:${toPct(-trackHalf)}%; width:${(trackHalf / maxRange) * 100}%` }));
    gauge.appendChild(el("div", { class: "band-center" }));

    if (zone.signed_delta_c !== null && zone.signed_delta_c !== undefined) {
      gauge.appendChild(el("div", { class: "band-marker", style: `left:${toPct(zone.signed_delta_c)}%`, title: `当前偏差 ${fmtSignedTemp(zone.signed_delta_c)}` }));
    }
    if (zone.trend && zone.signed_delta_c !== null) {
      const projDelta = zone.signed_delta_c + zone.trend.delta_10m_c;
      gauge.appendChild(el("div", { class: "band-projected", style: `left:${toPct(projDelta)}%`, title: `10分钟预测偏差 ${fmtSignedTemp(projDelta)}` }));
    }

    const scale = el("div", { class: "band-scale mono" }, [
      el("span", { text: "-1.0℃ (偏冷)" }),
      el("span", { text: `±${trackHalf}℃ 跟踪带` }),
      el("span", { text: "+1.0℃ (偏热)" }),
    ]);
    return el("div", {}, [gauge, scale]);
  }

  function renderSparkline(points, targetC, trackBand = 0.3) {
    if (!Array.isArray(points) || points.length < 2) {
      return el("div", { class: "sparkline-wrap", style: "color:var(--text-dim); font-size:11.5px; text-align:center;", text: "独立测温轨迹点较少，暂未绘制折线" });
    }
    const w = 360;
    const h = 76;
    const padX = 8;
    const padY = 10;
    const times = points.map((p) => p[0]);
    const temps = points.map((p) => p[1]);
    const tMin = Math.min(...times);
    const tMax = Math.max(...times);
    const tSpan = Math.max(60, tMax - tMin);
    const yValues = targetC !== null && targetC !== undefined ? [...temps, targetC - trackBand, targetC + trackBand] : temps;
    const vMin = Math.min(...yValues) - 0.15;
    const vMax = Math.max(...yValues) + 0.15;
    const vSpan = Math.max(0.4, vMax - vMin);

    const sx = (t) => padX + ((t - tMin) / tSpan) * (w - padX * 2);
    const sy = (v) => padY + (1 - (v - vMin) / vSpan) * (h - padY * 2);

    const poly = points.map((p) => `${sx(p[0]).toFixed(1)},${sy(p[1]).toFixed(1)}`).join(" ");
    let bandRect = "";
    let targetLine = "";
    if (targetC !== null && targetC !== undefined) {
      const yTop = sy(targetC + trackBand);
      const yBot = sy(targetC - trackBand);
      const yT = sy(targetC);
      bandRect = `<rect x="${padX}" y="${yTop.toFixed(1)}" width="${w - padX * 2}" height="${Math.max(2, yBot - yTop).toFixed(1)}" fill="rgba(52,211,153,0.16)" />`;
      targetLine = `<line x1="${padX}" y1="${yT.toFixed(1)}" x2="${w - padX}" y2="${yT.toFixed(1)}" stroke="var(--ok)" stroke-width="1" stroke-dasharray="4 3" />`;
    }
    const last = points[points.length - 1];
    const wrap = el("div", { class: "sparkline-wrap" });
    wrap.innerHTML = `<svg viewBox="0 0 ${w} ${h}" style="width:100%; height:72px; display:block;" role="img" aria-label="区域温度轨迹">
      ${bandRect}
      ${targetLine}
      <polyline points="${poly}" fill="none" stroke="var(--accent)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" />
      <circle cx="${sx(last[0]).toFixed(1)}" cy="${sy(last[1]).toFixed(1)}" r="3.5" fill="var(--accent)" />
      <text x="${padX + 2}" y="12" font-size="9.5" fill="var(--text-dim)">${vMax.toFixed(1)}℃</text>
      <text x="${padX + 2}" y="${h - 4}" font-size="9.5" fill="var(--text-dim)">${vMin.toFixed(1)}℃</text>
    </svg>`;
    return wrap;
  }

  function renderZones(snap) {
    const grid = document.getElementById("zones-grid");
    grid.replaceChildren();
    const zones = snap.state_machines?.zone_states || [];
    if (!zones.length) {
      grid.appendChild(el("div", { class: "empty-state", text: "暂无独立区域状态数据（未配置 topology.json 或控制器尚未输出观测）。" }));
      return;
    }

    for (const z of zones) {
      const card = el("article", { class: "card zone-card" });

      const badges = el("div", { style: "display:flex; flex-wrap:wrap; gap:5px; justify-content:flex-end;" }, [
        el("span", { class: `badge ${intentBadgeClass(z.state)}`, text: z.state_label }),
        z.is_owner ? el("span", { class: "badge badge-warn", text: "当前主区 Owner" }) : null,
        z.paused ? el("span", { class: "badge badge-danger", text: "已暂停" }) : null,
      ]);

      card.appendChild(
        el("div", { class: "zone-top" }, [
          el("div", {}, [
            el("h3", { class: "card-title", text: z.label }),
            el("div", { class: "card-subtitle", text: `${z.reason_label} (${z.reason})` }),
          ]),
          badges,
        ])
      );

      card.appendChild(
        el("div", { class: "zone-temp-row" }, [
          el("span", { class: "temp-big mono", text: fmtTemp(z.measured_c, 2) }),
          el("span", { class: "temp-meta mono", text: `目标 ${fmtTemp(z.target_c, 1)}` }),
          el("span", {
            class: `badge mono ${z.signed_delta_c === null ? "badge-muted" : Math.abs(z.signed_delta_c) <= 0.3 ? "badge-ok" : z.signed_delta_c < 0 ? "badge-cool" : "badge-heat"}`,
            text: `偏差 ${fmtSignedTemp(z.signed_delta_c)}`,
          }),
          el("span", { class: "temp-meta", text: z.report_age_s !== null ? `上报 ${fmtDur(z.report_age_s)}前` : "无新鲜上报" }),
        ])
      );

      card.appendChild(renderBandGauge(z, snap.policy));

      const trendText = z.trend
        ? `${z.trend.delta_10m_c >= 0 ? "+" : ""}${z.trend.delta_10m_c.toFixed(2)}℃ (±${z.trend.uncertainty_c.toFixed(2)}℃)`
        : "样本不足 / 等待新上报";
      const dwellVal = z.settling?.active
        ? `稳定确认 ${z.settling.reports}/2次 · ${fmtDur(z.settling.span_s)}`
        : `${fmtDur(z.dwell.duration_s)} / ${fmtDur(z.dwell.required_s)} (${z.dwell.pct}%)`;
      const reportsVal = `${z.independent_reports.count} 次 · 跨度 ${fmtDur(z.independent_reports.span_s)}`;
      const feasibleVal = z.feasible_actuators?.length ? z.feasible_actuators.join(", ") : "无 / 待分配";

      card.appendChild(
        el("div", { class: "zone-metrics-grid" }, [
          el("div", { class: "metric-box" }, [
            el("div", { class: "metric-box-label", text: "10 分钟温度趋势" }),
            el("div", { class: "metric-box-val mono", text: trendText }),
          ]),
          el("div", { class: "metric-box" }, [
            el("div", { class: "metric-box-label", text: z.settling?.active ? "回归稳定确认" : "持续偏差时长" }),
            el("div", { class: "metric-box-val mono", text: dwellVal }),
          ]),
          el("div", { class: "metric-box" }, [
            el("div", { class: "metric-box-label", text: "独立上报证据" }),
            el("div", { class: "metric-box-val mono", text: reportsVal }),
          ]),
          el("div", { class: "metric-box" }, [
            el("div", { class: "metric-box-label", text: "可用设备 / 方向" }),
            el("div", { class: "metric-box-val", text: `${feasibleVal} · ${z.direction_label}` }),
          ]),
        ])
      );

      let sparkPoints = z.points;
      if ((!Array.isArray(sparkPoints) || sparkPoints.length < 2) && snap.shadow_series) {
        const roomMatch = (snap.actuators || []).find((a) => a.zone === z.zone && snap.shadow_series[a.room]);
        if (roomMatch) {
          sparkPoints = snap.shadow_series[roomMatch.room]
            .filter((p) => p.temp !== null && p.temp !== undefined)
            .map((p) => [p.t, p.temp]);
        }
      }
      card.appendChild(renderSparkline(sparkPoints, z.target_c, snap.policy?.tracking_band ?? 0.3));

      if (z.unmet_reason_label) {
        card.appendChild(
          el("div", { class: "badge badge-warn", text: `未满足原因：${z.unmet_reason_label}` })
        );
      }

      grid.appendChild(card);
    }
  }

  function renderActuators(snap) {
    const grid = document.getElementById("actuators-grid");
    grid.replaceChildren();
    const actuators = snap.actuators || [];
    if (!actuators.length) {
      grid.appendChild(el("div", { class: "empty-state", text: "暂无空调设备配置或运行数据。" }));
      return;
    }

    for (const a of actuators) {
      const card = el("article", { class: "card actuator-card" });

      const statusPill = a.hold_reason
        ? el("span", { class: "badge badge-danger", text: "故障锁" })
        : a.manual_hold_reason
        ? el("span", { class: "badge badge-warn", text: "人工暂停" })
        : a.isolated
        ? el("span", { class: "badge badge-muted", text: "已隔离" })
        : a.parked
        ? el("span", { class: "badge badge-purple", text: "Parked 驻留" })
        : a.ready
        ? el("span", { class: "badge badge-ok", text: "可控就绪" })
        : el("span", { class: "badge badge-warn", text: "待就绪" });

      card.appendChild(
        el("div", { class: "zone-top" }, [
          el("div", {}, [
            el("h3", { class: "card-title", text: a.name }),
            el("div", { class: "card-subtitle mono", text: `本区 ${a.zone} · 服务 ${a.served_zone} · 锁组 ${a.lock_group}` }),
          ]),
          el("div", { style: "display:flex; gap:5px; flex-wrap:wrap; justify-content:flex-end;" }, [
            el("span", { class: `badge ${modeBadgeClass(a.mode)}`, text: a.mode_label }),
            statusPill,
          ]),
        ])
      );

      const limits = a.soft_limits?.[a.mode];
      const limText = Array.isArray(limits) ? `软限 ${limits[0]}–${limits[1]}℃` : "—";
      card.appendChild(
        el("div", { class: "actuator-stats" }, [
          el("div", { class: "stat-cell" }, [
            el("div", { class: "stat-label", text: "当前设温" }),
            el("div", { class: "stat-val mono", text: a.setpoint !== null ? `${a.setpoint}℃` : "—" }),
            el("div", { class: "stat-sub", text: limText }),
          ]),
          el("div", { class: "stat-cell" }, [
            el("div", { class: "stat-label", text: "实时功率" }),
            el("div", { class: "stat-val mono", text: fmtWatts(a.power_w) }),
            el("div", { class: "stat-sub", text: a.power_age_s !== null ? `${Math.round(a.power_age_s)}秒前` : "无上报" }),
          ]),
          el("div", { class: "stat-cell" }, [
            el("div", { class: "stat-label", text: "房间温度" }),
            el("div", { class: "stat-val mono", text: fmtTemp(a.room_c, 2) }),
            el("div", { class: "stat-sub", text: a.target_c !== null ? `目标 ${a.target_c}℃` : "—" }),
          ]),
        ])
      );

      // Power bands & output evidence
      const bands = a.power_bands || {};
      const bandRow = el("div", { class: "power-bands-row" }, [
        el("span", {
          class: `badge ${a.output_credible ? "badge-ok" : "badge-muted"}`,
          text: a.output_credible ? "有效输出已证实" : "无有效负载输出",
        }),
      ]);
      if (bands.low && bands.low.seconds > 0) {
        bandRow.appendChild(el("span", { class: "badge badge-purple mono", text: `低耗 ${fmtDur(bands.low.seconds)}` }));
      }
      if (bands.above_low && bands.above_low.seconds > 0) {
        bandRow.appendChild(el("span", { class: "badge badge-accent mono", text: `在载 ${fmtDur(bands.above_low.seconds)}` }));
      }
      if (bands.high && bands.high.seconds > 0) {
        bandRow.appendChild(el("span", { class: "badge badge-warn mono", text: `高功率 ${fmtDur(bands.high.seconds)}` }));
      }
      card.appendChild(bandRow);

      // Current decision / status reason
      const displayReason = a.hold_reason || a.manual_hold_reason || a.status_label || "—";
      const showCode = a.status_code && a.status_code !== displayReason;
      const reasonPanel = el("div", { class: "mini-panel" }, [
        el("div", { class: "mini-panel-title" }, [
          el("span", { text: "当前决策结论" }),
          showCode ? el("span", { class: "mono", style: "font-weight:400;", text: a.status_code }) : null,
        ]),
        el("div", { text: displayReason }),
      ]);
      card.appendChild(reasonPanel);

      // Risks / Learned maintenance / Shadow 24h mini summary
      const footerTags = el("div", { class: "power-bands-row" });
      for (const r of a.risks || []) {
        footerTags.appendChild(el("span", { class: "badge badge-warn", text: `风险(${r.point}): ${r.reason}` }));
      }
      for (const [mKey, mVal] of Object.entries(a.learned_maintenance || {})) {
        footerTags.appendChild(el("span", { class: "badge badge-muted mono", text: `维持学习(${mKey}): ${mVal.setpoint}℃` }));
      }
      if (a.shadow_latest?.within_comfort_pct !== null && a.shadow_latest?.within_comfort_pct !== undefined) {
        footerTags.appendChild(
          el("span", {
            class: "badge badge-muted mono",
            text: `24h 舒适 ${a.shadow_latest.within_comfort_pct}% · ${a.shadow_latest.metered_kwh !== null ? `${a.shadow_latest.metered_kwh.toFixed(2)} kWh` : "—"}`,
          })
        );
      }
      if (footerTags.childNodes.length) card.appendChild(footerTags);

      grid.appendChild(card);
    }
  }

  function renderCommandsTab(snap) {
    const cmds = snap.commands || [];
    document.getElementById("count-commands").textContent = String(cmds.length);
    const panel = document.getElementById("tab-commands");
    panel.replaceChildren();
    if (!cmds.length) {
      panel.appendChild(el("div", { class: "empty-state", text: "近 24 小时暂无实机发令记录。" }));
      return;
    }

    const table = el("table", { class: "data-table" });
    table.appendChild(
      el("thead", {}, [
        el("tr", {}, [
          el("th", { text: "发令时间" }),
          el("th", { text: "设备与动作" }),
          el("th", { text: "用途" }),
          el("th", { text: "回执状态" }),
          el("th", { text: "延迟 (中继/回显)" }),
          el("th", { text: "发令前" }),
          el("th", { text: "5分钟随访" }),
          el("th", { text: "15分钟随访" }),
        ]),
      ])
    );

    const tbody = el("tbody");
    for (const c of cmds) {
      const f5 = c.followup?.["5"];
      const f15 = c.followup?.["15"];
      const fmtF = (f) => (f ? `${fmtTemp(f.room_c, 2)} / ${fmtWatts(f.watts)}` : "待采");
      tbody.appendChild(
        el("tr", {}, [
          el("td", { class: "mono", text: fmtDateTime(c.sent_at) }),
          el("td", {}, [
            el("strong", { text: c.room_name }),
            el("div", { class: "mono", style: "color:var(--accent);", text: c.short_transition }),
          ]),
          el("td", { text: c.purpose_label }),
          el("td", {}, [
            el("span", { class: `badge ${statusBadgeClass(c.status)}`, text: c.status_label }),
            c.physical_verified ? el("div", { style: "margin-top:3px;" }, [el("span", { class: "badge badge-ok", text: "现场确认" })]) : null,
          ]),
          el("td", {
            class: "mono",
            text: `${c.relay_delay_s !== null ? `${c.relay_delay_s}s` : "—"} / ${c.echo_delay_s !== null ? `${c.echo_delay_s}s` : "—"}`,
          }),
          el("td", { class: "mono", text: `${fmtTemp(c.before_room_c, 2)} / ${fmtWatts(c.before_watts)}` }),
          el("td", { class: "mono", text: fmtF(f5) }),
          el("td", { class: "mono", text: fmtF(f15) }),
        ])
      );
    }
    table.appendChild(tbody);
    panel.appendChild(el("div", { class: "table-scroll" }, [table]));
  }

  function renderAuditTab(snap) {
    const rows = snap.audit_recent || [];
    document.getElementById("count-audit").textContent = String(rows.length);
    const panel = document.getElementById("tab-audit");
    panel.replaceChildren();

    const filterRow = el("div", { class: "filter-row" });
    const filters = [
      ["all", "全部事件"],
      ["actions", "仅发令 / 回执 / 启动"],
      ["alerts", "仅异常 / 拦截 / 复核取消"],
    ];
    for (const [id, label] of filters) {
      const btn = el("button", {
        type: "button",
        class: `filter-btn ${state.auditFilter === id ? "active" : ""}`,
        text: label,
      });
      btn.addEventListener("click", () => {
        state.auditFilter = id;
        renderAuditTab(state.snapshot);
      });
      filterRow.appendChild(btn);
    }
    panel.appendChild(filterRow);

    const filtered = rows.filter((r) => {
      if (state.auditFilter === "actions") {
        return r.event === "relay" || r.event === "receipt" || r.event === "startup" || r.outcome === "dispatch";
      }
      if (state.auditFilter === "alerts") {
        return Boolean(r.error) || r.rejections_count > 0 || r.outcome === "recheck_changed" || r.outcome === "permit_revoked";
      }
      return true;
    });

    if (!filtered.length) {
      panel.appendChild(el("div", { class: "empty-state", text: "当前筛选条件下暂无审计事件。" }));
      return;
    }

    const table = el("table", { class: "data-table" });
    table.appendChild(
      el("thead", {}, [
        el("tr", {}, [
          el("th", { text: "时间" }),
          el("th", { text: "事件类型" }),
          el("th", { text: "全屋需求 / 主区" }),
          el("th", { text: "摘要" }),
        ]),
      ])
    );
    const tbody = el("tbody");
    for (const r of filtered) {
      const evCls = r.error
        ? "badge-danger"
        : r.outcome === "dispatch" || r.event === "receipt"
        ? "badge-ok"
        : r.rejections_count > 0 || r.outcome === "recheck_changed"
        ? "badge-warn"
        : "badge-muted";
      tbody.appendChild(
        el("tr", {}, [
          el("td", { class: "mono", text: fmtDateTime(r.at) }),
          el("td", {}, [el("span", { class: `badge ${evCls}`, text: r.event === "cycle" ? r.outcome_label : r.event })]),
          el("td", { class: "mono", text: `${r.need || "—"} / ${r.owner || "—"}` }),
          el("td", { text: r.summary }),
        ])
      );
    }
    table.appendChild(tbody);
    panel.appendChild(el("div", { class: "table-scroll" }, [table]));
  }

  function renderShadowTab(snap) {
    const panel = document.getElementById("tab-shadow");
    panel.replaceChildren();
    const sh = snap.shadow || { available: false };
    const ns = snap.neural_shadow || { available: false };

    if (!sh.available && !ns.available && !(snap.effects || []).length) {
      panel.appendChild(el("div", { class: "empty-state", text: "未检测到影子评估输出（outputs/climate-shadow/shadow-status.json）。" }));
      return;
    }

    if (sh.available) {
      const act = sh.live_actuations || {};
      const grid = el("div", { class: "kv-grid", style: "margin-bottom:14px;" }, [
        el("div", { class: "kv-item" }, [
          el("div", { class: "kv-k", text: "影子服务健康状态" }),
          el("div", { class: "kv-v", text: sh.stale ? `数据过期 (${sh.health_error || "stale"})` : "健康运行 (healthy)" }),
        ]),
        el("div", { class: "kv-item" }, [
          el("div", { class: "kv-k", text: "24h 全屋共同计量电量" }),
          el("div", { class: "kv-v mono", text: sh.observed_home_kwh !== null ? `${sh.observed_home_kwh.toFixed(3)} kWh (覆盖 ${sh.observed_home_coverage_pct ?? "—"}%)` : "—" }),
        ]),
        el("div", { class: "kv-item" }, [
          el("div", { class: "kv-k", text: "窗口实机发令 / 伴侣回报" }),
          el("div", { class: "kv-v mono", text: `${act.attempts ?? 0} 次尝试 · ${act.proxy_echoes ?? 0} 次回报（启${act.starts ?? 0} / 停${act.stops ?? 0} / 换向${act.reversals ?? 0}）` }),
        ]),
        el("div", { class: "kv-item" }, [
          el("div", { class: "kv-k", text: "24h 验收就绪状态" }),
          el("div", { class: "kv-v", text: sh.data_ready_for_24h_review ? `已就绪 (${sh.assessment_completed_minutes} 分钟)` : `累计中 (${sh.assessment_completed_minutes ?? 0} 分钟)` }),
        ]),
      ]);
      panel.appendChild(grid);
    }

    if (ns.available) {
      const nsPanel = el("div", { class: "mini-panel", style: "margin-bottom:14px;" }, [
        el("div", { class: "mini-panel-title" }, [
          el("span", { text: "神经影子模型 (Neural Shadow · 仅观测评分，零发令)" }),
          el("span", { class: `badge ${ns.deployment_eligible ? "badge-ok" : "badge-muted"}`, text: ns.model_state || "unknown" }),
        ]),
        el("div", {
          class: "mono",
          text: `架构 ${ns.model_arch_version || "—"} · 特征 ${ns.feature_version || "—"} · 已封存 ${ns.counts?.sealed_predictions ?? 0} 次 · 已结算 ${ns.counts?.settled_horizons ?? 0} 视界`,
        }),
      ]);
      panel.appendChild(nsPanel);
    }

    const effects = snap.effects || [];
    if (effects.length) {
      panel.appendChild(el("h4", { style: "margin:10px 0 8px; font-size:13px;", text: "最近动作响应与能耗效果样本 (closed_loop.effects)" }));
      const tbl = el("table", { class: "data-table" });
      tbl.appendChild(
        el("thead", {}, [
          el("tr", {}, [
            el("th", { text: "结束时间" }),
            el("th", { text: "设备 / 区域" }),
            el("th", { text: "温度变化" }),
            el("th", { text: "偏差改善" }),
            el("th", { text: "区间电量" }),
            el("th", { text: "有效性" }),
          ]),
        ])
      );
      const tb = el("tbody");
      for (const e of effects) {
        tb.appendChild(
          el("tr", {}, [
            el("td", { class: "mono", text: fmtDateTime(e.ended_at) }),
            el("td", { class: "mono", text: `${e.unit} → ${e.zone} (${e.mode})` }),
            el("td", { class: "mono", text: `${fmtTemp(e.start_c, 2)} → ${fmtTemp(e.end_c, 2)}` }),
            el("td", { class: "mono", text: fmtSignedTemp(e.improvement) }),
            el("td", { class: "mono", text: `${Number(e.watt_hours || 0).toFixed(1)} Wh (${Math.round(e.covered_seconds || 0)}s)` }),
            el("td", {}, [el("span", { class: `badge ${e.useful ? "badge-ok" : "badge-muted"}`, text: e.useful ? "有效改善" : "未达阈值" })]),
          ])
        );
      }
      tbl.appendChild(tb);
      panel.appendChild(el("div", { class: "table-scroll" }, [tbl]));
    }
  }

  function renderPolicyTab(snap) {
    const panel = document.getElementById("tab-policy");
    panel.replaceChildren();
    const ctrl = snap.controller;
    const permit = ctrl.permit || {};
    const rel = ctrl.release || {};

    panel.appendChild(el("h4", { style: "margin:0 0 8px; font-size:13px;", text: "版本溯源与控制许可" }));
    panel.appendChild(
      el("div", { class: "kv-grid", style: "margin-bottom:14px;" }, [
        el("div", { class: "kv-item" }, [
          el("div", { class: "kv-k", text: "许可状态" }),
          el("div", { class: "kv-v", text: permit.active ? `已生效 (${permit.trial ? "限时试点" : "持续控制"} · ${permit.source})` : `未生效 (${permit.reason || "—"})` }),
        ]),
        el("div", { class: "kv-item" }, [
          el("div", { class: "kv-k", text: "Git Commit" }),
          el("div", { class: "kv-v mono", text: rel.commit ? rel.commit.slice(0, 12) : "未标记 (.release)" }),
        ]),
        el("div", { class: "kv-item" }, [
          el("div", { class: "kv-k", text: "code_sha256" }),
          el("div", { class: "kv-v mono", text: rel.code_sha256 ? rel.code_sha256.slice(0, 16) : "—" }),
        ]),
        el("div", { class: "kv-item" }, [
          el("div", { class: "kv-k", text: "config_sha256" }),
          el("div", { class: "kv-v mono", text: rel.config_sha256 ? rel.config_sha256.slice(0, 16) : "—" }),
        ]),
      ])
    );

    panel.appendChild(el("h4", { style: "margin:10px 0 8px; font-size:13px;", text: "当前生效策略参数 (house_policy.json)" }));
    const grid = el("div", { class: "kv-grid" });
    const policy = snap.policy || {};
    for (const [key, [label, unit]] of Object.entries(POLICY_LABELS)) {
      if (!(key in policy)) continue;
      const val = policy[key];
      const formatted = unit === "s" ? `${val}s (${fmtDur(val)})` : `${val} ${unit}`;
      grid.appendChild(
        el("div", { class: "kv-item" }, [
          el("div", { class: "kv-k", text: `${label} (${key})` }),
          el("div", { class: "kv-v mono", text: formatted }),
        ])
      );
    }
    panel.appendChild(grid);
  }

  function renderAll(snap) {
    state.snapshot = snap;
    state.receivedAtMs = Date.now();
    if (snap.mode === "demo" && !state.scenario) {
      state.scenario = snap.scenario;
    }

    renderDemoBanner(snap);
    renderTopbar(snap);
    renderNextCommand(snap);
    renderZoneStateMachineSvg(snap);
    renderHandoffStateMachine(snap);
    renderLedgerAndLocks(snap);
    renderZones(snap);
    renderActuators(snap);
    renderCommandsTab(snap);
    renderAuditTab(snap);
    renderShadowTab(snap);
    renderPolicyTab(snap);

    document.getElementById("footer-updated").textContent = `最后同步：${fmtClock(snap.generated_at)}`;
  }

  // ---------- Live transport & countdown ticker ----------

  function apiQuery() {
    return state.scenario ? `?scenario=${encodeURIComponent(state.scenario)}` : "";
  }

  function setStreamStatus(text, isLive) {
    const pill = document.getElementById("stream-status");
    pill.textContent = text;
    pill.className = `stream-pill ${isLive ? "live" : ""}`;
  }

  async function fetchSnapshot() {
    try {
      const resp = await fetch(`/api/snapshot${apiQuery()}`, { cache: "no-store" });
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const data = await resp.json();
      renderAll(data);
    } catch (_err) {
      setStreamStatus("连接中断，重试中…", false);
    }
  }

  function startPollingFallback() {
    if (state.pollTimer) return;
    setStreamStatus("轮询刷新 (3s)", false);
    state.pollTimer = setInterval(fetchSnapshot, 3000);
  }

  function stopPollingFallback() {
    if (state.pollTimer) {
      clearInterval(state.pollTimer);
      state.pollTimer = null;
    }
  }

  function connectStream() {
    if (state.eventSource) {
      state.eventSource.close();
      state.eventSource = null;
    }
    if (typeof EventSource === "undefined") {
      startPollingFallback();
      return;
    }
    const es = new EventSource(`/api/stream${apiQuery()}`);
    state.eventSource = es;
    es.addEventListener("snapshot", (ev) => {
      try {
        const data = JSON.parse(ev.data);
        stopPollingFallback();
        setStreamStatus("SSE 实时同步", true);
        renderAll(data);
      } catch (_err) {
        // Ignore malformed frame
      }
    });
    es.onerror = () => {
      startPollingFallback();
    };
  }

  function tickCountdowns() {
    document.querySelectorAll("[data-countdown]").forEach((node) => {
      const base = Number(node.getAttribute("data-countdown"));
      if (Number.isNaN(base)) return;
      const loop = Number(node.getAttribute("data-countdown-loop") || "0");
      const elapsed = (Date.now() - state.receivedAtMs) / 1000;
      if (loop > 0) {
        const mod = (((loop - base + elapsed) % loop) + loop) % loop;
        const rem = loop - mod;
        node.textContent = `${Math.max(1, Math.ceil(rem))}s`;
      } else {
        const rem = Math.max(0, base - elapsed);
        const prefix = node.textContent.startsWith("剩 ") ? "剩 " : "";
        node.textContent = `${prefix}${fmtDur(rem)}`;
      }
    });
  }

  function initUI() {
    // Tabs
    document.querySelectorAll(".tab-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        const tab = btn.getAttribute("data-tab");
        state.activeTab = tab;
        document.querySelectorAll(".tab-btn").forEach((b) => b.classList.toggle("active", b === btn));
        document.querySelectorAll(".tab-panel").forEach((p) => p.classList.toggle("active", p.id === `tab-${tab}`));
      });
    });

    // Theme toggle
    const savedTheme = localStorage.getItem("thermostat-dashboard-theme") || "dark";
    document.documentElement.setAttribute("data-theme", savedTheme);
    const themeBtn = document.getElementById("btn-theme");
    themeBtn.textContent = savedTheme === "dark" ? "亮色" : "深色";
    themeBtn.addEventListener("click", () => {
      const cur = document.documentElement.getAttribute("data-theme") === "light" ? "dark" : "light";
      document.documentElement.setAttribute("data-theme", cur);
      localStorage.setItem("thermostat-dashboard-theme", cur);
      themeBtn.textContent = cur === "dark" ? "亮色" : "深色";
    });

    // Manual refresh
    document.getElementById("btn-refresh").addEventListener("click", () => {
      fetchSnapshot();
    });

    // Initial load + SSE + 1s ticker
    fetchSnapshot();
    connectStream();
    setInterval(tickCountdowns, 1000);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initUI);
  } else {
    initUI();
  }
})();
