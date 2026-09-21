(() => {
  "use strict";

  const $ = (selector, root = document) => root.querySelector(selector);
  const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];
  const CANVAS_PRESETS = {
    portrait: [
      { ratio: "1:4", width: 512, height: 2048 },
      { ratio: "1:3", width: 576, height: 1728 },
      { ratio: "約9:16", width: 768, height: 1344 },
      { ratio: "約2:3", width: 832, height: 1216 },
      { ratio: "約3:4", width: 896, height: 1152 },
    ],
    landscape: [
      { ratio: "4:1", width: 2048, height: 512 },
      { ratio: "3:1", width: 1728, height: 576 },
      { ratio: "約16:9", width: 1344, height: 768 },
      { ratio: "約3:2", width: 1216, height: 832 },
      { ratio: "約4:3", width: 1152, height: 896 },
    ],
    square: [{ ratio: "1:1", width: 1024, height: 1024 }],
  };
  const state = {
    connected: false,
    capabilities: null,
    models: [],
    loras: [],
    fast4LoraId: null,
    activeJobId: null,
    activeJobMode: null,
    selectedHistoryItem: null,
    upscaleSourceItem: null,
    mode: "generate",
    pollTimer: null,
    generationStartedAt: 0,
    loadedModelId: null,
    loadedPreset: null,
    loadedAttentionBackend: null,
    loadedLoras: [],
    selectionRevision: 0,
    appliedRevision: 0,
    controlJobs: new Map(),
    controlPollTimers: new Map(),
    loraApplyTimer: null,
    pendingLoraApply: false,
    controlError: null,
    sizePresetIndex: 2,
  };

  const form = $("#generationForm");
  const generateButton = $("#generateButton");
  const formError = $("#formError");
  const modelSelect = $("#model");
  const modelNote = $("#modelNote");
  const loraList = $("#loraList");
  const previewImage = $("#previewImage");
  const emptyStage = $("#emptyStage");
  const generationOverlay = $("#generationOverlay");

  class ApiError extends Error {
    constructor(message, code = "request_failed", status = 0, details = null) {
      super(message);
      this.name = "ApiError";
      this.code = code;
      this.status = status;
      this.details = details;
    }
  }

  async function api(path, options = {}) {
    const controller = new AbortController();
    const timer = window.setTimeout(() => controller.abort(), options.timeout || 15000);
    try {
      const headers = new Headers(options.headers || {});
      if (options.body && typeof options.body !== "string") {
        headers.set("Content-Type", "application/json");
        options.body = JSON.stringify(options.body);
      }
      const response = await fetch(path, { ...options, headers, signal: controller.signal });
      const contentType = response.headers.get("content-type") || "";
      const payload = contentType.includes("json") ? await response.json() : null;
      if (!response.ok) {
        const error = payload?.error;
        throw new ApiError(error?.message || `リクエストに失敗しました (${response.status})`, error?.code, response.status, error?.details);
      }
      return payload;
    } catch (error) {
      if (error.name === "AbortError") throw new ApiError("エンジンからの応答がありません", "timeout");
      if (error instanceof ApiError) throw error;
      throw new ApiError("ローカル生成エンジンに接続できません", "connection_failed");
    } finally {
      window.clearTimeout(timer);
    }
  }

  function setConnection(connected, text) {
    state.connected = connected;
    const box = $("#backendState");
    box.classList.toggle("is-online", connected);
    box.classList.toggle("is-offline", !connected);
    $("#backendStateText").textContent = text;
    $("#dialogStatus").textContent = text;
    const dialogDot = $("#settingsDialog .status-dot");
    dialogDot.classList.toggle("is-online", connected);
    dialogDot.classList.toggle("is-offline", !connected);
    updateGenerateAvailability();
    updateModelRuntimeUI();
  }

  function showError(message) {
    formError.textContent = message;
    formError.hidden = false;
  }

  function clearError() {
    formError.textContent = "";
    formError.hidden = true;
  }

  function updateGenerateAvailability() {
    const hasModel = modelSelect.value && !modelSelect.disabled;
    const ready = state.mode === "upscale" ? Boolean(state.upscaleSourceItem) : hasModel;
    generateButton.disabled = !state.connected || !ready || Boolean(state.activeJobId) || state.controlJobs.size > 0;
    $(".generate-label", generateButton).textContent = state.activeJobId
      ? (state.activeJobMode === "upscale" ? "アップスケール中…" : "生成中…")
      : (state.mode === "upscale" ? "アップスケール" : "生成する");
  }

  function option(value, label, disabled = false) {
    const node = document.createElement("option");
    node.value = value;
    node.textContent = label;
    node.disabled = disabled;
    return node;
  }

  async function connect() {
    setConnection(false, "エンジンを確認中");
    clearError();
    try {
      const [capabilities, models, loras, engineState, settings] = await Promise.all([
        api("/api/capabilities"), api("/api/models"), api("/api/loras"), api("/api/state"), api("/api/settings"),
      ]);
      state.capabilities = capabilities;
      state.models = models.items || [];
      state.loras = loras.items || [];
      state.fast4LoraId = loras.fast4_lora_id || null;
      renderModels(models.default_id);
      renderCapabilities(capabilities, engineState);
      applyStoredSettings(settings);
      syncLoadedState(engineState);
      setConnection(true, statusLabel(engineState, capabilities));
      if (engineState.active_job_id) {
        if (["model_load", "loras_load"].includes(engineState.operation)) {
          trackControlJob(engineState.active_job_id, Number(engineState.requested_selection_revision ?? engineState.selection_revision) || 0, engineState.operation);
        } else {
          state.activeJobId = engineState.active_job_id;
          startPolling();
        }
      }
      await loadHistory();
    } catch (error) {
      modelSelect.replaceChildren(option("", "利用可能なモデルなし"));
      modelSelect.disabled = true;
      modelNote.textContent = "エンジンを起動してから再接続してください";
      setConnection(false, "エンジン未接続");
      showError(error.message);
      renderOfflineFacts();
      updateModelRuntimeUI();
    }
  }

  function statusLabel(engineState, capabilities) {
    if (!engineState) return "接続済み";
    if (engineState.status === "generating") return "生成中";
    if (engineState.status === "loading" || engineState.status === "configuring") return engineState.operation === "loras_load" ? "LoRAを適用中" : "モデルを読込中";
    if (engineState.status === "error") return "エンジンエラー";
    const deviceName = capabilities?.device?.name || capabilities?.device?.device_name;
    return deviceName ? `${deviceName} · 待機中` : "エンジン接続済み";
  }

  function renderModels(defaultId) {
    modelSelect.replaceChildren();
    const available = state.models.filter((model) => model.available);
    state.models.forEach((model) => {
      const node = option(model.id, `${model.name}${model.available ? "" : " — 利用不可"}`, !model.available);
      if (!model.available) node.title = localizedModelReason(model);
      modelSelect.append(node);
    });
    if (!state.models.length) modelSelect.append(option("", "モデルが見つかりません"));
    const selected = available.find((item) => item.id === defaultId) || available.find((item) => item.is_default) || available[0];
    modelSelect.value = selected?.id || "";
    modelSelect.disabled = !available.length;
    updateModelNote(selected);
    renderPresetAvailability();
    syncModelCompatibility(selectedPreset());
    updateGenerateAvailability();
  }

  function modelDescription(model) {
    const kind = model.kind === "diffusers" ? "Diffusers" : model.kind === "single_file_bf16" ? "BF16 単一ファイル" : "非対応形式";
    return `${kind} · ローカル`;
  }

  function localizedModelReason(model) {
    if (model.kind === "unsupported_quantized") return "ComfyUI向けINT8/ConvRot形式は、この独立エンジンでは読み込めません";
    const reason = model.reason || "このモデルは利用できません";
    if (/not found|missing/i.test(reason)) return "必要なモデルまたは構成ファイルが見つかりません";
    return reason;
  }

  function updateModelNote(selected) {
    const unavailable = state.models.filter((model) => !model.available);
    if (selected) {
      const quantizedUnavailable = unavailable.some((model) => model.kind === "unsupported_quantized");
      modelNote.textContent = `${modelDescription(selected)}${quantizedUnavailable ? " · INT8/ConvRotは独立エンジン非対応" : ""}`;
    } else {
      modelNote.textContent = unavailable.length ? `利用できるモデルなし · ${localizedModelReason(unavailable[0])}` : "対応するローカルモデルを配置してください";
    }
    modelNote.title = unavailable.map((model) => `${model.name}: ${localizedModelReason(model)}`).join("\n");
  }

  function syncLoadedState(engineState) {
    state.loadedModelId = engineState.loaded_model_id || engineState.model_id || null;
    state.loadedPreset = engineState.loaded_preset || null;
    state.loadedAttentionBackend = engineState.attention_backend || null;
    state.loadedLoras = Array.isArray(engineState.loaded_loras) ? engineState.loaded_loras : [];
    const revision = Number(engineState.selection_revision) || 0;
    const requestedRevision = Number(engineState.requested_selection_revision) || revision;
    state.appliedRevision = Math.max(state.appliedRevision, revision);
    state.selectionRevision = Math.max(state.selectionRevision, requestedRevision, revision);
    if (state.loadedModelId === modelSelect.value && state.loadedPreset === selectedPreset() && loraSelectionMatchesLoaded()) {
      state.pendingLoraApply = false;
    }
    updateModelRuntimeUI();
  }

  function modelName(modelId) {
    return state.models.find((model) => model.id === modelId)?.name || modelId || "なし";
  }

  function updateModelRuntimeUI() {
    const root = $("#modelRuntimeStatus");
    const text = $("#modelRuntimeText");
    const button = $("#loadModel");
    if (!root || !button) return;
    root.classList.remove("is-loaded", "is-busy", "is-error");
    const jobs = [...state.controlJobs.values()];
    const latestJob = jobs[jobs.length - 1];
    const selectedMatchesLoaded = Boolean(modelSelect.value) && modelSelect.value === state.loadedModelId;
    const requestedAttention = $("#attentionBackend").value;
    const attentionMatches = requestedAttention === "auto"
      ? Boolean(state.loadedAttentionBackend)
      : requestedAttention === state.loadedAttentionBackend;
    const configMatches = selectedMatchesLoaded
      && state.loadedPreset === selectedPreset()
      && attentionMatches
      && loraSelectionMatchesLoaded();

    if (!state.connected) {
      text.textContent = "エンジン接続後に読み込めます";
    } else if (latestJob) {
      root.classList.add("is-busy");
      text.textContent = latestJob.operation === "loras_load" ? "LoRAを適用しています…" : `${modelName(latestJob.modelId)} を読み込んでいます…`;
    } else if (state.controlError) {
      root.classList.add("is-error");
      text.textContent = state.controlError;
    } else if (state.pendingLoraApply && selectedMatchesLoaded) {
      text.textContent = "LoRAの変更を適用予定";
    } else if (configMatches) {
      root.classList.add("is-loaded");
      text.textContent = `読込済み: ${modelName(state.loadedModelId)} · ${state.loadedLoras.length} LoRA`;
    } else if (state.loadedModelId) {
      root.classList.add("is-loaded");
      const suffix = selectedMatchesLoaded ? " · 現在の設定は未適用" : " · 選択モデルは未読込";
      text.textContent = `GPU: ${modelName(state.loadedModelId)}${suffix}`;
    } else {
      text.textContent = state.pendingLoraApply ? "モデル読込時にLoRAを適用します" : "GPUモデル未読込";
    }

    const busy = state.controlJobs.size > 0 || Boolean(state.activeJobId);
    button.disabled = !state.connected || !modelSelect.value || modelSelect.disabled || busy || configMatches;
    button.textContent = jobs.length ? "処理中…" : state.loadedModelId ? (selectedMatchesLoaded ? "更新する" : "読み込む") : "読み込む";
    updateGenerateAvailability();
  }

  function canonicalStyleLoras(loras) {
    return (loras || [])
      .filter((lora) => lora.enabled !== false && Math.abs(Number(lora.weight) || 0) > 1e-8)
      .filter((lora) => {
        const known = state.loras.find((entry) => entry.id === lora.id);
        return lora.id !== state.fast4LoraId && known?.category !== "distillation" && lora.role !== "fast4";
      })
      .map((lora) => `${lora.id}:${Number(lora.weight).toFixed(6)}`);
  }

  function loraSelectionMatchesLoaded() {
    const selected = canonicalStyleLoras(collectLoraSelection());
    const loaded = canonicalStyleLoras(state.loadedLoras);
    return selected.length === loaded.length && selected.every((value, index) => value === loaded[index]);
  }

  function attentionSelectionMatchesLoaded() {
    const requested = $("#attentionBackend").value;
    return requested === "auto" ? Boolean(state.loadedAttentionBackend) : requested === state.loadedAttentionBackend;
  }

  function modelFamily(model) {
    if (model.family === "turbo" || model.family === "raw") return model.family;
    const searchable = `${model.id || ""} ${model.name || ""}`.toLowerCase();
    return searchable.includes("raw") ? "raw" : "turbo";
  }

  function renderPresetAvailability() {
    const hasTurbo = state.models.some((model) => model.available && modelFamily(model) === "turbo");
    const hasRaw = state.models.some((model) => model.available && modelFamily(model) === "raw");
    const availability = {
      turbo8: { available: hasTurbo, reason: "対応するTurboモデルが登録されていません", shortReason: "対応モデル未登録" },
      fast4: { available: hasTurbo && Boolean(state.fast4LoraId), reason: hasTurbo ? "4-step LoRAが見つかりません" : "Turboモデルが登録されていません", shortReason: hasTurbo ? "4-step LoRAなし" : "対応モデル未登録" },
      raw: { available: hasRaw, reason: "対応するRawモデルが登録されていません", shortReason: "対応モデル未登録" },
    };
    const reported = state.capabilities?.features?.presets;
    const reportedItems = Array.isArray(reported)
      ? reported
      : Object.entries(reported || {}).map(([id, value]) => ({ id, ...(typeof value === "object" ? value : { available: Boolean(value) }) }));
    reportedItems.forEach((preset) => {
      if (availability[preset.id]) availability[preset.id] = { ...availability[preset.id], available: Boolean(preset.available), reason: preset.reason || availability[preset.id].reason };
    });
    Object.entries(availability).forEach(([preset, status]) => {
      const input = $(`input[name="preset"][value="${preset}"]`);
      if (!input) return;
      input.disabled = !status.available;
      const card = input.closest("label");
      card.setAttribute("aria-disabled", String(!status.available));
      card.title = status.available ? "" : status.reason;
      const detail = $("small", card);
      if (!detail.dataset.originalText) detail.dataset.originalText = detail.textContent;
      detail.textContent = status.available ? detail.dataset.originalText : status.shortReason;
    });
    const checked = $("input[name='preset']:checked");
    if (!checked || checked.disabled) {
      const fallback = $("input[name='preset']:not(:disabled)");
      if (fallback) fallback.checked = true;
    }
  }

  function syncModelCompatibility(preset) {
    const family = preset === "raw" ? "raw" : "turbo";
    [...modelSelect.options].forEach((node) => {
      const model = state.models.find((item) => item.id === node.value);
      if (!model) return;
      node.disabled = !model.available || modelFamily(model) !== family;
    });
    const selected = state.models.find((model) => model.id === modelSelect.value);
    if (!selected || !selected.available || modelFamily(selected) !== family) {
      const compatible = state.models.find((model) => model.available && modelFamily(model) === family);
      modelSelect.value = compatible?.id || "";
    }
    const current = state.models.find((model) => model.id === modelSelect.value);
    modelSelect.disabled = !current;
    updateModelNote(current);
    updateGenerateAvailability();
  }

  function renderCapabilities(capabilities, engineState) {
    const attention = capabilities.features?.attention_backends || [];
    const attentionSelect = $("#attentionBackend");
    attentionSelect.replaceChildren(option("auto", "自動（推奨）"));
    if (attention.includes("sdpa")) attentionSelect.append(option("sdpa", "PyTorch SDPA"));
    if (attention.includes("sage2")) attentionSelect.append(option("sage2", "SageAttention 2"));
    const upscaler = capabilities.features?.upscaler;
    $("#hiresMethod option[value='neural']").disabled = upscaler?.backend !== "spandrel";
    if (!capabilities.features?.hires) {
      $("#upscalerNote").textContent = "このエンジンでは高解像度仕上げを利用できません";
    } else if (upscaler?.backend === "spandrel") {
      $("#upscalerNote").textContent = `${upscaler.model || "ローカルモデル"} で拡大後、指定ステップで細部を再描画します`;
    } else {
      const reason = upscaler?.reason ? `（${upscaler.reason}）` : "";
      $("#upscalerNote").textContent = `Lanczos 拡大後、指定ステップで細部を再描画します${reason}`;
    }

    const device = capabilities.device || {};
    const facts = [
      ["エンジン", [capabilities.engine, capabilities.version].filter(Boolean).join(" ") || "Krea 2 Engine"],
      ["デバイス", device.name || device.device_name || "取得できません"],
      ["VRAM", device.vram_bytes != null ? formatMemory(Number(device.vram_bytes) / (1024 * 1024)) : formatMemory(device.vram_total_mb ?? device.total_memory_mb)],
      ["精度", device.dtype || engineState?.dtype || "エンジン既定"],
      ["アテンション", engineState?.attention_backend || "自動"],
      ["キュー上限", capabilities.queue?.max_pending != null ? `${capabilities.queue.max_pending} 件` : "取得できません"],
    ];
    renderDefinitionList($("#factsList"), facts);

    const details = $("#optimizationDetails");
    details.replaceChildren();
    appendParagraph(details, attention.length ? `利用可能なアテンション: ${attention.join("、")}` : "標準のアテンションを使用します。");
    const vc = capabilities.features?.vc_attention;
    appendParagraph(details, vc?.available ? "VC Attention は検証済みで利用できます。" : `VC Attention は利用できません。${vc?.reason ? ` ${vc.reason}` : "独立実装と数値検証が完了していないためです。"}`);
    appendParagraph(details, "選択した最適化と実際に使われたバックエンドは、生成画像のメタデータに記録されます。");
  }

  function renderOfflineFacts() {
    renderDefinitionList($("#factsList"), [
      ["接続", "ローカルエンジンから応答がありません"],
      ["対処", "ランチャーでエンジンを起動して再接続してください"],
    ]);
    const details = $("#optimizationDetails");
    details.replaceChildren();
    appendParagraph(details, "稼働中の最適化は、エンジン接続後にここへ表示されます。VC Attention は独立実装と検証が確認できるまで有効として扱いません。");
  }

  function formatMemory(value) {
    if (value == null || Number.isNaN(Number(value))) return "取得できません";
    const mb = Number(value);
    return mb >= 1024 ? `${(mb / 1024).toFixed(1)} GB` : `${mb.toFixed(0)} MB`;
  }

  function renderDefinitionList(root, entries) {
    root.replaceChildren();
    entries.forEach(([term, description]) => {
      const row = document.createElement("div");
      const dt = document.createElement("dt");
      const dd = document.createElement("dd");
      dt.textContent = term;
      dd.textContent = String(description);
      row.append(dt, dd);
      root.append(row);
    });
  }

  function appendParagraph(root, content) {
    const paragraph = document.createElement("p");
    paragraph.textContent = content;
    root.append(paragraph);
  }

  function addLoraRow(id = "", weight = 1, enabled = true) {
    if (!state.loras.length) {
      showError("利用できる LoRA が見つかりません");
      return null;
    }
    const fragment = $("#loraRowTemplate").content.cloneNode(true);
    const row = $(".lora-row", fragment);
    const select = $(".lora-select", fragment);
    const preset = selectedPreset();
    state.loras.forEach((lora) => {
      const suffix = lora.available ? "" : ` — ${lora.reason || "非対応"}`;
      select.append(option(lora.id, `${lora.name}${suffix}`, !lora.available || !loraAllowed(lora, preset)));
    });
    const selected = state.loras.find((lora) => lora.id === id && lora.available && loraAllowed(lora, preset))
      || state.loras.find((lora) => lora.available && loraAllowed(lora, preset));
    if (!selected) {
      showError(preset === "fast4" ? "Turbo 4 で利用できる LoRA がありません" : "このプリセットで利用できるスタイル LoRA がありません");
      return null;
    }
    select.value = selected.id;
    $(".lora-weight", fragment).value = String(weight);
    $(".lora-enabled", fragment).checked = enabled;
    $(".remove-lora", fragment).addEventListener("click", () => {
      row.remove();
      noteLoraSelectionChanged();
    });
    select.addEventListener("change", noteLoraSelectionChanged);
    $(".lora-weight", fragment).addEventListener("input", noteLoraSelectionChanged);
    $(".lora-enabled", fragment).addEventListener("change", noteLoraSelectionChanged);
    const handle = $(".drag-handle", fragment);
    handle.addEventListener("keydown", (event) => moveLoraWithKeyboard(event, row));
    row.draggable = true;
    row.addEventListener("dragstart", () => row.classList.add("is-dragging"));
    row.addEventListener("dragend", () => {
      row.classList.remove("is-dragging");
      noteLoraSelectionChanged();
    });
    row.addEventListener("dragover", reorderLoraOnDrag);
    loraList.append(fragment);
    return row;
  }

  function loraAllowed(lora, preset) {
    return lora.category !== "distillation";
  }

  function syncLoraCompatibility(preset) {
    $$(".lora-row", loraList).forEach((row) => {
      const selectedLora = state.loras.find((lora) => lora.id === $(".lora-select", row).value);
      if (selectedLora?.category === "distillation" && !loraAllowed(selectedLora, preset)) {
        row.remove();
        return;
      }
      [...$(".lora-select", row).options].forEach((node) => {
        const lora = state.loras.find((item) => item.id === node.value);
        if (lora) node.disabled = !lora.available || !loraAllowed(lora, preset);
      });
    });
  }

  function moveLoraWithKeyboard(event, row) {
    if (event.key !== "ArrowUp" && event.key !== "ArrowDown") return;
    event.preventDefault();
    if (event.key === "ArrowUp" && row.previousElementSibling) loraList.insertBefore(row, row.previousElementSibling);
    if (event.key === "ArrowDown" && row.nextElementSibling) loraList.insertBefore(row.nextElementSibling, row);
    $(".drag-handle", row).focus();
    noteLoraSelectionChanged();
  }

  function reorderLoraOnDrag(event) {
    event.preventDefault();
    const dragging = $(".is-dragging", loraList);
    if (!dragging || dragging === event.currentTarget) return;
    const target = event.currentTarget;
    const rect = target.getBoundingClientRect();
    loraList.insertBefore(dragging, event.clientY < rect.top + rect.height / 2 ? target : target.nextSibling);
  }

  function selectedPreset() {
    return $("input[name='preset']:checked")?.value || "turbo8";
  }

  function applyPreset(preset) {
    const values = preset === "fast4" ? [4, 0] : preset === "raw" ? [28, 4] : [8, 0];
    $("#steps").value = values[0];
    $("#guidance").value = values[1];
    $("#steps").disabled = preset === "fast4";
    $("#steps").title = preset === "fast4" ? "Turbo 4 は4ステップ固定です" : "";
    $("#guidance").disabled = preset !== "raw";
    $("#guidance").title = preset !== "raw" ? "Turbo はガイダンス0固定です" : "";
    syncLoraCompatibility(preset);
    syncModelCompatibility(preset);
  }

  function applyStoredSettings(settings) {
    if (!settings || typeof settings !== "object") return;
    if (settings.width) $("#width").value = settings.width;
    if (settings.height) $("#height").value = settings.height;
    syncCanvasIndicators();
    const preset = $(`input[name="preset"][value="${safeSelectorValue(settings.preset || "turbo8")}"]`);
    if (preset && !preset.disabled) {
      preset.checked = true;
      applyPreset(preset.value);
    }
    if (settings.attention_backend && $(`#attentionBackend option[value="${safeSelectorValue(settings.attention_backend)}"]`)) {
      $("#attentionBackend").value = settings.attention_backend;
    }
    const hires = settings.hires || {};
    if (hires.scale) $("#upscaleFactor").value = hires.scale;
    if (hires.method && $(`#hiresMethod option[value="${safeSelectorValue(hires.method)}"]`)) $("#hiresMethod").value = hires.method;
    if (hires.refine_steps) $("#refineSteps").value = hires.refine_steps;
    if (hires.denoise_strength) $("#refineStrength").value = hires.denoise_strength;
  }

  function collectRequest() {
    const preset = selectedPreset();
    const rawSeed = Number($("#seed").value);
    return {
      prompt: $("#prompt").value.trim(),
      negative_prompt: $("#negativePrompt").value.trim(),
      width: Number($("#width").value),
      height: Number($("#height").value),
      seed: rawSeed < 0 ? null : rawSeed,
      preset,
      model_id: modelSelect.value,
      steps: Number($("#steps").value),
      guidance_scale: Number($("#guidance").value),
      loras: collectLoraSelection(),
      attention_backend: $("#attentionBackend").value,
    };
  }

  function validateRequest(request) {
    if (!request.prompt) return "プロンプトを入力してください";
    if (!request.model_id) return "利用するモデルを選択してください";
    if (request.width % 16 || request.height % 16) return "幅と高さは16の倍数にしてください";
    if (request.preset === "fast4" && !state.fast4LoraId) return "Turbo 4 に必要な蒸留 LoRA が見つかりません";
    const duplicateIds = request.loras.filter((item) => item.enabled).map((item) => item.id);
    if (new Set(duplicateIds).size !== duplicateIds.length) return "同じ LoRA を複数回有効にはできません";
    return null;
  }

  async function submitForm(event) {
    event.preventDefault();
    if (state.mode === "upscale") {
      await upscaleSelected();
      return;
    }
    await generate();
  }

  async function generate() {
    clearError();
    window.clearTimeout(state.loraApplyTimer);
    const request = collectRequest();
    const validationError = validateRequest(request);
    if (validationError) {
      showError(validationError);
      return;
    }
    try {
      const job = await api("/api/generate", { method: "POST", body: request, timeout: 30000 });
      state.activeJobMode = "generate";
      updateJobLanguage("generate");
      api("/api/settings", {
        method: "PUT",
        body: {
          width: request.width,
          height: request.height,
          preset: request.preset,
          attention_backend: request.attention_backend,
          hires: { enabled: false, ...collectUpscaleSettings() },
        },
      }).catch(() => {});
      state.activeJobId = job.job_id;
      state.generationStartedAt = Date.now();
      showProgress({ status: job.status, progress: 0, stage: "queued", message: "生成キューに追加しました" });
      $("#queuePosition").textContent = job.position > 0 ? `待機 ${job.position}` : "";
      $("#activeJobPrompt").textContent = request.prompt;
      $("#activeJob").hidden = false;
      updateGenerateAvailability();
      startPolling();
    } catch (error) {
      showError(error.message);
    }
  }

  function collectUpscaleSettings() {
    return {
      scale: Number($("#upscaleFactor").value),
      method: $("#hiresMethod").value,
      refine_steps: Number($("#refineSteps").value),
      denoise_strength: Number($("#refineStrength").value),
    };
  }

  function collectLoraSelection() {
    return $$(".lora-row", loraList).map((row) => ({
      id: $(".lora-select", row).value,
      weight: Number($(".lora-weight", row).value),
      enabled: $(".lora-enabled", row).checked,
    }));
  }

  function controlRequest(revision) {
    return {
      model_id: modelSelect.value,
      preset: selectedPreset(),
      attention_backend: $("#attentionBackend").value,
      loras: collectLoraSelection(),
      selection_revision: revision,
    };
  }

  async function loadSelectedModel() {
    if (!state.connected || !modelSelect.value || state.activeJobId || state.controlJobs.size) return;
    window.clearTimeout(state.loraApplyTimer);
    state.selectionRevision += 1;
    state.pendingLoraApply = false;
    state.controlError = null;
    const revision = state.selectionRevision;
    await submitControlJob("/api/load-model", "model_load", revision, controlRequest(revision));
  }

  function noteLoraSelectionChanged() {
    state.selectionRevision += 1;
    state.pendingLoraApply = true;
    state.controlError = null;
    window.clearTimeout(state.loraApplyTimer);
    updateModelRuntimeUI();
    if (!canAutoApplyLoras()) return;
    state.loraApplyTimer = window.setTimeout(applySelectedLoras, 500);
  }

  function canAutoApplyLoras() {
    const modelLoadActive = [...state.controlJobs.values()].some((job) => job.operation === "model_load");
    return state.connected
      && !state.activeJobId
      && !modelLoadActive
      && modelSelect.value === state.loadedModelId
      && selectedPreset() === state.loadedPreset
      && attentionSelectionMatchesLoaded();
  }

  async function applySelectedLoras() {
    if (!state.pendingLoraApply || !canAutoApplyLoras()) {
      updateModelRuntimeUI();
      return;
    }
    const revision = state.selectionRevision;
    state.pendingLoraApply = false;
    await submitControlJob("/api/load-loras", "loras_load", revision, controlRequest(revision));
  }

  async function submitControlJob(path, operation, revision, request) {
    try {
      const response = await api(path, { method: "POST", body: request, timeout: 30000 });
      const jobRevision = Number(response.selection_revision) || revision;
      state.controlJobs.set(response.job_id, { operation, revision: jobRevision, modelId: request.model_id });
      updateModelRuntimeUI();
      pollControlJob(response.job_id);
    } catch (error) {
      if (revision === state.selectionRevision) state.controlError = error.message;
      if (operation === "loras_load") state.pendingLoraApply = true;
      updateModelRuntimeUI();
    }
  }

  function trackControlJob(jobId, revision, operation) {
    state.controlJobs.set(jobId, { operation, revision, modelId: modelSelect.value });
    updateModelRuntimeUI();
    pollControlJob(jobId);
  }

  async function pollControlJob(jobId) {
    const tracked = state.controlJobs.get(jobId);
    if (!tracked) return;
    try {
      const job = await api(`/api/jobs/${encodeURIComponent(jobId)}`, { timeout: 10000 });
      const revision = Number(job.selection_revision ?? job.result?.selection_revision ?? tracked.revision) || 0;
      if (job.status === "completed") {
        if (revision >= state.appliedRevision) {
          const result = job.result || {};
          state.loadedModelId = result.model_id || tracked.modelId || state.loadedModelId;
          state.loadedPreset = result.preset || state.loadedPreset;
          state.loadedAttentionBackend = result.attention_backend || state.loadedAttentionBackend;
          state.loadedLoras = Array.isArray(result.loras) ? result.loras : state.loadedLoras;
          state.appliedRevision = revision;
        }
        state.controlJobs.delete(jobId);
        state.controlPollTimers.delete(jobId);
        if (revision === state.selectionRevision) {
          state.pendingLoraApply = false;
          state.controlError = null;
        }
        updateModelRuntimeUI();
        if (state.pendingLoraApply && canAutoApplyLoras()) {
          window.clearTimeout(state.loraApplyTimer);
          state.loraApplyTimer = window.setTimeout(applySelectedLoras, 150);
        }
        return;
      }
      if (job.status === "failed" || job.status === "cancelled") {
        state.controlJobs.delete(jobId);
        state.controlPollTimers.delete(jobId);
        let errorMessage = null;
        if (revision === state.selectionRevision) {
          errorMessage = job.status === "cancelled" ? "設定の適用を中止しました" : (job.error?.message || job.message || "設定を適用できませんでした");
          state.controlError = errorMessage;
          state.pendingLoraApply = tracked.operation === "loras_load";
        }
        updateModelRuntimeUI();
        api("/api/state").then((engineState) => {
          syncLoadedState(engineState);
          if (errorMessage && revision === state.selectionRevision) state.controlError = errorMessage;
          updateModelRuntimeUI();
        }).catch(() => {});
        return;
      }
    } catch (error) {
      if (tracked.revision === state.selectionRevision) state.controlError = error.message;
      updateModelRuntimeUI();
    }
    const timer = window.setTimeout(() => pollControlJob(jobId), 600);
    state.controlPollTimers.set(jobId, timer);
  }

  async function upscaleSelected() {
    clearError();
    window.clearTimeout(state.loraApplyTimer);
    const source = state.upscaleSourceItem;
    const sourceUrl = source?.result?.image_url || source?.image_url;
    if (!sourceUrl) {
      showError("Hiresに使用するソース画像を履歴から選んでください");
      return;
    }
    const request = {
      source_image: sourceUrl,
      ...collectUpscaleSettings(),
      attention_backend: $("#attentionBackend").value,
      model_id: modelSelect.value || undefined,
      prompt: $("#prompt").value.trim() || undefined,
      negative_prompt: $("#negativePrompt").value.trim() || undefined,
      seed: source.result?.seed ?? source.seed ?? undefined,
      loras: collectLoraSelection(),
    };
    try {
      const job = await api("/api/upscale", { method: "POST", body: request, timeout: 30000 });
      state.activeJobId = job.job_id;
      state.activeJobMode = "upscale";
      updateJobLanguage("upscale");
      state.generationStartedAt = Date.now();
      showProgress({ status: job.status, progress: 0, stage: "queued", message: "Hiresキューに追加しました" });
      $("#queuePosition").textContent = job.position > 0 ? `待機 ${job.position}` : "";
      $("#activeJobPrompt").textContent = request.prompt || "Hiresアップスケール";
      $("#activeJob").hidden = false;
      updateGenerateAvailability();
      startPolling();
    } catch (error) {
      showError(error.message);
    }
  }

  function startPolling() {
    window.clearTimeout(state.pollTimer);
    pollJob();
  }

  async function pollJob() {
    if (!state.activeJobId) return;
    try {
      const job = await api(`/api/jobs/${encodeURIComponent(state.activeJobId)}`, { timeout: 10000 });
      if (!state.activeJobMode) state.activeJobMode = job.operation === "upscale" || job.request?.source_image ? "upscale" : "generate";
      updateJobLanguage(state.activeJobMode);
      showProgress(job);
      if (job.request?.prompt) {
        $("#activeJobPrompt").textContent = job.request.prompt;
        $("#activeJob").hidden = false;
      }
      if (job.status === "completed") {
        finishJob(job);
        return;
      }
      if (job.status === "failed" || job.status === "cancelled") {
        endActiveJob();
        if (job.status === "failed") showError(job.error?.message || job.message || "生成に失敗しました");
        return;
      }
    } catch (error) {
      showError(error.message);
    }
    state.pollTimer = window.setTimeout(pollJob, 600);
  }

  function showProgress(job) {
    generationOverlay.hidden = false;
    const progress = Math.max(0, Math.min(1, Number(job.progress) || 0));
    const percent = Math.round(progress * 100);
    const titles = { queued: "順番を待っています", loading: "モデルを準備しています", running: "イメージを生成中", base: "イメージを生成中", hires_upscale: "画像を拡大しています", hires_refine: "細部を仕上げています", refining: "細部を仕上げています" };
    $("#progressTitle").textContent = titles[job.stage] || titles[job.status] || "生成しています";
    $("#progressDetail").textContent = job.message || "処理を続けています";
    $("#progressBar").style.width = `${percent}%`;
    $("#progressTrack").setAttribute("aria-valuenow", String(percent));
    $("#progressPercent").textContent = `${percent}%`;
    if (progress > .03 && progress < 1 && state.generationStartedAt) {
      const elapsed = (Date.now() - state.generationStartedAt) / 1000;
      const remaining = Math.max(0, elapsed * (1 - progress) / progress);
      $("#progressEta").textContent = remaining < 3600 ? `残り約 ${formatDuration(remaining)}` : "";
    } else {
      $("#progressEta").textContent = "";
    }
  }

  function finishJob(job) {
    showImage({ ...job, result: job.result, request: job.request });
    endActiveJob();
    loadHistory();
  }

  function endActiveJob() {
    window.clearTimeout(state.pollTimer);
    state.pollTimer = null;
    state.activeJobId = null;
    state.activeJobMode = null;
    generationOverlay.hidden = true;
    $("#activeJob").hidden = true;
    updateGenerateAvailability();
    api("/api/state").then((engineState) => {
      syncLoadedState(engineState);
      if (state.pendingLoraApply && canAutoApplyLoras()) {
        window.clearTimeout(state.loraApplyTimer);
        state.loraApplyTimer = window.setTimeout(applySelectedLoras, 150);
      }
    }).catch(() => {});
  }

  async function cancelJob() {
    if (!state.activeJobId) return;
    const button = $("#cancelButton");
    button.disabled = true;
    button.textContent = state.activeJobMode === "upscale" ? "アップスケールを中止しています…" : "生成を中止しています…";
    try {
      await api(`/api/jobs/${encodeURIComponent(state.activeJobId)}/cancel`, { method: "POST" });
    } catch (error) {
      showError(error.message);
    } finally {
      button.disabled = false;
      updateJobLanguage(state.activeJobMode || state.mode);
    }
  }

  function updateJobLanguage(mode) {
    const isUpscale = mode === "upscale";
    $("#activeJobLabel").textContent = isUpscale ? "アップスケール中" : "生成中";
    $("#cancelButton").textContent = isUpscale ? "アップスケールを中止" : "生成を中止";
  }

  async function loadHistory() {
    try {
      const response = await api("/api/history?limit=30");
      renderHistory(response.items || []);
    } catch (error) {
      if (state.connected) showError(`履歴を読み込めません: ${error.message}`);
    }
  }

  function renderHistory(items) {
    const root = $("#historyList");
    root.replaceChildren();
    if (!items.length) {
      const empty = document.createElement("div");
      empty.className = "history-empty";
      const mark = document.createElement("span");
      mark.setAttribute("aria-hidden", "true");
      mark.textContent = "◌";
      empty.append(mark);
      const text = document.createElement("p");
      text.textContent = "生成履歴はまだありません";
      empty.append(text);
      root.append(empty);
      return;
    }
    items.forEach((item) => {
      if (!item.result?.image_url) return;
      const wrap = document.createElement("div");
      wrap.className = "history-item-wrap";
      const button = document.createElement("button");
      button.type = "button";
      button.className = "history-item";
      button.title = item.request?.prompt || "生成画像";
      const image = document.createElement("img");
      image.src = item.result.image_url;
      image.alt = item.request?.prompt ? `生成画像: ${truncate(item.request.prompt, 70)}` : "生成画像";
      image.loading = "lazy";
      const badge = document.createElement("span");
      badge.className = "history-badge";
      badge.textContent = `${item.result.width || "?"}×${item.result.height || "?"}`;
      button.append(image, badge);
      button.addEventListener("click", () => showImage(item));
      const hiresButton = document.createElement("button");
      hiresButton.type = "button";
      hiresButton.className = "history-hires-button";
      hiresButton.textContent = "Hires";
      hiresButton.setAttribute("aria-label", "この画像をHiresアップスケールで開く");
      hiresButton.addEventListener("click", () => openItemInHires(item));
      wrap.append(button, hiresButton);
      root.append(wrap);
    });
  }

  async function showImage(item) {
    state.selectedHistoryItem = item;
    const result = item.result || item;
    if (!result.image_url) return;
    previewImage.src = result.image_url;
    previewImage.hidden = false;
    emptyStage.hidden = true;
    $("#stageActions").hidden = false;
    $("#stageCaption").hidden = false;
    $("#stageFilename").textContent = filenameFromUrl(result.image_url);
    $("#stageMeta").textContent = [result.width && result.height ? `${result.width}×${result.height}` : "", result.elapsed_seconds != null ? formatDuration(result.elapsed_seconds) : ""].filter(Boolean).join(" · ");
    $("#captionPrompt").textContent = item.request?.prompt || "";
    $("#downloadImage").href = result.image_url;
    $("#downloadImage").download = filenameFromUrl(result.image_url);
    renderMetadata(item);
    if (result.metadata_url && !item.metadata) {
      try {
        const metadata = await api(result.metadata_url);
        item.metadata = metadata;
        if (state.selectedHistoryItem === item) renderMetadata(item);
      } catch (_) { /* Sidecar is redundant; the image remains usable. */ }
    }
  }

  function setMode(mode) {
    state.mode = mode === "upscale" ? "upscale" : "generate";
    const isUpscale = state.mode === "upscale";
    $("#generationControls").hidden = isUpscale;
    $("#upscaleControls").hidden = !isUpscale;
    $("#generateModeTab").setAttribute("aria-selected", String(!isUpscale));
    $("#upscaleModeTab").setAttribute("aria-selected", String(isUpscale));
    $("#openInHires").hidden = isUpscale;
    $("#prompt").required = !isUpscale;
    modelSelect.required = !isUpscale;
    $("#panelTitle").textContent = isUpscale ? "Hiresアップスケール" : "新しいイメージ";
    $("#resetForm").textContent = isUpscale ? "ソース解除" : "リセット";
    clearError();
    updateGenerateAvailability();
  }

  function setUpscaleSource(item) {
    state.upscaleSourceItem = item;
    const result = item?.result || item;
    const request = item?.metadata || item?.request || {};
    const hasSource = Boolean(result?.image_url);
    $("#sourceEmpty").hidden = hasSource;
    $("#sourceCard").hidden = !hasSource;
    $("#sourceSettings").hidden = !hasSource;
    if (!hasSource) {
      $("#sourceThumbnail").removeAttribute("src");
      updateGenerateAvailability();
      return;
    }
    $("#sourceThumbnail").src = result.image_url;
    $("#sourceFilename").textContent = filenameFromUrl(result.image_url);
    $("#sourceDimensions").textContent = result.width && result.height ? `${result.width} × ${result.height}` : "サイズ情報なし";
    $("#sourcePrompt").textContent = request.prompt || "画像メタデータから継承";
    $("#sourceModel").textContent = request.model_id || request.model?.id || "画像メタデータから継承";
    const styles = (request.loras || []).filter((lora) => {
      const known = state.loras.find((entry) => entry.id === lora.id);
      return known?.category !== "distillation";
    });
    $("#sourceLoras").textContent = request.loras == null
      ? "画像メタデータから継承"
      : (styles.length ? styles.map((lora) => `${lora.id} (${lora.weight})`).join(" → ") : "なし");
    updateGenerateAvailability();
  }

  function applyItemSettings(item) {
    const request = item?.metadata || item?.request;
    if (!request) return false;
    $("#prompt").value = request.prompt || "";
    $("#negativePrompt").value = request.negative_prompt || "";
    $("#width").value = request.width || 1024;
    $("#height").value = request.height || 1024;
    syncCanvasIndicators();
    $("#seed").value = item?.result?.seed ?? request.seed ?? -1;
    const modelId = request.model_id || request.model?.id;
    if (modelId && state.models.some((model) => model.id === modelId && model.available)) modelSelect.value = modelId;
    const presetInput = $(`input[name="preset"][value="${safeSelectorValue(request.preset || "turbo8")}"]`);
    if (presetInput && !presetInput.disabled) presetInput.checked = true;
    if (request.attention_backend && $(`#attentionBackend option[value="${safeSelectorValue(request.attention_backend)}"]`)) $("#attentionBackend").value = request.attention_backend;
    loraList.replaceChildren();
    (request.loras || []).filter((lora) => state.loras.find((known) => known.id === lora.id)?.category !== "distillation")
      .forEach((lora) => addLoraRow(lora.id, lora.weight, lora.enabled));
    applyPreset(selectedPreset());
    if (selectedPreset() !== "fast4" && request.steps != null) $("#steps").value = request.steps;
    if (!(["turbo8", "fast4"].includes(selectedPreset())) && request.guidance_scale != null) $("#guidance").value = request.guidance_scale;
    const hires = request.hires || {};
    if (hires.scale) $("#upscaleFactor").value = hires.scale;
    if (hires.method && $(`#hiresMethod option[value="${safeSelectorValue(hires.method)}"]`)) $("#hiresMethod").value = hires.method;
    if (hires.refine_steps) $("#refineSteps").value = hires.refine_steps;
    if (hires.denoise_strength) $("#refineStrength").value = hires.denoise_strength;
    updatePromptCount();
    updateModelRuntimeUI();
    return true;
  }

  async function openItemInHires(item = state.selectedHistoryItem) {
    if (!item?.result?.image_url) {
      setMode("upscale");
      showError("右側の履歴からHiresに使用する画像を選んでください");
      return;
    }
    state.selectedHistoryItem = item;
    await showImage(item);
    applyItemSettings(item);
    setUpscaleSource(item);
    setMode("upscale");
    if (window.innerWidth <= 760 || window.innerHeight <= 400) form.scrollIntoView({ behavior: "smooth", block: "start" });
    else $("#controlScroll").scrollTo({ top: 0, behavior: "smooth" });
  }

  function renderMetadata(item) {
    const result = item.result || item;
    const request = item.request || item.metadata?.request || {};
    const data = item.metadata || {};
    const loras = request.loras?.filter((lora) => lora.enabled !== false).map((lora) => `${lora.id} (${lora.weight})`).join(" → ") || "なし";
    const upscale = data.passes?.find((pass) => pass.kind === "hires_refine")?.upscaler;
    const upscaleLabel = typeof upscale === "object" && upscale
      ? `${upscale.backend || "不明"}${upscale.model ? ` · ${upscale.model}` : ""}`
      : "なし";
    renderDefinitionList($("#metadataPanel"), [
      ["Seed", result.seed ?? data.seed ?? "—"],
      ["Preset", request.preset || data.preset || "—"],
      ["Steps / Guidance", `${request.steps ?? data.steps ?? "—"} / ${request.guidance_scale ?? data.guidance_scale ?? "—"}`],
      ["Model", request.model_id || request.model?.id || data.model_id || data.model?.id || "—"],
      ["LoRA", loras],
      ["Backend", data.attention_backend || data.backend || "メタデータを参照"],
      ["Upscale", upscaleLabel],
    ]);
  }

  function reuseSelectedSettings() {
    if (!applyItemSettings(state.selectedHistoryItem)) {
      showError("この画像から再利用できる設定が見つかりません");
      return;
    }
    setMode("generate");
    clearError();
    if (window.innerWidth <= 760 || window.innerHeight <= 400) form.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  async function openOutputFolder() {
    try {
      await api("/api/open-output-folder", { method: "POST" });
    } catch (error) {
      showError(error.status === 501 ? "この環境では保存先フォルダーを直接開けません" : error.message);
    }
  }

  function updatePromptCount() {
    $("#promptCount").textContent = `${$("#prompt").value.length} / 2000`;
  }

  function syncCanvasIndicators() {
    const width = Number($("#width").value);
    const height = Number($("#height").value);
    const orientation = width === height ? "square" : (height > width ? "portrait" : "landscape");
    const orientationInput = $(`input[name="orientation"][value="${orientation}"]`);
    if (orientationInput) orientationInput.checked = true;
    const matchIndex = CANVAS_PRESETS[orientation].findIndex((preset) => preset.width === width && preset.height === height);
    if (matchIndex >= 0 && orientation !== "square") state.sizePresetIndex = matchIndex;
    renderSizePresets(orientation, matchIndex >= 0 ? matchIndex : null);
    const custom = $("#customSizeNote");
    custom.hidden = matchIndex >= 0;
    custom.textContent = matchIndex >= 0 ? "" : `カスタムサイズ · ${width} × ${height}`;
  }

  function renderSizePresets(orientation, selectedIndex) {
    const root = $("#sizePresetGrid");
    root.replaceChildren();
    root.classList.toggle("is-square", orientation === "square");
    CANVAS_PRESETS[orientation].forEach((preset, index) => {
      const label = document.createElement("label");
      const input = document.createElement("input");
      input.type = "radio";
      input.name = "size_preset";
      input.value = String(index);
      input.checked = index === selectedIndex;
      input.addEventListener("change", () => applySizePreset(orientation, index));
      const content = document.createElement("span");
      const ratio = document.createElement("strong");
      const dimensions = document.createElement("small");
      ratio.textContent = preset.ratio;
      dimensions.textContent = `${preset.width} × ${preset.height}`;
      content.append(ratio, dimensions);
      label.append(input, content);
      root.append(label);
    });
  }

  function applySizePreset(orientation, index) {
    const preset = CANVAS_PRESETS[orientation]?.[index];
    if (!preset) return;
    if (orientation !== "square") state.sizePresetIndex = index;
    $("#width").value = preset.width;
    $("#height").value = preset.height;
    $("#customSizeNote").hidden = true;
  }

  function applyOrientation(orientation) {
    const index = orientation === "square" ? 0 : Math.min(state.sizePresetIndex, CANVAS_PRESETS[orientation].length - 1);
    renderSizePresets(orientation, index);
    applySizePreset(orientation, index);
  }

  function resetForm() {
    form.reset();
    loraList.replaceChildren();
    applyOrientation("square");
    applyPreset("turbo8");
    $("#negativeWrap").hidden = true;
    $("#toggleNegative").setAttribute("aria-expanded", "false");
    updatePromptCount();
    clearError();
    noteLoraSelectionChanged();
  }

  function filenameFromUrl(url) {
    try { return decodeURIComponent(new URL(url, location.href).pathname.split("/").pop()) || "image.png"; }
    catch (_) { return "image.png"; }
  }
  function truncate(value, max) { return value.length <= max ? value : `${value.slice(0, max - 1)}…`; }
  function formatDuration(seconds) {
    const rounded = Math.max(0, Math.round(Number(seconds) || 0));
    if (rounded < 60) return `${rounded}秒`;
    return `${Math.floor(rounded / 60)}分${rounded % 60}秒`;
  }
  function safeSelectorValue(value) { return window.CSS?.escape ? CSS.escape(String(value)) : String(value).replace(/["\\]/g, "\\$&"); }

  form.addEventListener("submit", submitForm);
  $("#prompt").addEventListener("input", updatePromptCount);
  $("#toggleNegative").addEventListener("click", (event) => {
    const expanded = event.currentTarget.getAttribute("aria-expanded") === "true";
    event.currentTarget.setAttribute("aria-expanded", String(!expanded));
    $("#negativeWrap").hidden = expanded;
    if (!expanded) $("#negativePrompt").focus();
  });
  $("#advancedTrigger").addEventListener("click", (event) => {
    const expanded = event.currentTarget.getAttribute("aria-expanded") === "true";
    event.currentTarget.setAttribute("aria-expanded", String(!expanded));
    $("#advancedPanel").hidden = expanded;
  });
  $("#toggleMetadata").addEventListener("click", (event) => {
    const expanded = event.currentTarget.getAttribute("aria-expanded") === "true";
    event.currentTarget.setAttribute("aria-expanded", String(!expanded));
    $("#metadataPanel").hidden = expanded;
  });
  $("#addLora").addEventListener("click", () => {
    if (addLoraRow()) noteLoraSelectionChanged();
  });
  $("#loadModel").addEventListener("click", loadSelectedModel);
  $("#cancelButton").addEventListener("click", cancelJob);
  $("#reuseSettings").addEventListener("click", reuseSelectedSettings);
  $("#openInHires").addEventListener("click", () => openItemInHires());
  $("#generateModeTab").addEventListener("click", () => setMode("generate"));
  $("#upscaleModeTab").addEventListener("click", () => setMode("upscale"));
  $("#clearSource").addEventListener("click", () => setUpscaleSource(null));
  $("#chooseSource").addEventListener("click", () => {
    showError("右側の履歴にある画像の「Hires」を押してください");
    $("#historyHeading").scrollIntoView({ behavior: "smooth", block: "start" });
  });
  $("#openOutput").addEventListener("click", openOutputFolder);
  $("#refreshHistory").addEventListener("click", loadHistory);
  $("#refreshModels").addEventListener("click", connect);
  $("#reconnectBackend").addEventListener("click", connect);
  $("#resetForm").addEventListener("click", () => state.mode === "upscale" ? setUpscaleSource(null) : resetForm());
  $("#randomSeed").addEventListener("click", () => { $("#seed").value = Math.floor(Math.random() * 4294967296); });
  modelSelect.addEventListener("change", () => {
    const model = state.models.find((item) => item.id === modelSelect.value);
    updateModelNote(model);
    state.controlError = null;
    updateModelRuntimeUI();
  });
  $$("input[name='preset']").forEach((input) => input.addEventListener("change", () => {
    applyPreset(input.value);
    state.controlError = null;
    updateModelRuntimeUI();
  }));
  $("#attentionBackend").addEventListener("change", () => {
    state.controlError = null;
    updateModelRuntimeUI();
  });
  $$("input[name='orientation']").forEach((input) => input.addEventListener("change", () => applyOrientation(input.value)));
  $("#width").addEventListener("input", syncCanvasIndicators);
  $("#height").addEventListener("input", syncCanvasIndicators);
  $("#openSettings").addEventListener("click", () => $("#settingsDialog").showModal());
  document.addEventListener("keydown", (event) => {
    if ((event.ctrlKey || event.metaKey) && event.key === "Enter" && !generateButton.disabled) {
      event.preventDefault();
      form.requestSubmit();
    }
  });

  updatePromptCount();
  renderSizePresets("square", 0);
  connect();
})();
