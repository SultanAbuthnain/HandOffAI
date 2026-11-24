/**
 * AI Customer Support Widget (RAG + Human-in-the-Loop)
 * Self-contained Vanilla JS widget isolated via Shadow DOM.
 *
 * Usage:
 *   <script
 *     src="https://your-server.com/widget.js"
 *     data-title="Support Assistant"
 *     data-subtitle="AI + Live Human Support"
 *     data-color="#2563eb"
 *     data-position="right"
 *     data-greeting="Hi there! How can I help you today?"
 *   ></script>
 */
(function () {
  "use strict";

  // Prevent duplicate initialization if script is accidentally included twice
  if (window.__AI_SUPPORT_HITL_WIDGET_LOADED__) return;
  window.__AI_SUPPORT_HITL_WIDGET_LOADED__ = true;

  // ---------------------------------------------------------------------------
  // Configuration & Script Attribute Parsing
  // ---------------------------------------------------------------------------
  const currentScript =
    document.currentScript ||
    Array.from(document.getElementsByTagName("script")).find((s) =>
      (s.src || "").includes("widget.js")
    );

  function deriveServerBase() {
    if (currentScript && currentScript.getAttribute("data-server")) {
      return currentScript.getAttribute("data-server").replace(/\/+$/, "");
    }
    if (currentScript && currentScript.src) {
      try {
        const u = new URL(currentScript.src, window.location.href);
        return u.origin;
      } catch (_err) {
        // Fallback to current origin
      }
    }
    return window.location.origin;
  }

  function sanitizeCssColor(rawColor, fallback) {
    if (!rawColor) return fallback;
    const trimmed = String(rawColor).trim();
    if (/^(#[0-9a-fA-F]{3,8}|(?:rgb|hsl)a?\(\s*[0-9.,%\s\/]+\)|[a-zA-Z]{3,25})$/.test(trimmed)) {
      return trimmed;
    }
    return fallback;
  }

  const CONFIG = {
    serverBase: deriveServerBase(),
    title: (currentScript && currentScript.getAttribute("data-title")) || "Support Assistant",
    subtitle:
      (currentScript && currentScript.getAttribute("data-subtitle")) ||
      "Instant AI answers with human handoff",
    primaryColor: sanitizeCssColor(
      currentScript && currentScript.getAttribute("data-color"),
      "#2563eb"
    ),
    position:
      (currentScript && currentScript.getAttribute("data-position")) === "left"
        ? "left"
        : "right",
    greeting:
      (currentScript && currentScript.getAttribute("data-greeting")) ||
      "Hi! Ask me anything about our documentation, or click 'Human' anytime to reach our team.",
    visitorName:
      (currentScript && currentScript.getAttribute("data-visitor-name")) || "Visitor",
    suggestions: (function () {
      const raw = currentScript && currentScript.getAttribute("data-suggestions");
      if (raw === "") return [];
      if (raw) {
        return raw.split("|").map((s) => s.trim()).filter(Boolean).slice(0, 4);
      }
      return ["Pricing plans", "Refund policy", "Talk to a human"];
    })(),
  };

  const STORAGE_KEY_SESSION = "ai_support_hitl_session_id";
  const STORAGE_KEY_TOKEN = "ai_support_hitl_visitor_token";
  const STORAGE_KEY_OPEN = "ai_support_hitl_open";
  const STORAGE_KEY_ACTIVE = "ai_support_hitl_active";

  function generateUUID() {
    if (window.crypto && typeof window.crypto.randomUUID === "function") {
      return window.crypto.randomUUID();
    }
    return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, function (c) {
      const r = (Math.random() * 16) | 0;
      const v = c === "x" ? r : (r & 0x3) | 0x8;
      return v.toString(16);
    });
  }

  function getOrCreateSessionId() {
    try {
      let sid = window.localStorage.getItem(STORAGE_KEY_SESSION);
      if (!sid) {
        sid = generateUUID();
        window.localStorage.setItem(STORAGE_KEY_SESSION, sid);
      }
      return sid;
    } catch (_err) {
      return generateUUID();
    }
  }

  function setStoredSessionId(sid) {
    try {
      window.localStorage.setItem(STORAGE_KEY_SESSION, sid);
    } catch (_err) {
      // Ignore storage restrictions (e.g. sandboxed iframes)
    }
  }

  function getOrCreateVisitorToken() {
    try {
      let tok = window.localStorage.getItem(STORAGE_KEY_TOKEN);
      if (!tok) {
        tok = generateUUID();
        window.localStorage.setItem(STORAGE_KEY_TOKEN, tok);
      }
      return tok;
    } catch (_err) {
      return generateUUID();
    }
  }

  function setStoredVisitorToken(tok) {
    try {
      window.localStorage.setItem(STORAGE_KEY_TOKEN, tok);
    } catch (_err) {
      // Ignore storage restrictions
    }
  }

  function buildWebSocketUrl(sessionId, visitorToken) {
    const httpUrl = new URL(CONFIG.serverBase);
    const wsProtocol = httpUrl.protocol === "https:" ? "wss:" : "ws:";
    const params = new URLSearchParams({
      visitor_name: CONFIG.visitorName,
      visitor_token: visitorToken,
    });
    return `${wsProtocol}//${httpUrl.host}/ws/chat/${encodeURIComponent(sessionId)}?${params.toString()}`;
  }

  // ---------------------------------------------------------------------------
  // Widget State
  // ---------------------------------------------------------------------------
  const state = {
    sessionId: getOrCreateSessionId(),
    visitorToken: getOrCreateVisitorToken(),
    visitorEmail: null,
    isOpen: false,
    connected: false,
    sessionState: "BOT_ACTIVE", // BOT_ACTIVE | WAITING_FOR_HUMAN | HUMAN_ACTIVE
    assignedAgent: null,
    messages: [],
    messageIds: new Set(),
    streamBubbles: new Map(),
    unreadCount: 0,
    typing: { active: false, label: "" },
    ws: null,
    reconnectTimer: null,
    reconnectAttempts: 0,
    heartbeatTimer: null,
    errorTimer: null,
  };

  // ---------------------------------------------------------------------------
  // Shadow DOM Mount & Isolated Styles
  // ---------------------------------------------------------------------------
  const host = document.createElement("div");
  host.id = "ai-support-hitl-root";
  const shadow = host.attachShadow({ mode: "open" });

  const styles = document.createElement("style");
  styles.textContent = `
    :host {
      all: initial;
      position: fixed;
      bottom: 20px;
      ${CONFIG.position === "left" ? "left: 20px;" : "right: 20px;"}
      z-index: 2147483647;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Inter, Helvetica, Arial, sans-serif;
      font-size: 14px;
      line-height: 1.45;
      color: #0f172a;
      -webkit-font-smoothing: antialiased;
    }

    *, *::before, *::after {
      box-sizing: border-box;
      margin: 0;
      padding: 0;
    }

    .launcher {
      width: 58px;
      height: 58px;
      border-radius: 9999px;
      border: none;
      background: ${CONFIG.primaryColor};
      color: #ffffff;
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
      box-shadow: 0 10px 25px rgba(15, 23, 42, 0.22), 0 2px 6px rgba(15, 23, 42, 0.12);
      transition: transform 0.18s ease, box-shadow 0.18s ease;
      position: relative;
      margin-left: ${CONFIG.position === "right" ? "auto" : "0"};
    }

    .launcher:hover {
      transform: translateY(-2px) scale(1.03);
      box-shadow: 0 14px 30px rgba(15, 23, 42, 0.28);
    }

    .launcher:focus-visible {
      outline: 3px solid rgba(37, 99, 235, 0.45);
      outline-offset: 3px;
    }

    .unread-badge {
      position: absolute;
      top: -4px;
      right: -4px;
      min-width: 21px;
      height: 21px;
      padding: 0 6px;
      border-radius: 9999px;
      background: #ef4444;
      color: #ffffff;
      font-size: 11px;
      font-weight: 700;
      display: none;
      align-items: center;
      justify-content: center;
      border: 2px solid #ffffff;
    }

    .unread-badge.visible {
      display: flex;
    }

    .panel {
      width: 380px;
      max-width: calc(100vw - 24px);
      height: 560px;
      max-height: calc(100vh - 96px);
      margin-bottom: 14px;
      background: #ffffff;
      border-radius: 16px;
      box-shadow: 0 20px 50px rgba(15, 23, 42, 0.22), 0 0 0 1px rgba(15, 23, 42, 0.06);
      display: none;
      flex-direction: column;
      overflow: hidden;
      transform-origin: ${CONFIG.position === "left" ? "bottom left" : "bottom right"};
    }

    .panel.open {
      display: flex;
      animation: panelPop 0.18s ease-out;
    }

    @keyframes panelPop {
      from { opacity: 0; transform: translateY(10px) scale(0.97); }
      to   { opacity: 1; transform: translateY(0) scale(1); }
    }

    .header {
      background: ${CONFIG.primaryColor};
      color: #ffffff;
      padding: 14px 16px;
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: 10px;
      flex-shrink: 0;
    }

    .header-info {
      min-width: 0;
      flex: 1;
    }

    .header-title {
      font-size: 15px;
      font-weight: 700;
      letter-spacing: -0.01em;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }

    .header-subtitle {
      font-size: 12px;
      opacity: 0.88;
      margin-top: 2px;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }

    .header-actions {
      display: flex;
      align-items: center;
      gap: 4px;
      flex-shrink: 0;
    }

    .icon-btn {
      width: 30px;
      height: 30px;
      border-radius: 8px;
      border: none;
      background: rgba(255, 255, 255, 0.16);
      color: #ffffff;
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
      transition: background 0.15s ease;
    }

    .icon-btn:hover {
      background: rgba(255, 255, 255, 0.28);
    }

    .status-bar {
      background: #f8fafc;
      border-bottom: 1px solid #e2e8f0;
      padding: 8px 14px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
      flex-shrink: 0;
      font-size: 12px;
    }

    .status-pill {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      font-weight: 600;
      padding: 3px 9px;
      border-radius: 9999px;
      font-size: 11.5px;
    }

    .status-dot {
      width: 7px;
      height: 7px;
      border-radius: 9999px;
      flex-shrink: 0;
    }

    .status-pill.bot {
      background: #eff6ff;
      color: #1d4ed8;
    }
    .status-pill.bot .status-dot { background: #2563eb; }

    .status-pill.waiting {
      background: #fffbeb;
      color: #b45309;
    }
    .status-pill.waiting .status-dot {
      background: #f59e0b;
      animation: pulseDot 1.3s infinite;
    }

    .status-pill.human {
      background: #ecfdf5;
      color: #047857;
    }
    .status-pill.human .status-dot { background: #10b981; }

    .status-pill.offline {
      background: #f1f5f9;
      color: #64748b;
    }
    .status-pill.offline .status-dot { background: #94a3b8; }

    @keyframes pulseDot {
      0%, 100% { opacity: 1; transform: scale(1); }
      50% { opacity: 0.4; transform: scale(0.85); }
    }

    .human-btn {
      border: 1px solid #cbd5e1;
      background: #ffffff;
      color: #334155;
      font-size: 11.5px;
      font-weight: 600;
      padding: 4px 9px;
      border-radius: 7px;
      cursor: pointer;
      transition: all 0.15s ease;
      white-space: nowrap;
    }

    .human-btn:hover {
      background: #f1f5f9;
      border-color: #94a3b8;
      color: #0f172a;
    }

    .human-btn.hidden {
      display: none;
    }

    .error-banner {
      display: none;
      background: #fef2f2;
      color: #b91c1c;
      border-bottom: 1px solid #fecaca;
      padding: 8px 14px;
      font-size: 12px;
      font-weight: 500;
      flex-shrink: 0;
    }

    .error-banner.visible {
      display: block;
    }

    .messages {
      flex: 1;
      overflow-y: auto;
      padding: 14px;
      background: #f8fafc;
      display: flex;
      flex-direction: column;
      gap: 10px;
    }

    .msg-row {
      display: flex;
      flex-direction: column;
      max-width: 85%;
    }

    .msg-row.user {
      align-self: flex-end;
      align-items: flex-end;
    }

    .msg-row.bot, .msg-row.agent {
      align-self: flex-start;
      align-items: flex-start;
    }

    .msg-row.system {
      align-self: center;
      align-items: center;
      max-width: 92%;
    }

    .msg-meta {
      font-size: 10.5px;
      color: #64748b;
      margin-bottom: 3px;
      padding: 0 4px;
      font-weight: 500;
    }

    .bubble {
      padding: 9px 12px;
      border-radius: 14px;
      font-size: 13.5px;
      line-height: 1.45;
      word-wrap: break-word;
      white-space: pre-wrap;
    }

    .msg-row.user .bubble {
      background: ${CONFIG.primaryColor};
      color: #ffffff;
      border-bottom-right-radius: 4px;
    }

    .msg-row.bot .bubble {
      background: #ffffff;
      color: #0f172a;
      border: 1px solid #e2e8f0;
      border-bottom-left-radius: 4px;
    }

    .msg-row.agent .bubble {
      background: #ecfdf5;
      color: #064e3b;
      border: 1px solid #a7f3d0;
      border-bottom-left-radius: 4px;
    }

    .msg-row.system .bubble {
      background: #e2e8f0;
      color: #475569;
      font-size: 11.5px;
      padding: 5px 11px;
      border-radius: 9999px;
      text-align: center;
    }

    .bubble a {
      color: inherit;
      text-decoration: underline;
    }

    .bubble strong {
      font-weight: 700;
    }

    .bubble code {
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      font-size: 12px;
      background: rgba(15, 23, 42, 0.08);
      padding: 1px 5px;
      border-radius: 5px;
    }

    .msg-row.user .bubble code {
      background: rgba(255, 255, 255, 0.22);
    }

    .md-li {
      display: flex;
      align-items: flex-start;
      gap: 6px;
      margin-top: 3px;
    }

    .md-li-dot {
      flex-shrink: 0;
      font-weight: 700;
    }

    .suggestions {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      margin-top: 6px;
    }

    .suggestion-chip {
      border: 1px solid #cbd5e1;
      background: #ffffff;
      color: #1e293b;
      font-size: 12px;
      font-weight: 500;
      padding: 5px 10px;
      border-radius: 9999px;
      cursor: pointer;
      transition: all 0.15s ease;
    }

    .suggestion-chip:hover {
      border-color: ${CONFIG.primaryColor};
      color: ${CONFIG.primaryColor};
      background: #f8fafc;
    }

    .email-bar {
      display: none;
      background: #fffbeb;
      border-top: 1px solid #fde68a;
      padding: 8px 12px;
      gap: 6px;
      align-items: center;
      flex-shrink: 0;
    }

    .email-bar.visible {
      display: flex;
    }

    .email-bar input {
      flex: 1;
      min-width: 0;
      border: 1px solid #fcd34d;
      border-radius: 8px;
      padding: 6px 9px;
      font-size: 12px;
      font-family: inherit;
      background: #ffffff;
      color: #0f172a;
      outline: none;
    }

    .email-bar button {
      border: none;
      background: #d97706;
      color: #ffffff;
      font-size: 11.5px;
      font-weight: 600;
      padding: 6px 10px;
      border-radius: 8px;
      cursor: pointer;
      white-space: nowrap;
    }

    .email-bar button:hover {
      background: #b45309;
    }

    .typing-row {
      display: none;
      align-items: center;
      gap: 8px;
      padding: 6px 14px 10px;
      background: #f8fafc;
      font-size: 12px;
      color: #64748b;
      flex-shrink: 0;
    }

    .typing-row.visible {
      display: flex;
    }

    .dots {
      display: inline-flex;
      gap: 3px;
    }

    .dots span {
      width: 6px;
      height: 6px;
      border-radius: 9999px;
      background: #94a3b8;
      animation: bounceDot 1.2s infinite ease-in-out;
    }

    .dots span:nth-child(2) { animation-delay: 0.16s; }
    .dots span:nth-child(3) { animation-delay: 0.32s; }

    @keyframes bounceDot {
      0%, 80%, 100% { transform: translateY(0); opacity: 0.45; }
      40% { transform: translateY(-4px); opacity: 1; }
    }

    .composer {
      padding: 10px 12px;
      background: #ffffff;
      border-top: 1px solid #e2e8f0;
      display: flex;
      align-items: flex-end;
      gap: 8px;
      flex-shrink: 0;
    }

    .composer textarea {
      flex: 1;
      resize: none;
      border: 1px solid #cbd5e1;
      border-radius: 10px;
      padding: 9px 11px;
      font-family: inherit;
      font-size: 13.5px;
      line-height: 1.4;
      color: #0f172a;
      background: #f8fafc;
      max-height: 96px;
      min-height: 38px;
      outline: none;
      transition: border-color 0.15s ease, background 0.15s ease;
    }

    .composer textarea:focus {
      border-color: ${CONFIG.primaryColor};
      background: #ffffff;
    }

    .send-btn {
      width: 38px;
      height: 38px;
      border-radius: 10px;
      border: none;
      background: ${CONFIG.primaryColor};
      color: #ffffff;
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
      flex-shrink: 0;
      transition: opacity 0.15s ease, transform 0.12s ease;
    }

    .send-btn:hover:not(:disabled) {
      opacity: 0.92;
      transform: translateY(-1px);
    }

    .send-btn:disabled {
      opacity: 0.45;
      cursor: not-allowed;
    }

    @media (max-width: 440px) {
      :host {
        bottom: 12px;
        ${CONFIG.position === "left" ? "left: 12px;" : "right: 12px;"}
      }
      .panel {
        width: calc(100vw - 24px);
        height: calc(100vh - 96px);
      }
    }
  `;
  shadow.appendChild(styles);

  // ---------------------------------------------------------------------------
  // DOM Construction (inside Shadow Root)
  // ---------------------------------------------------------------------------
  const container = document.createElement("div");

  // Chat Panel
  const panel = document.createElement("div");
  panel.className = "panel";
  panel.setAttribute("role", "dialog");
  panel.setAttribute("aria-label", CONFIG.title);

  // Header
  const header = document.createElement("div");
  header.className = "header";

  const headerInfo = document.createElement("div");
  headerInfo.className = "header-info";
  const titleEl = document.createElement("div");
  titleEl.className = "header-title";
  titleEl.textContent = CONFIG.title;
  const subtitleEl = document.createElement("div");
  subtitleEl.className = "header-subtitle";
  subtitleEl.textContent = CONFIG.subtitle;
  headerInfo.append(titleEl, subtitleEl);

  const headerActions = document.createElement("div");
  headerActions.className = "header-actions";

  const resetBtn = document.createElement("button");
  resetBtn.className = "icon-btn";
  resetBtn.type = "button";
  resetBtn.title = "Start new conversation";
  resetBtn.setAttribute("aria-label", "Start new conversation");
  resetBtn.innerHTML =
    '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"/><path d="M3 3v5h5"/></svg>';

  const closeBtn = document.createElement("button");
  closeBtn.className = "icon-btn";
  closeBtn.type = "button";
  closeBtn.title = "Close chat";
  closeBtn.setAttribute("aria-label", "Close chat");
  closeBtn.innerHTML =
    '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round"><path d="M18 6 6 18M6 6l12 12"/></svg>';

  headerActions.append(resetBtn, closeBtn);
  header.append(headerInfo, headerActions);

  // Status Bar
  const statusBar = document.createElement("div");
  statusBar.className = "status-bar";

  const statusPill = document.createElement("span");
  statusPill.className = "status-pill offline";
  const statusDot = document.createElement("span");
  statusDot.className = "status-dot";
  const statusText = document.createElement("span");
  statusText.textContent = "Connecting...";
  statusPill.append(statusDot, statusText);

  const humanBtn = document.createElement("button");
  humanBtn.className = "human-btn";
  humanBtn.type = "button";
  humanBtn.textContent = "Talk to a human";

  statusBar.append(statusPill, humanBtn);

  // Error Banner
  const errorBanner = document.createElement("div");
  errorBanner.className = "error-banner";

  // Message List
  const messagesEl = document.createElement("div");
  messagesEl.className = "messages";
  messagesEl.setAttribute("aria-live", "polite");

  // Typing Indicator
  const typingRow = document.createElement("div");
  typingRow.className = "typing-row";
  typingRow.innerHTML = '<div class="dots"><span></span><span></span><span></span></div>';
  const typingLabel = document.createElement("span");
  typingRow.appendChild(typingLabel);

  // Email Follow-Up Capture Bar (shown when waiting for human & email not set)
  const emailBar = document.createElement("form");
  emailBar.className = "email-bar";
  const emailInput = document.createElement("input");
  emailInput.type = "email";
  emailInput.placeholder = "Optional: leave email for follow-up...";
  emailInput.setAttribute("aria-label", "Email for follow-up");
  const emailSubmitBtn = document.createElement("button");
  emailSubmitBtn.type = "submit";
  emailSubmitBtn.textContent = "Save";
  emailBar.append(emailInput, emailSubmitBtn);

  // Composer
  const composer = document.createElement("form");
  composer.className = "composer";

  const inputEl = document.createElement("textarea");
  inputEl.rows = 1;
  inputEl.placeholder = "Type your message...";
  inputEl.maxLength = 4000;
  inputEl.setAttribute("aria-label", "Chat message");

  const sendBtn = document.createElement("button");
  sendBtn.className = "send-btn";
  sendBtn.type = "submit";
  sendBtn.setAttribute("aria-label", "Send message");
  sendBtn.innerHTML =
    '<svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="m22 2-7 20-4-9-9-4Z"/><path d="M22 2 11 13"/></svg>';

  composer.append(inputEl, sendBtn);
  panel.append(header, statusBar, errorBanner, messagesEl, typingRow, emailBar, composer);

  // Floating Launcher Button
  const launcher = document.createElement("button");
  launcher.className = "launcher";
  launcher.type = "button";
  launcher.setAttribute("aria-label", "Open support chat");
  launcher.innerHTML =
    '<svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.1" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>';

  const unreadBadge = document.createElement("span");
  unreadBadge.className = "unread-badge";
  launcher.appendChild(unreadBadge);

  container.append(panel, launcher);
  shadow.appendChild(container);

  // ---------------------------------------------------------------------------
  // Rendering Helpers (XSS-safe: never injects raw HTML for message content)
  // ---------------------------------------------------------------------------
  function appendInlineSegments(parent, text) {
    const tokenRegex = /(https?:\/\/[^\s<]+|`[^`]+`|\*\*[^*]+\*\*)/g;
    const parts = String(text || "").split(tokenRegex);
    for (let i = 0; i < parts.length; i++) {
      const part = parts[i];
      if (!part) continue;
      if (/^https?:\/\//.test(part)) {
        const cleanUrl = part.replace(/[.,!?;:)\]'"]+$/, "");
        const trailing = part.slice(cleanUrl.length);
        const a = document.createElement("a");
        a.href = cleanUrl;
        a.target = "_blank";
        a.rel = "noopener noreferrer";
        a.textContent = cleanUrl;
        parent.appendChild(a);
        if (trailing) {
          parent.appendChild(document.createTextNode(trailing));
        }
      } else if (part.startsWith("`") && part.endsWith("`") && part.length > 2) {
        const code = document.createElement("code");
        code.textContent = part.slice(1, -1);
        parent.appendChild(code);
      } else if (part.startsWith("**") && part.endsWith("**") && part.length > 4) {
        const strong = document.createElement("strong");
        strong.textContent = part.slice(2, -2);
        parent.appendChild(strong);
      } else {
        parent.appendChild(document.createTextNode(part));
      }
    }
  }

  function renderFormattedMessage(parent, text) {
    parent.textContent = "";
    const lines = String(text || "").split("\n");
    for (let i = 0; i < lines.length; i++) {
      const line = lines[i];
      const bulletMatch = line.match(/^\s*[-*]\s+(.+)$/);
      if (bulletMatch) {
        const li = document.createElement("div");
        li.className = "md-li";
        const dot = document.createElement("span");
        dot.className = "md-li-dot";
        dot.textContent = "•";
        const body = document.createElement("span");
        appendInlineSegments(body, bulletMatch[1]);
        li.append(dot, body);
        parent.appendChild(li);
      } else {
        if (i > 0 && !lines[i - 1].match(/^\s*[-*]\s+/)) {
          parent.appendChild(document.createTextNode("\n"));
        }
        appendInlineSegments(parent, line);
      }
    }
  }

  function formatTime(isoString) {
    if (!isoString) return "";
    try {
      const d = new Date(isoString);
      return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
    } catch (_err) {
      return "";
    }
  }

  function renderStatus() {
    const showEmailBar =
      state.connected && state.sessionState === "WAITING_FOR_HUMAN" && !state.visitorEmail;
    emailBar.classList.toggle("visible", showEmailBar);

    if (!state.connected) {
      statusPill.className = "status-pill offline";
      statusText.textContent = "Reconnecting...";
      humanBtn.classList.add("hidden");
      return;
    }

    if (state.sessionState === "HUMAN_ACTIVE") {
      statusPill.className = "status-pill human";
      statusText.textContent = state.assignedAgent
        ? `Live Agent: ${state.assignedAgent}`
        : "Live Human Agent";
      humanBtn.classList.add("hidden");
    } else if (state.sessionState === "WAITING_FOR_HUMAN") {
      statusPill.className = "status-pill waiting";
      statusText.textContent = "Waiting for a human agent...";
      humanBtn.classList.add("hidden");
    } else {
      statusPill.className = "status-pill bot";
      statusText.textContent = "AI Assistant";
      humanBtn.classList.remove("hidden");
    }
  }

  function renderUnread() {
    if (state.unreadCount > 0 && !state.isOpen) {
      unreadBadge.textContent = state.unreadCount > 9 ? "9+" : String(state.unreadCount);
      unreadBadge.classList.add("visible");
    } else {
      unreadBadge.classList.remove("visible");
    }
  }

  function renderTyping() {
    if (state.typing.active) {
      typingLabel.textContent = state.typing.label || "Typing...";
      typingRow.classList.add("visible");
    } else {
      typingRow.classList.remove("visible");
    }
  }

  function showError(msg) {
    errorBanner.textContent = msg;
    errorBanner.classList.add("visible");
    if (state.errorTimer) clearTimeout(state.errorTimer);
    state.errorTimer = setTimeout(() => {
      errorBanner.classList.remove("visible");
    }, 5500);
  }

  function createMessageNode(msg) {
    const row = document.createElement("div");
    row.className = `msg-row ${msg.sender || "bot"}`;

    if (msg.sender !== "system") {
      const meta = document.createElement("div");
      meta.className = "msg-meta";
      let senderLabel = "You";
      if (msg.sender === "bot") senderLabel = "AI Assistant";
      else if (msg.sender === "agent") {
        senderLabel = (msg.metadata && msg.metadata.agent_name) || "Support Agent";
      }
      const timeStr = formatTime(msg.created_at);
      meta.textContent = timeStr ? `${senderLabel} · ${timeStr}` : senderLabel;
      row.appendChild(meta);
    }

    const bubble = document.createElement("div");
    bubble.className = "bubble";
    renderFormattedMessage(bubble, msg.content);
    row.appendChild(bubble);
    return row;
  }

  function scrollToBottom() {
    messagesEl.scrollTop = messagesEl.scrollHeight;
  }

  function renderAllMessages() {
    messagesEl.innerHTML = "";
    state.streamBubbles.clear();
    if (state.messages.length === 0) {
      messagesEl.appendChild(
        createMessageNode({
          sender: "bot",
          content: CONFIG.greeting,
          created_at: null,
        })
      );
      if (CONFIG.suggestions.length > 0) {
        const chipsWrap = document.createElement("div");
        chipsWrap.className = "suggestions";
        for (const label of CONFIG.suggestions) {
          const chip = document.createElement("button");
          chip.type = "button";
          chip.className = "suggestion-chip";
          chip.textContent = label;
          chip.addEventListener("click", () => {
            if (sendJson({ type: "message", content: label })) {
              chipsWrap.remove();
            }
          });
          chipsWrap.appendChild(chip);
        }
        messagesEl.appendChild(chipsWrap);
      }
    } else {
      for (const msg of state.messages) {
        messagesEl.appendChild(createMessageNode(msg));
      }
    }
    scrollToBottom();
  }

  function handleStreamDelta(streamId, delta) {
    if (!streamId || !delta) return;
    let entry = state.streamBubbles.get(streamId);
    if (!entry) {
      if (state.messages.length === 0) {
        messagesEl.innerHTML = "";
        messagesEl.appendChild(
          createMessageNode({
            sender: "bot",
            content: CONFIG.greeting,
            created_at: null,
          })
        );
      }
      const row = document.createElement("div");
      row.className = "msg-row bot";
      const meta = document.createElement("div");
      meta.className = "msg-meta";
      meta.textContent = "AI Assistant";
      const bubble = document.createElement("div");
      bubble.className = "bubble";
      row.append(meta, bubble);
      messagesEl.appendChild(row);
      entry = { row, meta, bubble, text: "" };
      state.streamBubbles.set(streamId, entry);
    }
    entry.text += delta;
    renderFormattedMessage(entry.bubble, entry.text);
    scrollToBottom();
  }

  function abortStreamBubble(streamId) {
    const entry = state.streamBubbles.get(streamId);
    if (entry) {
      entry.row.remove();
      state.streamBubbles.delete(streamId);
    }
  }

  function appendMessage(msg, streamId) {
    if (msg.id && state.messageIds.has(msg.id)) return;
    if (msg.id) state.messageIds.add(msg.id);

    if (streamId && state.streamBubbles.has(streamId)) {
      const entry = state.streamBubbles.get(streamId);
      state.streamBubbles.delete(streamId);
      state.messages.push(msg);
      const timeStr = formatTime(msg.created_at);
      entry.meta.textContent = timeStr ? `AI Assistant · ${timeStr}` : "AI Assistant";
      renderFormattedMessage(entry.bubble, msg.content);
      scrollToBottom();
      if (!state.isOpen && msg.sender !== "user") {
        state.unreadCount += 1;
        renderUnread();
      }
      return;
    }

    const wasEmpty = state.messages.length === 0;
    state.messages.push(msg);

    if (wasEmpty) {
      renderAllMessages();
    } else {
      messagesEl.appendChild(createMessageNode(msg));
      scrollToBottom();
    }

    if (!state.isOpen && msg.sender !== "user") {
      state.unreadCount += 1;
      renderUnread();
    }
  }

  // ---------------------------------------------------------------------------
  // WebSocket Lifecycle
  // ---------------------------------------------------------------------------
  function clearTimers() {
    if (state.reconnectTimer) {
      clearTimeout(state.reconnectTimer);
      state.reconnectTimer = null;
    }
    if (state.heartbeatTimer) {
      clearInterval(state.heartbeatTimer);
      state.heartbeatTimer = null;
    }
  }

  function connectWebSocket() {
    if (
      state.ws &&
      (state.ws.readyState === WebSocket.OPEN || state.ws.readyState === WebSocket.CONNECTING)
    ) {
      return;
    }

    clearTimers();
    const url = buildWebSocketUrl(state.sessionId, state.visitorToken);
    let ws;
    try {
      ws = new WebSocket(url);
    } catch (_err) {
      scheduleReconnect();
      return;
    }

    state.ws = ws;

    ws.onopen = function () {
      if (state.ws !== ws) return;
      state.connected = true;
      state.reconnectAttempts = 0;
      renderStatus();
      state.heartbeatTimer = setInterval(() => {
        if (ws.readyState === WebSocket.OPEN) {
          ws.send(JSON.stringify({ type: "ping" }));
        }
      }, 25000);
    };

    ws.onmessage = function (event) {
      if (state.ws !== ws) return;
      let data;
      try {
        data = JSON.parse(event.data);
      } catch (_err) {
        return;
      }

      switch (data.type) {
        case "session_init":
          if (data.session) {
            state.sessionState = data.session.state || "BOT_ACTIVE";
            state.assignedAgent = data.session.assigned_agent || null;
            state.visitorEmail = data.session.visitor_email || null;
          }
          state.messages = Array.isArray(data.messages) ? data.messages : [];
          state.messageIds = new Set(state.messages.map((m) => m.id).filter(Boolean));
          renderStatus();
          renderAllMessages();
          break;

        case "message_stream":
          handleStreamDelta(data.stream_id, data.delta);
          break;

        case "message_stream_abort":
          abortStreamBubble(data.stream_id);
          break;

        case "message":
          if (data.session) {
            state.sessionState = data.session.state || state.sessionState;
            state.assignedAgent = data.session.assigned_agent || null;
            state.visitorEmail = data.session.visitor_email || state.visitorEmail;
            renderStatus();
          }
          if (data.message) {
            appendMessage(data.message, data.stream_id);
          }
          break;

        case "state_change":
          if (data.session) {
            state.sessionState = data.session.state || state.sessionState;
            state.assignedAgent = data.session.assigned_agent || null;
            state.visitorEmail = data.session.visitor_email || state.visitorEmail;
            renderStatus();
          }
          break;

        case "typing":
          if (data.is_typing) {
            state.typing = {
              active: true,
              label:
                data.sender === "agent"
                  ? `${data.agent_name || "Support Agent"} is typing...`
                  : "AI is thinking...",
            };
          } else {
            state.typing = { active: false, label: "" };
          }
          renderTyping();
          break;

        case "error":
          showError(data.message || "Unable to send message.");
          break;

        case "pong":
        default:
          break;
      }
    };

    ws.onclose = function (evt) {
      if (state.ws !== ws) return;
      state.connected = false;
      state.typing = { active: false, label: "" };
      clearTimers();
      renderStatus();
      renderTyping();

      if (evt.code === 1008) {
        if ((evt.reason || "").toLowerCase().includes("token")) {
          const newId = generateUUID();
          const newToken = generateUUID();
          setStoredSessionId(newId);
          setStoredVisitorToken(newToken);
          state.sessionId = newId;
          state.visitorToken = newToken;
          state.messages = [];
          state.messageIds = new Set();
          scheduleReconnect();
          return;
        }
        showError("This website origin is not authorized to connect to the support server.");
        return;
      }
      scheduleReconnect();
    };

    ws.onerror = function () {
      // onclose handles reconnect cleanly
    };
  }

  function scheduleReconnect() {
    if (state.reconnectTimer) return;
    const delay = Math.min(1000 * Math.pow(2, state.reconnectAttempts), 15000);
    state.reconnectAttempts += 1;
    state.reconnectTimer = setTimeout(() => {
      state.reconnectTimer = null;
      connectWebSocket();
    }, delay);
  }

  function sendJson(payload) {
    if (!state.ws || state.ws.readyState !== WebSocket.OPEN) {
      showError("Not connected yet. Reconnecting...");
      connectWebSocket();
      return false;
    }
    state.ws.send(JSON.stringify(payload));
    return true;
  }

  // ---------------------------------------------------------------------------
  // UI Interactions
  // ---------------------------------------------------------------------------
  function setOpen(open) {
    state.isOpen = open;
    panel.classList.toggle("open", open);
    try {
      window.sessionStorage.setItem(STORAGE_KEY_OPEN, open ? "1" : "0");
      if (open) {
        window.localStorage.setItem(STORAGE_KEY_ACTIVE, "1");
      }
    } catch (_err) {
      // ignore
    }
    if (open) {
      state.unreadCount = 0;
      renderUnread();
      connectWebSocket();
      setTimeout(() => {
        scrollToBottom();
        inputEl.focus();
      }, 60);
    }
  }

  launcher.addEventListener("click", () => setOpen(!state.isOpen));
  closeBtn.addEventListener("click", () => setOpen(false));
  panel.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && state.isOpen) {
      setOpen(false);
      launcher.focus();
    }
  });

  resetBtn.addEventListener("click", () => {
    const newId = generateUUID();
    const newToken = generateUUID();
    setStoredSessionId(newId);
    setStoredVisitorToken(newToken);
    state.sessionId = newId;
    state.visitorToken = newToken;
    state.visitorEmail = null;
    state.messages = [];
    state.messageIds = new Set();
    state.streamBubbles.clear();
    state.sessionState = "BOT_ACTIVE";
    state.assignedAgent = null;
    state.typing = { active: false, label: "" };
    if (state.ws) {
      const oldWs = state.ws;
      state.ws = null;
      oldWs.close(1000, "New conversation");
    }
    renderAllMessages();
    renderStatus();
    renderTyping();
    connectWebSocket();
  });

  emailBar.addEventListener("submit", (e) => {
    e.preventDefault();
    const email = emailInput.value.trim();
    if (!email) return;
    if (sendJson({ type: "set_email", email })) {
      emailInput.value = "";
    }
  });

  humanBtn.addEventListener("click", () => {
    sendJson({ type: "request_human" });
  });

  function autoResizeInput() {
    inputEl.style.height = "auto";
    inputEl.style.height = `${Math.min(inputEl.scrollHeight, 96)}px`;
  }

  inputEl.addEventListener("input", autoResizeInput);
  inputEl.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      composer.requestSubmit();
    }
  });

  composer.addEventListener("submit", (e) => {
    e.preventDefault();
    const content = inputEl.value.trim();
    if (!content) return;
    if (sendJson({ type: "message", content })) {
      inputEl.value = "";
      autoResizeInput();
    }
  });

  // ---------------------------------------------------------------------------
  // Mount into Host Document
  // ---------------------------------------------------------------------------
  function mount() {
    if (!document.body) return;
    document.body.appendChild(host);
    renderAllMessages();
    renderStatus();

    let previouslyOpen = false;
    let hasActiveSession = false;
    try {
      previouslyOpen = window.sessionStorage.getItem(STORAGE_KEY_OPEN) === "1";
      hasActiveSession = window.localStorage.getItem(STORAGE_KEY_ACTIVE) === "1";
    } catch (_err) {
      previouslyOpen = false;
      hasActiveSession = false;
    }
    if (previouslyOpen) {
      setOpen(true);
    } else if (hasActiveSession) {
      // Keep WebSocket connected across multi-page navigations when panel is minimized
      // so live human agent replies increment the unread badge immediately.
      connectWebSocket();
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", mount);
  } else {
    mount();
  }
})();
