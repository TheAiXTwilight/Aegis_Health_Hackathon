// frontend/src/features/report/ChatPanel.jsx
import { useState, useRef, useEffect } from "react";
import "./ChatPanel.css";

const MAX_TURNS = 7;

export default function ChatPanel({ jobId, isVisible }) {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [suggested, setSuggested] = useState([]);
  const [turn, setTurn] = useState(0);
  const [loading, setLoading] = useState(false);
  const [initializing, setInitializing] = useState(true);
  const [limitReached, setLimitReached] = useState(false);
  const bottomRef = useRef(null);

  // Auto-scroll to latest message
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // Load chat state (prior messages + remaining turns) when the panel
  // becomes visible.
  //
  // Session semantics: turn usage is PERSISTENT per report. Closing
  // and re-opening the chat panel does NOT reset the counter or wipe
  // history — if you've used 5/7 turns, reopening shows 5/7 (2 left)
  // and replays the prior conversation. A report that reached 7/7
  // stays exhausted until a new report/assessment is generated.
  //
  // No `loadedJobRef` guard — we still call /init on every open so
  // switching between reports (or reopening the same one) always
  // reflects the latest server-side state.
  useEffect(() => {
    if (!isVisible || !jobId) return;

    let cancelled = false;
    setInitializing(true);

    // Reset local UI immediately so stale messages, chips, and the
    // stale turn counter don't flash before the init response arrives
    setMessages([]);
    setSuggested([]);
    setTurn(0);
    setLimitReached(false);
    setInput("");

    (async () => {
      try {
        const resp = await fetch(
          `/queue/chat/${encodeURIComponent(jobId)}/init`,
          { credentials: "include" }
        );
        if (!resp.ok) throw new Error("init failed");
        const data = await resp.json();
        if (cancelled) return;

        const serverLimit = Boolean(data.limit_reached);

        // Backend always returns an empty messages array on init now,
        // but map defensively so any future non-empty response still
        // renders correctly with the same shape as live messages.
        setMessages(
          (data.messages || []).map((m) => ({
            role: m.role,
            content: m.content,
            severity_delta: m.severity_delta || null,
            enriched: false,
          }))
        );
        setTurn(data.turn || 0);
        setLimitReached(serverLimit);

        // Defensive: if the report is exhausted, NEVER show chips even
        // if the server accidentally returned some. A limit-reached
        // report cannot accept another question, so any clickable
        // chip would just fail silently or send a wasted request.
        setSuggested(serverLimit ? [] : (data.suggested_questions || []));
      } catch {
        // Init fetch failed — show baseline suggestions so the user
        // isn't left with an empty chat and no starting point.
        // (Won't run when limitReached is true because that state
        // only gets set inside the successful branch above.)
        if (!cancelled) {
          setSuggested([
            "What are the most critical factors in my report?",
            "What should I do next?",
            "Is my condition getting worse?",
          ]);
        }
      } finally {
        if (!cancelled) setInitializing(false);
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [isVisible, jobId]);

  if (!isVisible) return null;

  // Math.max keeps turnLeft from ever going negative if the server
  // and client counters briefly disagree
  const turnLeft = Math.max(0, MAX_TURNS - turn);

  const handleSend = async (overrideText) => {
    const text = (overrideText ?? input).trim();
    if (!text || turnLeft <= 0 || loading || limitReached) return;

    const userMsg = {
      role: "user",
      content: text,
      severity_delta: null,
      enriched: false,
    };

    setMessages((prev) => [...prev, userMsg]);
    setInput("");
    // Clear suggestions immediately so stale chips don't linger
    // while the deterministic answer (+ optional enrichment) loads
    setSuggested([]);
    setLoading(true);

    try {
      const resp = await fetch("/queue/chat", {
        method: "POST",
        credentials: "include",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          job_id: jobId,
          message: userMsg.content,
        }),
      });

      const data = await resp.json().catch(() => ({}));
      if (!resp.ok) throw new Error(data.detail || "Chat request failed");

      const assistantMsg = {
        role: "assistant",
        content: data.answer,
        severity_delta: data.severity_delta || null,
        // enriched = true when the idle model added a connecting sentence
        // on top of the deterministic answer. The deterministic answer is
        // always shown regardless — this is purely additive.
        enriched: Boolean(data.enriched),
      };

      setMessages((prev) => [...prev, assistantMsg]);

      // Trust the server's turn count but never regress. Handles the
      // case where two answers arrive out of order (fast clicks) —
      // the badge always reflects the highest turn seen so far.
      const newTurn = data.turn || 0;
      setTurn((prev) => Math.max(prev, newTurn));

      // Same defensive rule as init: if this response pushed us to
      // the limit, clear chips regardless of what the server sent.
      const nowLimited = newTurn >= MAX_TURNS;
      if (nowLimited) {
        setLimitReached(true);
        setSuggested([]);
      } else {
        setSuggested(data.suggested_questions || []);
      }
    } catch {
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: "Sorry, something went wrong. Please try again.",
          severity_delta: null,
          enriched: false,
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleSuggestedClick = (question) => {
    handleSend(question);
  };

  // Consolidated disabled state — input, send button, and chips all
  // use the same rule so they can never disagree.
  const inputDisabled = loading || limitReached || turnLeft <= 0;

  return (
    <div className="chat-panel">

      {/* ── Header ── */}
      <div className="chat-header">
        <div className="chat-title-group">
          <svg
            width="18"
            height="18"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2.5"
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
          </svg>
          <span>Follow-up Questions</span>
        </div>
        <span
          className={`chat-turn-badge ${turnLeft <= 1 ? "chat-turn-low" : ""}`}
        >
          {limitReached
            ? `${MAX_TURNS}/${MAX_TURNS} used`
            : `${turnLeft} turn${turnLeft !== 1 ? "s" : ""} left`}
        </span>
      </div>

      {/* ── Message thread ── */}
      <div className="chat-messages">

        {/* Loading initial context */}
        {initializing && messages.length === 0 && (
          <p className="chat-placeholder">Loading report context…</p>
        )}

        {/* Limit reached — show WHETHER OR NOT there are messages, so a
            greeting-only exhausted session (e.g. a "Hello Sahil…" bubble
            with 7/7 already used) still tells the user this report is
            closed for further questions. Previously this was gated by
            `messages.length === 0`, which hid the notice whenever any
            message was present. */}
        {!initializing && limitReached && (
          <p className="chat-placeholder">
            You've used all {MAX_TURNS} follow-up questions for this report.
            Start a new assessment to ask further questions.
          </p>
        )}

        {/* Empty state — ready for first question */}
        {!initializing && messages.length === 0 && !limitReached && (
          <p className="chat-placeholder">
            Ask a question about your triage report. You have {turnLeft}{" "}
            follow-up turn{turnLeft !== 1 ? "s" : ""} left.
          </p>
        )}

        {/* Message bubbles */}
        {messages.map((msg, i) => (
          <div key={i} className={`chat-msg ${msg.role}`}>
            <div className="chat-msg-bubble">
              {msg.content}

              {/* Severity change tag — only shown when severity changed */}
              {msg.severity_delta && msg.severity_delta !== "unchanged" && (
                <span
                  className={`chat-severity-tag chat-sev-${msg.severity_delta}`}
                >
                  Severity: {msg.severity_delta}
                </span>
              )}
            </div>
          </div>
        ))}

        {/* Typing indicator */}
        {loading && (
          <div className="chat-msg assistant">
            <div className="chat-msg-bubble chat-typing">
              <span className="chat-dot" />
              <span className="chat-dot" />
              <span className="chat-dot" />
            </div>
          </div>
        )}

        <div ref={bottomRef} />
      </div>

      {/* ── Suggested question chips ──
          Generated from actual report findings — lab values, symptoms,
          severity reasons, drug interactions, X-ray findings.
          Hidden when limit reached so exhausted sessions don't offer
          chips the user can't click anyway. Also hidden while loading
          so no stale chips flash between turns. */}
      {suggested.length > 0 && !limitReached && !loading && (
        <div className="chat-suggested">
          {suggested.map((q, i) => (
            <button
              key={i}
              className="chat-suggested-btn"
              onClick={() => handleSuggestedClick(q)}
              disabled={inputDisabled}
            >
              {q}
            </button>
          ))}
        </div>
      )}

      {/* ── Input row ── */}
      <div className="chat-input-row">
        <textarea
          className="chat-input"
          placeholder={
            limitReached
              ? "All 7 questions used — start a new assessment to continue"
              : turnLeft > 0
                ? "Type your question..."
                : "Turn limit reached"
          }
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          disabled={inputDisabled}
          rows={1}
        />
        <button
          className="chat-send-btn"
          onClick={() => handleSend()}
          disabled={!input.trim() || inputDisabled}
        >
          <svg
            width="16"
            height="16"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2.5"
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            <line x1="22" y1="2" x2="11" y2="13" />
            <polygon points="22 2 15 22 11 13 2 9 22 2" />
          </svg>
        </button>
      </div>

    </div>
  );
}