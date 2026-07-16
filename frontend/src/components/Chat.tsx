// src/components/Chat.tsx
import { useState, type KeyboardEvent } from "react";
import { sendChatMessage, type Message } from "../api/chat";

export default function Chat() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState<string>("");
  const [loading, setLoading] = useState<boolean>(false);
  const [sources, setSources] = useState<string[]>([]);
  const [error, setError] = useState<string | null>(null);

  async function handleSend(): Promise<void> {
    if (!input.trim() || loading) return;

    const userMessage: Message = { role: "user", content: input };
    const nextMessages: Message[] = [...messages, userMessage];

    setMessages(nextMessages);
    setInput("");
    setLoading(true);
    setError(null);

    try {
      const data = await sendChatMessage(nextMessages);
      setMessages([...nextMessages, { role: "assistant", content: data.answer }]);
      setSources(data.sources);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Unknown error");
      setMessages(messages);
    } finally {
      setLoading(false);
    }
  }

  function handleKeyDown(e: KeyboardEvent<HTMLInputElement>): void {
    if (e.key === "Enter") handleSend();
  }

  return (
    <div>
      <div>
        {messages.map((m, i) => (
          <div key={i}><strong>{m.role}:</strong> {m.content}</div>
        ))}
        {loading && <div>Thinking...</div>}
      </div>

      {error && <div style={{ color: "red" }}>{error}</div>}

      <input
        value={input}
        onChange={(e) => setInput(e.target.value)}
        onKeyDown={handleKeyDown}
        disabled={loading}
      />
      <button onClick={handleSend} disabled={loading}>Send</button>

      {sources.length > 0 && (
        <div>
          <h4>Sources</h4>
          {sources.map((s, i) => <div key={i}>{s}</div>)}
        </div>
      )}
    </div>
  );
}