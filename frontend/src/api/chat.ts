// src/api/chat.ts

export type Role = "user" | "assistant";

export interface Message {
  role: Role;
  content: string;
}

export interface ChatResponse {
  answer: string;
  sources: string[];
}

const API_URL = "http://localhost:8000";

export async function sendChatMessage(messages: Message[]): Promise<ChatResponse> {
  const res = await fetch(`${API_URL}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ messages }),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({} as { detail?: string }));
    throw new Error(err.detail || `Request failed: ${res.status}`);
  }

  return res.json() as Promise<ChatResponse>;
}