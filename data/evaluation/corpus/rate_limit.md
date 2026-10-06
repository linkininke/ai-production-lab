网关用 X-RateLimit-Remaining 告诉客户端还剩多少额度。剩余为 0 时返回 429，客户端应按 Retry-After 等待。
