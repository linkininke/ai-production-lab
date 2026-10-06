Redis 内存达到上限时由 maxmemory-policy 决定淘汰策略。allkeys-lru 会在全部键里淘汰最近最少使用的键。不要把过期时间当成淘汰策略。
