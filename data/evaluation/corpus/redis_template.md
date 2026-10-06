RedisTemplate 默认使用 JDK 序列化。改成 StringRedisTemplate 之后，键和值按字符串读写，避免客户端之间看不懂二进制。
