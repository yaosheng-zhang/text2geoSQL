# Dockerfile - PostGIS 16 + pgvector（多阶段构建，稳定版）
FROM postgis/postgis:16-3.5 AS builder

# 安装编译依赖
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    postgresql-server-dev-16 \
    && rm -rf /var/lib/apt/lists/*

# 克隆并编译 pgvector（使用浅克隆加快速度）
RUN git clone --branch v0.8.2 https://ghfast.top/github.com/pgvector/pgvector.git \
    && cd pgvector \
    && make \
    && make install \
    && cd .. \
    && rm -rf pgvector

# 最终镜像
FROM postgis/postgis:16-3.5

# 从 builder 阶段复制编译好的 pgvector 文件
COPY --from=builder /usr/lib/postgresql/16/lib/vector.so /usr/lib/postgresql/16/lib/vector.so
COPY --from=builder /usr/share/postgresql/16/extension/vector* /usr/share/postgresql/16/extension/

# 清理不必要的包，减小镜像大小
RUN apt-get update && apt-get install -y --no-install-recommends \
    postgresql-server-dev-16 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*