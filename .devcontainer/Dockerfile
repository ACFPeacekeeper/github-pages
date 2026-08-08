# Dev Container image: Node.js for the Next.js site plus a Python
# toolchain for the notebooks/ research workspace.
FROM node:20-slim

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    curl \
    git \
    python3 \
    && rm -rf /var/lib/apt/lists/*

# uv for fast Python package management (notebooks/)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Create a non-root user for development
RUN useradd -m -s /bin/bash vscode
USER vscode
WORKDIR /app

CMD ["/bin/bash"]
