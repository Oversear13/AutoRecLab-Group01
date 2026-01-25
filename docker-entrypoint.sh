#!/bin/bash
set -e

# Check if embeddings need to be generated
if [ ! -d "/app/ragEmbeddings/omnirec" ] || [ ! -d "/app/ragEmbeddings/lenskit" ] || [ ! -d "/app/ragEmbeddings/recbole" ]; then
    echo "Embeddings not found. Generating now..."
    echo ""
    
    cd /app && uv run python -m cli.embeddings.main generate --all
    
    echo ""
    echo "Embeddings generated successfully!"
    echo ""
fi

exec "$@"
