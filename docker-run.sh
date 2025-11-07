#!/bin/bash
# Helper script to run common Docker commands

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Build the image
build() {
    echo "Building Docker image..."
    docker build -t vit-deepfake:latest .
}

# Run prepare_data.py
prepare_data() {
    echo "Running prepare_data.py..."
    docker run --rm \
        -v "$(pwd)/data:/app/data" \
        vit-deepfake:latest \
        python scripts/prepare_data.py --train_ratio 0.9
}

# Run training
train() {
    echo "Running training..."
    docker run --rm --gpus all \
        -v "$(pwd)/data:/app/data" \
        -v "$(pwd)/models:/app/models" \
        -v "$(pwd)/results:/app/results" \
        vit-deepfake:latest \
        python scripts/train.py "$@"
}

# Run evaluation
evaluate() {
    echo "Running evaluation..."
    docker run --rm --gpus all \
        -v "$(pwd)/data:/app/data" \
        -v "$(pwd)/models:/app/models" \
        -v "$(pwd)/json:/app/json" \
        vit-deepfake:latest \
        python scripts/evaluate.py "$@"
}

# Run visualization
visualize() {
    echo "Running visualization..."
    docker run --rm \
        -v "$(pwd)/results:/app/results" \
        -v "$(pwd)/plots:/app/plots" \
        vit-deepfake:latest \
        python scripts/visualize_training.py "$@"
}

# Run inference
inference() {
    echo "Running inference..."
    docker run --rm --gpus all \
        -v "$(pwd)/data:/app/data" \
        -v "$(pwd)/models:/app/models" \
        vit-deepfake:latest \
        python scripts/inference.py "$@"
}

# Show usage
usage() {
    echo "Usage: $0 {build|prepare|train|evaluate|visualize|inference} [args...]"
    echo ""
    echo "Commands:"
    echo "  build              - Build Docker image"
    echo "  prepare            - Run prepare_data.py"
    echo "  train [args...]    - Run training (pass train.py args)"
    echo "  evaluate [args...] - Run evaluation (pass evaluate.py args)"
    echo "  visualize [args...]- Run visualization (pass visualize_training.py args)"
    echo "  inference [args...]- Run inference (pass inference.py args)"
    echo ""
    echo "Examples:"
    echo "  $0 build"
    echo "  $0 prepare"
    echo "  $0 train --pretrained --epochs 15 --batch_size 16"
    echo "  $0 evaluate --model_name vit_base_patch16_224"
    echo "  $0 visualize --model_name vit_base_patch16_224"
}

# Main
case "${1:-}" in
    build)
        build
        ;;
    prepare)
        prepare_data
        ;;
    train)
        shift
        train "$@"
        ;;
    evaluate)
        shift
        evaluate "$@"
        ;;
    visualize)
        shift
        visualize "$@"
        ;;
    inference)
        shift
        inference "$@"
        ;;
    *)
        usage
        exit 1
        ;;
esac

