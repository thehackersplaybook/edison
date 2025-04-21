#!/bin/bash
set -e

echo "Running unit tests..."
pytest tests/ -v --cov=edison

echo -e "\nRunning e2e tests..."
pytest e2e/ -v

echo -e "\nAll tests completed successfully!"