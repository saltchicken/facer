.PHONY: all build server clean help

# Default target: builds UI and runs server
all: build server

# Build the React frontend
build:
	@echo "📦 Building Frontend..."
	cd facer-ui && npm run build

# Run the Python backend
server:
	@echo "🚀 Starting Backend..."
	python -m facer.server

# Run both in development mode (Optional, requires concurrently or separate terminals)
# This assumes you have the tools installed
dev:
	@echo "⚠️  Run 'npm run dev' in facer-ui and 'python -m facer.server' in separate terminals for hot-reloading."

# Clean build artifacts
clean:
	rm -rf facer-ui/dist

# Help command
help:
	@echo "Available commands:"
	@echo "  make         - Build UI and start Server (Streamlined)"
	@echo "  make build   - Only build the UI"
	@echo "  make server  - Only start the Python server (skip build)"
	@echo "  make clean   - Remove dist folder"
