.PHONY: build run test vet lint clean docker

BINARY_NAME=lunargate
BUILD_DIR=bin
GO=go
VERSION?=dev
LDFLAGS=-s -w -X main.version=$(VERSION)

build:
	$(GO) build -trimpath -ldflags "$(LDFLAGS)" -o $(BUILD_DIR)/$(BINARY_NAME) ./cmd/gateway

run: build
	$(BUILD_DIR)/$(BINARY_NAME) --config config.yaml

test:
	$(GO) test -v -race ./...

vet:
	$(GO) vet ./...

lint: vet
	@echo "Lint passed (add golangci-lint later)"

goreleaser-check:
	goreleaser check

goreleaser-snapshot:
	goreleaser release --snapshot --clean

goreleaser-release:
	goreleaser release --clean

clean:
	rm -rf $(BUILD_DIR)

docker-build:
	docker build -t lunargate:latest .

docker-run:
	docker-compose up -d

tidy:
	$(GO) mod tidy

deps:
	$(GO) mod download

all: vet test build
