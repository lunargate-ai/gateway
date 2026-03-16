FROM golang:1.22-alpine AS builder

WORKDIR /app
COPY go.mod go.sum ./
RUN go mod download

COPY . .
RUN CGO_ENABLED=0 GOOS=linux go build -trimpath -ldflags="-s -w" -o /lunargate ./cmd/gateway

FROM alpine:3.19 AS certs
RUN apk add --no-cache ca-certificates

FROM busybox:1.36.1-musl AS busybox

# Runtime
FROM scratch

WORKDIR /
ENV PATH=/bin
COPY --from=builder /lunargate /lunargate
COPY --from=busybox /bin/busybox /bin/wget
COPY --from=certs /etc/ssl/certs/ca-certificates.crt /etc/ssl/certs/ca-certificates.crt
COPY configs/config.example.yaml /etc/lunargate/config.yaml

EXPOSE 8080

USER 65532:65532

ENTRYPOINT ["/lunargate"]
CMD ["--config", "/etc/lunargate/config.yaml"]
