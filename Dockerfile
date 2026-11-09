# The runtime stage copies a file from the test stage, so every test, unit and
# integration, runs on every build.

FROM golang:1.27.3-trixie AS build
WORKDIR /src
COPY go.mod go.sum ./
RUN go mod download
COPY . .
# sherpa-onnx's Go binding links its shared libraries through cgo.
RUN CGO_ENABLED=1 go build -trimpath -ldflags="-s -w" -o /out/ai . \
 && mkdir -p /out/lib \
 && lib="$(go list -m -f '{{.Dir}}' github.com/k2-fsa/sherpa-onnx-go-linux)/lib/x86_64-unknown-linux-gnu" \
 && cp "$lib/libsherpa-onnx-c-api.so" "$lib/libonnxruntime.so" /out/lib/

FROM debian:trixie-slim AS fetch
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl \
 && rm -rf /var/lib/apt/lists/*
COPY scripts/ /scripts/

FROM fetch AS models
RUN /scripts/fetch-assets.sh models /models

FROM fetch AS testwavs
RUN /scripts/fetch-assets.sh testwavs /testwavs

FROM build AS test
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
 && rm -rf /var/lib/apt/lists/* \
 && cp /out/lib/*.so /usr/local/lib/ && ldconfig
COPY --from=models /models /models
COPY --from=testwavs /testwavs /testwavs
RUN go vet ./... \
 && AI_TEST_MODELS=/models AI_TEST_WAVS=/testwavs go test -race -tags model ./... \
 && touch /tests-passed

FROM debian:trixie-slim
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
 && rm -rf /var/lib/apt/lists/*
COPY --from=build /out/lib/ /usr/local/lib/
RUN ldconfig
COPY --from=models /models /models
COPY --from=build /out/ai /usr/local/bin/ai
COPY --from=test /tests-passed /tests-passed
USER 65532:65532
EXPOSE 8000
ENTRYPOINT ["/usr/local/bin/ai"]
