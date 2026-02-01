# Architecture

The platform is split into services for API, planner, workers, critic, policy gate, and UI.
Redis Streams provide event based coordination and Postgres provides durable state.
OpenTelemetry traces and Prometheus metrics provide observability.
