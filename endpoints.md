# `/observations` (iNaturalist data)

| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/observations/` | Return all stored observations (with pagination) |
| GET | `/observations/latest/` | Get newly fetched observations from iNaturalist |
| POST | `/observations/refresh` | Manually trigger data fetch from iNaturalist |
| GET | `/observations/{id}` | Get a single observation by ID |

# `/datasets` (Model prep)

| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/datasets/latest/` | Get the latest processed training dataset |
| POST | `/datasets/generate` | Combine obs + weather → structured model dataset |

# `/status` (Utility/Health)

| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/status/health` | Ping for uptime or healthcheck |
| GET | `/status/stats` | Summary stats (obs count, last pull) |