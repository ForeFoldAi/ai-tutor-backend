# LearnMestro Loki + Promtail (Coolify)

## Deploy in Coolify

1. Coolify → **+ New** → **Docker Compose**
2. Use this directory (`deploy/loki/`) as the compose source (Git repo path or paste files)
3. Deploy
4. Do **not** assign a public domain to Loki
5. On the **API** Coolify app, set:

```bash
LOKI_URL=http://loki:3100
```

If Coolify puts compose services on a project network with a different hostname, use that hostname instead of `loki`.

6. Redeploy API
7. Admin → Monitor → Production Logs

## Verify

```bash
curl -s http://127.0.0.1:3100/ready
# ready
```

From the API container:

```bash
curl -s http://loki:3100/ready
```

## Notes

- Retention: 14 days (`loki-config.yaml`)
- Promtail reads Docker logs via `/var/run/docker.sock`
- If Coolify blocks Docker socket mounts, enable it for this compose app or switch Promtail to Coolify’s supported log drain method
- Never expose Loki publicly without auth
