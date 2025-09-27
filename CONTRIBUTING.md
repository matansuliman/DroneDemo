# Contributing

Thanks for your interest! This org welcomes **small, focused** contributions — docs fixes, CI tweaks, Dockerfile improvements, Helm/Kubernetes examples, and tiny features.

## Ways to contribute
- **Issues:** Bug reports with clear reproduction, versions, and logs.
- **Docs:** Clarify README, add quickstarts (kind/minikube), fix typos.
- **DevOps:** GitHub Actions, Dockerfiles, Helm charts, Kubernetes manifests, Prometheus/Grafana snippets.

## Ground rules
- Be kind and specific.
- Ask before large refactors.
- Keep PRs small and reviewable (≲ ~300 changed lines).
- Use clear commit messages (Conventional Commits preferred, e.g., `fix: ...`, `feat: ...`, `docs: ...`).

## Getting started
1) **Discuss/assign** the Issue you want to tackle (label `good first issue` if you’re new).  
2) **Fork** the repo (or create a branch if you have write access).  
3) Create a feature branch:
```bash
git checkout -b feat/<short-topic>
```
4) Make changes and run checks locally.  
5) Push your branch and open a Pull Request describing:
   - What changed and **why**
   - How you tested it (commands, env versions)
   - Screenshots/logs if relevant

## Development
### Python projects
- Python ≥ 3.10, `venv` recommended.
- Lint/format (if configured): `ruff`, `black`.
- Tests (if configured): `pytest`.
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scriptsctivate
pip install -r requirements.txt
pytest -q
```

### Node projects (if present)
- Install with `npm ci` or `pnpm i --frozen-lockfile`.
- Lint with ESLint/Prettier if configured.
- Run tests: `npm test`.

### Docker / Kubernetes / Helm
- Build: `docker build -t <image>:dev .`
- Run:  `docker run --rm -p 8080:8080 <image>:dev`
- kind / minikube:
```bash
kubectl apply -f k8s/               # or:
helm install <name> ./helm/chart -f values.yaml
kubectl get pods
```
- Please include **readiness/liveness** probes and **resources** (requests/limits) for new Deployments.

### Observability
- If you add endpoints, consider exposing `/metrics` (Prometheus).
- Include sample **scrape annotations** in README or the chart.

## Testing your change
Include steps for others to reproduce:
- OS / Docker Desktop / k8s version
- Exact commands
- Logs (trimmed)

## Repo-specific notes
### devops-playground
- Keep examples tiny and self‑contained.
- CI should finish in < 5 minutes (GitHub Actions preferred).
- Each example should have a short **“Try it”** section in the README.
- Suggested labels: `good first issue`, `help wanted`, `docs`, `k8s`, `helm`, `docker`, `ci`.

### Workshop-in-Autonomous-Systems-Simulation
- Do **not** commit datasets, checkpoints, or secrets; use `.gitignore`.
- Notebooks should run top-to-bottom; clear private paths.
- **License note:** Ensure the README’s license text matches the repo’s LICENSE file.

## Code style & checks
- Format and lint before pushing.
- Add or update tests when fixing bugs or adding features.

## Pull Request checklist
- [ ] Small, focused change
- [ ] Updated README/docs if needed
- [ ] Lint/tests pass locally and in CI
- [ ] Repro steps / logs included (when relevant)

## Security
- Never commit secrets (tokens, keys, passwords).
- If you find a security issue, report it privately; don’t open a public issue.

## License
By contributing, you agree your contributions are licensed under this repo’s LICENSE.
