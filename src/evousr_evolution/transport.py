"""OpenAI-compatible transport; one shared semaphore for proposals and predictions.

Retains the experiment's decoding, retry, model-identity and non-thinking checks.
Credentials are read only from environment variables and are never serialized.
"""

import collections, http.client, json, os, threading, time, urllib.error, urllib.request
from pathlib import Path
from .common import read, save, append, digest, sha


class Transport:
    def __init__(self, root, config):
        self.root = Path(root)
        self.cfg = config
        self.url = os.environ.get("EVOUSR_BASE_URL", "").strip().rstrip("/")
        self.key = os.environ.get("EVOUSR_API_KEY", "").strip()
        if not self.url or not self.key:
            raise ValueError(
                "Set EVOUSR_BASE_URL and EVOUSR_API_KEY for your model provider"
            )
        self.lock = threading.RLock()
        self.pace_lock = threading.Lock()
        self.sem = threading.BoundedSemaphore(config["workers"])
        self.next = 0.0
        self.cooldown = 0.0
        self.active = 0
        self.peak = 0
        self.calls = 0
        self.fatal = None
        self.usage = collections.Counter()
        self.statuses = collections.Counter()
        identity = dict(
            base_url=self.url,
            model=config["model"],
            response_models=config["response_models"],
        )
        identity_path = self.root / "service_identity.json"
        if identity_path.exists() and read(identity_path) != identity:
            raise ValueError("Resume endpoint or model identity changed")
        save(identity_path, identity)
        if (self.root / "api-events.jsonl").exists():
            for e in map(json.loads, (self.root / "api-events.jsonl").open()):
                self.calls += 1
                self.statuses[str(e["status"])] += 1
                self.usage.update(
                    {
                        k: v
                        for k, v in e.get("usage", {}).items()
                        if k in ["prompt_tokens", "completion_tokens", "total_tokens"]
                        and isinstance(v, int)
                    }
                )

    def check(self):
        if self.fatal or (self.root / "STOP").exists():
            raise RuntimeError(self.fatal or "STOP requested")

    def stats(self):
        with self.lock:
            return dict(
                calls=self.calls,
                active=self.active,
                peak_concurrency=self.peak,
                statuses=dict(self.statuses),
                usage=dict(self.usage),
                fatal=self.fatal,
            )

    def call(self, jid, messages, max_tokens=256, temperature=0, seed=0):
        self.check()
        payload = dict(
            model=self.cfg["model"],
            messages=messages,
            temperature=temperature,
            top_p=1,
            seed=seed,
            max_tokens=max_tokens,
            stream=False,
            n=1,
            thinking={"type": "disabled"},
            chat_template_kwargs={"enable_thinking": False},
        )
        trace = self.root / "traces" / digest(jid)
        trace.mkdir(parents=True, exist_ok=True)
        if (trace / "result.json").exists():
            assert read(trace / "request.json") == payload, "Resume request changed"
            cached = read(trace / "result.json")
            if not cached["valid_transport"]:
                raise RuntimeError("Cached transport failure requires recovery")
            return cached
        save(trace / "request.json", payload)
        result = dict(
            text="",
            finish_reason=None,
            valid_transport=False,
            response_model=None,
            attempts=[],
            usage={},
            request_sha256=sha(trace / "request.json"),
            trace=str(trace),
        )
        for attempt in range(1, self.cfg["transport"]["attempts_per_request"] + 1):
            self.check()
            with self.sem:
                with self.pace_lock:
                    while time.monotonic() < max(self.next, self.cooldown):
                        self.check()
                        time.sleep(
                            max(
                                0,
                                min(
                                    0.2,
                                    max(self.next, self.cooldown) - time.monotonic(),
                                ),
                            )
                        )
                    self.next = time.monotonic() + self.cfg["request_pacing_seconds"]
                with self.lock:
                    self.check()
                    if self.calls >= self.cfg["max_api_attempts"]:
                        raise RuntimeError("Frozen API cap exhausted")
                    self.calls += 1
                    number = self.calls
                    self.active += 1
                    self.peak = max(self.peak, self.active)
                start = time.time()
                status = 0
                usage = {}
                model = None
                error = None
                body = b""
                try:
                    req = urllib.request.Request(
                        self.url + "/chat/completions",
                        data=json.dumps(payload).encode(),
                        headers={
                            "Authorization": "Bearer " + self.key,
                            "Content-Type": "application/json",
                        },
                    )
                    try:
                        with urllib.request.urlopen(
                            req, timeout=self.cfg["timeout_seconds"]
                        ) as response:
                            status = response.status
                            body = response.read()
                    except urllib.error.HTTPError as e:
                        status = e.code
                        body = e.read()
                        if status in (408, 429, 500, 502, 503, 504):
                            minimum = self.cfg["transport"][
                                (
                                    "cooldown_429_seconds"
                                    if status == 429
                                    else "cooldown_server_network_seconds"
                                )
                            ]
                            try:
                                delay = max(
                                    minimum,
                                    float(e.headers.get("Retry-After", str(minimum))),
                                )
                            except (ValueError, TypeError):
                                delay = minimum
                            with self.lock:
                                self.cooldown = max(
                                    self.cooldown, time.monotonic() + delay
                                )
                    (trace / f"response-{number}.txt").write_bytes(body)
                    if status == 200:
                        try:
                            v = json.loads(body)
                            model = v.get("model")
                            usage = v.get("usage") or {}
                            choice = v["choices"][0]
                            msg = choice["message"]
                            if model not in self.cfg["response_models"]:
                                self.fatal = "Unexpected model identity"
                                raise RuntimeError(self.fatal)
                            result.update(valid_transport=True, response_model=model)
                            if (
                                msg.get("reasoning")
                                or msg.get("reasoning_content")
                                or (usage.get("completion_tokens_details") or {}).get(
                                    "reasoning_tokens", 0
                                )
                            ):
                                result.update(
                                    text="",
                                    finish_reason="protocol_violation",
                                    protocol_error="unexpected_thinking_response",
                                )
                            else:
                                result.update(
                                    text=msg.get("content") or "",
                                    finish_reason=choice.get("finish_reason"),
                                )
                        except (
                            ValueError,
                            KeyError,
                            IndexError,
                            TypeError,
                            AttributeError,
                        ):
                            result.update(
                                valid_transport=True,
                                finish_reason="malformed_response",
                                protocol_error="malformed_response",
                            )
                    elif status in (401, 403, 404):
                        self.fatal = "Service HTTP " + str(status)
                except (
                    urllib.error.URLError,
                    TimeoutError,
                    OSError,
                    http.client.IncompleteRead,
                    http.client.RemoteDisconnected,
                ) as e:
                    error = type(e).__name__
                    with self.lock:
                        self.cooldown = max(
                            self.cooldown,
                            time.monotonic()
                            + self.cfg["transport"]["cooldown_server_network_seconds"],
                        )
                finally:
                    event = dict(
                        arm=getattr(self, "arm", "setup"),
                        phase=(
                            "proposal"
                            if jid.startswith("proposal-")
                            else getattr(self, "phase", "setup")
                        ),
                        number=number,
                        jid=jid,
                        attempt=attempt,
                        start=start,
                        end=time.time(),
                        status=status,
                        error=error,
                        response_model=model,
                        usage=usage,
                    )
                    with self.lock:
                        self.active -= 1
                        self.statuses[str(status)] += 1
                        for k in ["prompt_tokens", "completion_tokens", "total_tokens"]:
                            self.usage[k] += usage.get(k, 0) or 0
                            result["usage"][k] = result["usage"].get(k, 0) + (
                                usage.get(k, 0) or 0
                            )
                        append(self.root / "api-events.jsonl", event)
                        save(self.root / "api-progress.json", self.stats())
                    result["attempts"].append(event)
            if status == 200 or status not in (0, 408, 429, 500, 502, 503, 504):
                break
            if attempt < self.cfg["transport"]["attempts_per_request"]:
                time.sleep(2**attempt)
        save(trace / "result.json", result)
        if not result["valid_transport"]:
            with self.lock:
                self.fatal = (
                    self.fatal or "Transport exhausted; stop before candidate scoring"
                )
        self.check()
        return result
