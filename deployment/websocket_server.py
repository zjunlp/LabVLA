"""
WebSocket Policy Server - Compatible with openpi_client.WebsocketClientPolicy protocol.

This server implements the same WebSocket protocol used by OpenPI:
1. On connection: send metadata dict via msgpack
2. Loop: receive observation (msgpack), run inference, send action result (msgpack)

LabUtopia's RemoteInferenceEngine uses openpi_client.WebsocketClientPolicy to connect,
so this server is directly compatible.
"""

import asyncio
import functools
import http
import logging
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, Optional
from urllib.parse import parse_qs, urlsplit

import msgpack
import numpy as np

logger = logging.getLogger(__name__)


# ---- msgpack-numpy serialization (compatible with openpi_client.msgpack_numpy) ----

def _pack_array(obj):
    """Pack numpy arrays for msgpack serialization."""
    if isinstance(obj, (np.ndarray, np.generic)) and obj.dtype.kind in ("V", "O", "c"):
        raise ValueError(f"Unsupported dtype: {obj.dtype}")
    if isinstance(obj, np.ndarray):
        return {
            b"__ndarray__": True,
            b"data": obj.tobytes(),
            b"dtype": obj.dtype.str,
            b"shape": obj.shape,
        }
    if isinstance(obj, np.generic):
        return {
            b"__npgeneric__": True,
            b"data": obj.item(),
            b"dtype": obj.dtype.str,
        }
    return obj


def _unpack_array(obj):
    """Unpack numpy arrays from msgpack deserialization."""
    if b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])
    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj


Packer = functools.partial(msgpack.Packer, default=_pack_array)
packb = functools.partial(msgpack.packb, default=_pack_array)
unpackb = functools.partial(msgpack.unpackb, object_hook=_unpack_array)


class WebsocketPolicyServer:
    """
    A WebSocket server that serves a policy inference function.

    Protocol (identical to openpi's WebsocketPolicyServer):
    1. On new connection: send metadata as msgpack bytes
    2. Loop:
       a. Receive observation dict (msgpack-encoded with numpy arrays)
       b. Call infer_fn(observation) -> result dict
       c. Send result dict (msgpack-encoded) back to client
    """

    def __init__(
        self,
        infer_fn: Callable[[Dict[str, Any]], Dict[str, Any]],
        host: str = "127.0.0.1",
        port: int = 8000,
        metadata: Optional[Dict[str, Any]] = None,
        max_workers: int = 4,
        auth_token: Optional[str] = None,
        max_message_size: int = 16 * 1024 * 1024,
        max_inflight: Optional[int] = None,
    ):
        """
        Args:
            infer_fn: A callable that takes an observation dict and returns an action dict.
            host: Server host address.
            port: Server port.
            metadata: Metadata dict to send to clients on connection.
            max_workers: Size of the executor pool that runs blocking inference
                off the asyncio event loop. Default 4 suits a single GPU serving
                one or two clients; raise only with spare GPU memory for parallel
                in-flight inference.
            max_inflight: Hard admission-control cap on concurrently accepted
                inference requests (running + queued). At the cap, a new request
                is rejected immediately with {"error": "overload", ...} instead
                of piling onto the shared GPU. Default None -> max_workers * 2.
                This bounds queue growth and gives clients a retry-later signal
                rather than a generic error after the GPU is overwhelmed.
        """
        if host not in {"127.0.0.1", "localhost", "::1"} and not auth_token:
            raise ValueError(
                "Refusing to serve a network-exposed websocket without auth_token"
            )
        self._infer_fn = infer_fn
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        self._auth_token = auth_token
        self._max_message_size = int(max_message_size)
        # Explicit executor sizing; the backlog warning fires once per backlog
        # crossing to avoid log spam.
        self._max_workers = int(max_workers)
        self._executor = ThreadPoolExecutor(
            max_workers=self._max_workers,
            thread_name_prefix="ws_infer",
        )
        self._inflight_count = 0
        self._backlog_warned = False
        # Hard admission cap. Default 2x the pool so a little queuing smooths
        # bursty clients while unbounded pile-up is rejected. Floor at 1 so the
        # server always accepts at least one request.
        if max_inflight is None:
            self._max_inflight = max(1, self._max_workers * 2)
        else:
            self._max_inflight = max(1, int(max_inflight))

    def serve_forever(self):
        """Blocking call to run the server."""
        asyncio.run(self._run())

    async def _run(self):
        import websockets

        logger.info(f"Starting WebSocket policy server on {self._host}:{self._port}")
        logger.info(
            "WebSocket security: auth=%s max_message_size=%d bytes",
            "required" if self._auth_token else "disabled",
            self._max_message_size,
        )
        logger.info(
            f"[M-27] inference executor: ThreadPoolExecutor(max_workers={self._max_workers}); "
            f"backlog warning threshold={self._max_workers * 2}"
        )
        logger.info(
            f"[C22] admission control: max_inflight={self._max_inflight} "
            f"(requests beyond this are rejected with error=overload)"
        )

        # websockets 12.x (legacy API) -- compatible with openpi_client
        async with websockets.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=self._max_message_size,
            process_request=self._health_check_legacy,
            ping_interval=300,
            ping_timeout=300,
            close_timeout=60,
        ) as server:
            await asyncio.Future()  # run forever

    async def _handler(self, websocket, path=None):
        """Handle a single WebSocket connection (compatible with websockets 12.x)."""
        logger.info(f"Connection from {websocket.remote_address} opened")
        packer = Packer()

        # Step 1: Send metadata
        await websocket.send(packer.pack(self._metadata))

        prev_total_time = None
        while True:
            try:
                start_time = time.monotonic()

                # Step 2a: Receive observation
                raw_data = await websocket.recv()
                obs = unpackb(raw_data)

                # Step 2b: Run inference in a thread pool so the event loop isn't
                # blocked during slow inference and can still answer WebSocket
                # ping/pong (otherwise "keepalive ping timeout" errors).
                infer_start = time.monotonic()
                loop = asyncio.get_running_loop()
                # At the in-flight cap, reject with an overload signal rather than
                # queuing onto the shared GPU (unbounded growth -> eventual OOM).
                # The connection stays open so the client can retry later.
                if self._inflight_count >= self._max_inflight:
                    logger.warning(
                        "[C22] rejecting request: in-flight %d >= cap %d "
                        "(max_workers=%d). Client told to retry later.",
                        self._inflight_count, self._max_inflight, self._max_workers,
                    )
                    await websocket.send(packer.pack({
                        "error": "overload",
                        "detail": (
                            f"server at capacity ({self._inflight_count}/"
                            f"{self._max_inflight} in-flight); retry later"
                        ),
                        "retry_after_ms": 100,
                    }))
                    continue
                # Track in-flight count and fire a single backlog warning when
                # queue depth exceeds 2x the pool size; don't re-warn until the
                # pool drains below the threshold.
                self._inflight_count += 1
                threshold = self._max_workers * 2
                if self._inflight_count > threshold and not self._backlog_warned:
                    logger.warning(
                        f"[M-27] inference backlog: {self._inflight_count} in-flight "
                        f"> threshold {threshold} (max_workers={self._max_workers}). "
                        f"Consider increasing --max_workers or reducing client rate."
                    )
                    self._backlog_warned = True
                elif self._inflight_count <= self._max_workers and self._backlog_warned:
                    self._backlog_warned = False
                try:
                    result = await loop.run_in_executor(
                        self._executor, self._infer_fn, obs
                    )
                finally:
                    self._inflight_count -= 1
                infer_time = time.monotonic() - infer_start

                # Add timing info
                result["server_timing"] = {
                    "infer_ms": infer_time * 1000,
                }
                if prev_total_time is not None:
                    result["server_timing"]["prev_total_ms"] = prev_total_time * 1000

                # Step 2c: Send result
                await websocket.send(packer.pack(result))
                prev_total_time = time.monotonic() - start_time

            except Exception as e:
                if "ConnectionClosed" in type(e).__name__:
                    logger.info(f"Connection from {websocket.remote_address} closed")
                    break
                logger.error(f"Error during inference: {traceback.format_exc()}")
                try:
                    import websockets.frames
                    await websocket.send(packer.pack({"error": "internal_server_error"}))
                    await websocket.close(
                        code=websockets.frames.CloseCode.INTERNAL_ERROR,
                        reason="Internal server error.",
                    )
                except Exception:
                    pass
                raise

    async def _health_check(self, connection, request):
        """Handle HTTP health check requests (websockets >= 13)."""
        if request.path == "/healthz":
            return connection.respond(http.HTTPStatus.OK, "OK\n")
        if not self._is_authorized(request.path, request.headers):
            return connection.respond(http.HTTPStatus.UNAUTHORIZED, "Unauthorized\n")
        return None

    async def _health_check_legacy(self, path, request_headers):
        """Handle HTTP health check requests (websockets 12.x)."""
        if path == "/healthz":
            return http.HTTPStatus.OK, [], b"OK\n"
        if not self._is_authorized(path, request_headers):
            return http.HTTPStatus.UNAUTHORIZED, [], b"Unauthorized\n"
        return None

    def _is_authorized(self, path: str, headers) -> bool:
        """Authorize websocket upgrades via bearer header only."""
        if not self._auth_token:
            return True

        header_value = ""
        try:
            header_value = headers.get("Authorization", "") or ""
        except AttributeError:
            header_value = ""
        if header_value == f"Bearer {self._auth_token}":
            return True
        if parse_qs(urlsplit(path).query).get("token"):
            logger.warning(
                "Rejecting websocket query-string token auth; use "
                "Authorization: Bearer <token> so credentials do not enter URL logs."
            )
        return False
