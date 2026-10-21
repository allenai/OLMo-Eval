"""Modal deployment adapter with encrypted HTTP transport and reliable shutdown."""

from __future__ import annotations

import asyncio
import contextlib
import time

import modal
import modal.experimental
from swerex.deployment.modal import ModalDeployment
from swerex.runtime.remote import RemoteRuntime


async def stop_modal_app(app_name: str) -> None:
    """Stop a named Modal app so it does not outlive the run that created it.

    Apps created by ``modal.App.lookup(name, create_if_missing=True)`` stay
    deployed after their sandboxes exit. An app that is already gone is fine.
    """
    with contextlib.suppress(modal.exception.NotFoundError):
        await modal.experimental.stop_app.aio(app_name)


class ManagedModalDeployment(ModalDeployment):
    """SWE-ReX Modal deployment with managed lifecycle and encrypted transport."""

    async def start(self) -> None:
        if self._runtime is not None and self._sandbox is not None:
            self.logger.warning("Deployment is already started. Ignoring duplicate start() call.")
            return

        self.logger.info("Starting modal sandbox with encrypted runtime tunnel")
        self._hooks.on_custom_step("Starting modal sandbox")
        started_at = time.time()
        token = self._get_token()
        modal_kwargs = dict(self._modal_kwargs)
        modal_kwargs.pop("unencrypted_ports", None)
        modal_kwargs.pop("encrypted_ports", None)
        try:
            self._sandbox = await modal.Sandbox.create.aio(
                "/usr/bin/env",
                "bash",
                "-c",
                self._start_swerex_cmd(token),
                image=self._image,
                timeout=int(self._deployment_timeout),
                encrypted_ports=[self._port],
                app=self._app,
                **modal_kwargs,
            )
            tunnels = await self._sandbox.tunnels.aio()
            tunnel = tunnels[self._port]
            creation_seconds = time.time() - started_at
            self.logger.info(
                "Sandbox (%s) created in %.2fs",
                self._sandbox.object_id,
                creation_seconds,
            )
            self.logger.info("Check sandbox logs at %s", await self.get_modal_log_url())
            await asyncio.sleep(1)
            self.logger.info("Starting runtime at encrypted tunnel %s", tunnel.url)
            self._hooks.on_custom_step("Starting runtime")
            self._runtime = RemoteRuntime(
                host=tunnel.url,
                timeout=self._runtime_timeout,
                auth_token=token,
                logger=self.logger,
            )
            remaining_timeout = max(0, self._startup_timeout - creation_seconds)
            runtime_started_at = time.time()
            await self._wait_until_alive(timeout=remaining_timeout)
            self.logger.info("Runtime started in %.2fs", time.time() - runtime_started_at)
        except BaseException:
            await self.stop()
            raise

    async def stop(self) -> None:
        """Terminate Modal directly without calling the runtime over HTTP."""
        # SandboxExecutor already force-terminates first. Keep this idempotent
        # for callers that stop the deployment directly, and never call /close
        # on a server that has already been terminated.
        if self._sandbox is not None:
            with contextlib.suppress(Exception):
                await self._sandbox.terminate.aio()
        self._runtime = None
        self._sandbox = None
        self._app = None
