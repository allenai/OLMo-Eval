"""Dashboard read API (/api/*), served behind IAP."""

from fastapi import APIRouter

from . import catalog, compare, misc, run_tabs, runs

router = APIRouter(prefix="/api", tags=["dashboard"])
router.include_router(misc.router)
router.include_router(runs.router)
router.include_router(run_tabs.router)
router.include_router(compare.router)
router.include_router(catalog.router)
