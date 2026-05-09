"""
CATIA REST API Module

Provides FastAPI-based REST endpoints for the CATIA system.

Endpoints:
    - /api/v1/perils - Peril type information
    - /api/v1/analysis - Run catastrophe analyses
    - /api/v1/simulation - Financial impact simulations
    - /api/v1/mitigation - Mitigation recommendations
"""

__all__ = ["app", "schemas", "routes"]

