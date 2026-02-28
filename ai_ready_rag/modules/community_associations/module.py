"""Community Associations module entry point.

Called once at startup by ModuleRegistry.load_module().
All imports from core platform are allowed; core NEVER imports from modules/.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ai_ready_rag.core.module_registry import ModuleRegistry

logger = logging.getLogger(__name__)

# Entity-to-table map for structured SQL queries
# Keys are extracted entity names; values are fully-qualified column paths.
CA_ENTITY_TO_TABLE_MAP: dict[str, str] = {
    "unit_count": "insurance_accounts.units_residential",
    "reserve_fund_balance": "ca_reserve_studies.actual_reserve_balance",
    "reserve_fund_percent_funded": "ca_reserve_studies.percent_funded",
    "replacement_cost_new": "ca_reserve_studies.replacement_cost_new",
    "association_name": "insurance_accounts.account_name",
    "management_company": "insurance_accounts.custom_fields",
    "ccr_requirement": "insurance_requirements.requirement_text",
    "ho6_requirement": "insurance_requirements.requirement_text",
    "policy_number": "insurance_policies.policy_number",
    "carrier_name": "insurance_policies.carrier_name",
}


def register(registry: ModuleRegistry) -> None:
    """Wire CA module into the core platform registry.

    Called once during application startup after ModuleRegistry is initialized.
    Stubs are registered now; full implementations are wired in subsequent issues.
    """
    import pathlib

    module_dir = pathlib.Path(__file__).parent

    # 1. Document classifiers (implemented in #357)
    classifiers_yaml = module_dir / "classifiers.yaml"
    if classifiers_yaml.exists():
        registry.register_document_classifiers("community_associations", str(classifiers_yaml))
    else:
        logger.debug("CA classifiers.yaml not yet available — skipping classifier registration")

    # 2. Entity-to-table map (used by QueryRouter for SQL generation)
    registry.register_entity_map("community_associations", CA_ENTITY_TO_TABLE_MAP)

    # 3. SQL templates (implemented in #365)
    sql_templates_yaml = module_dir / "sql_templates.yaml"
    if sql_templates_yaml.exists():
        registry.register_sql_templates("community_associations", str(sql_templates_yaml))
    else:
        logger.debug("CA sql_templates.yaml not yet available — skipping template registration")

    # 4. Compliance checker (implemented in #363)
    try:
        from ai_ready_rag.modules.community_associations.services.compliance import (
            CommunityAssociationsComplianceChecker,
        )

        registry.register_compliance_checker(
            "community_associations", CommunityAssociationsComplianceChecker
        )
    except ImportError:
        logger.debug("CA ComplianceChecker not yet implemented — skipping")

    # 5. API router (implemented in #373)
    try:
        from ai_ready_rag.modules.community_associations.api.router import ca_router

        registry.register_api_router("community_associations", ca_router, prefix="/api/ca")
    except ImportError:
        logger.debug("CA API router not yet implemented — skipping")

    logger.info(
        "Community Associations module registered (partial — pending #357, #363, #365, #373)"
    )
