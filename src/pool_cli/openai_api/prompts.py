from __future__ import annotations

from collections.abc import Iterable

from pool_cli.pools import PoolDefinition


USER_POOL_SCHEMA_NAME = "user_pool_metadata"
POOL_ACTION_SCHEMA_NAME = "pool_action_suggestion"
POOL_DEFINITIONS_PREFIX = "Pool definitions:\n"


def build_pool_catalog(pools: Iterable[PoolDefinition]) -> str:
    return "\n".join(f"- {pool.name}: {pool.description}" for pool in pools)


def build_user_pool_schema() -> dict[str, object]:
    return {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "description": {"type": "string"},
            "action_title": {"type": "string"},
            "why": {"type": "string"},
        },
        "required": ["name", "description", "action_title", "why"],
        "additionalProperties": False,
    }


def build_user_pool_system_prompt() -> str:
    return (
        "You design one adaptive user-specific screenshot pool from examples.\n"
        "Return concise JSON fields: name, description, action_title, why.\n"
        "Rules:\n"
        "- Name: short, specific, 2-5 words.\n"
        "- Description: one sentence.\n"
        "- Action title: imperative and useful.\n"
        "- Why: one sentence explaining utility.\n"
        "- Avoid generic names like Misc, Other, Random."
    )


def build_pool_action_schema() -> dict[str, object]:
    return {
        "type": "object",
        "properties": {
            "action_title": {"type": "string", "minLength": 3, "maxLength": 90},
            "why": {"type": "string", "minLength": 8, "maxLength": 220},
            "notes": {"type": "string", "minLength": 8, "maxLength": 220},
            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        },
        "required": ["action_title", "why", "notes", "confidence"],
        "additionalProperties": False,
    }


def build_pool_action_system_prompt() -> str:
    return (
        "You design one action suggestion for a screenshot pool.\n"
        "Return concise JSON fields: action_title, why, notes, confidence.\n"
        "Core principle:\n"
        "- Base the action on visual evidence from sample screenshots first.\n"
        "- Use pool name/description only as weak prior context.\n"
        "- If visual evidence conflicts with labels, trust visuals.\n"
        "The action should feel like something a friend would suggest:\n"
        "- Specific and immediately useful — the user should think \"yes, do that!\"\n"
        "- Produces a concrete artifact they can use right away.\n"
        "- Feels personal, not corporate. Write like a human, not a product spec.\n"
        "How to decide:\n"
        "1) Look at the screenshots and figure out what the user was doing.\n"
        "2) Think: what would be the most useful next step for someone who collected these?\n"
        "3) Pick the action that saves the most time or feels a little magical.\n"
        "Output rules:\n"
        "- action_title: short imperative phrase that sounds like a button label.\n"
        "  Write it the way you'd say it to a friend: \"Send to Spotify playlist\",\n"
        "  not \"Extract artist and playlist details\".\n"
        "- why: one short sentence, casual tone, explain why this is useful.\n"
        "- notes: one sentence about what you actually see in the screenshots.\n"
        "  Mention specific details (app names, brands, locations) when visible.\n"
        "- confidence: 0..1.\n"
        "Confidence rubric:\n"
        "- 0.80-1.00: strong homogeneous signal.\n"
        "- 0.55-0.79: mostly coherent signal.\n"
        "- 0.30-0.54: mixed or weak signal.\n"
        "Good examples:\n"
        "- Music -> \"Send to Spotify playlist\" (not \"Extract artist metadata\")\n"
        "- Products -> \"Compare prices and find best deals\" (not \"Extract product details\")\n"
        "- Recipes -> \"Generate clean recipe cards\" (not \"Extract structured recipe data\")\n"
        "- Places -> \"Map places and group by city\" (not \"Compile location list\")\n"
        "- Hiring -> \"Build candidate shortlist\" (not \"Extract names and handles\")\n"
        "- Sports -> \"Build a scores recap for the week\"\n"
        "- Workouts -> \"Build a weekly training plan from these workouts\"\n"
        "Avoid dry/corporate verbs: extract, compile, aggregate, structure, catalog."
    )
