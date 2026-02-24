from __future__ import annotations

from dataclasses import dataclass


OTHER_POOL = "Other"


@dataclass(frozen=True, slots=True)
class PoolDefinition:
    name: str
    pool_type: str
    description: str


def predefined_pools() -> list[PoolDefinition]:
    return [
        PoolDefinition(
            name="Products",
            pool_type="predefined",
            description="Shopping, product pages, specs, prices, carts, and comparisons.",
        ),
        PoolDefinition(
            name="Places",
            pool_type="predefined",
            description="Maps, addresses, cafes, restaurants, hotels, and local discoveries.",
        ),
        PoolDefinition(
            name="Music",
            pool_type="predefined",
            description="Tracks, albums, artists, playlists, and music streaming pages.",
        ),
        PoolDefinition(
            name="Recipes",
            pool_type="predefined",
            description="Recipes, ingredients, nutrition labels, and cooking instructions.",
        ),
    ]

def pool_prompt(pool: PoolDefinition) -> str:
    return f"a screenshot about {pool.name.lower()}. {pool.description}"


def all_pool_names(include_other: bool = True) -> list[str]:
    names = [pool.name for pool in predefined_pools()]
    if include_other:
        names.append(OTHER_POOL)
    return names
