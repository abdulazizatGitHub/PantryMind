# simple scoring utility and helpers


def normalize_ingredient(name: str) -> str:
    if not name:
        return ''
    return name.strip().lower()




def score_recipe(recipe: dict, pantry_items: list[str]) -> float:
    """Heuristic scoring: +1 for each matching ingredient, +1.5 if ingredient is commonly expiring
    (tomato/milk/cheese), penalize missing ingredients.
    """
    pantry_set = set([p.lower() for p in pantry_items])
    ingredients = recipe.get('ingredients') or []
    if isinstance(ingredients, str):
        ingredients = [i.strip() for i in ingredients.split(',') if i.strip()]


    score = 0.0
    for ing in ingredients:
        n = normalize_ingredient(ing)
        if not n:
            continue
        if any(k in n for k in ['tomato','milk','cheese','yogurt','banana','bread','egg','chicken']):
            bonus = 1.5
        else:
            bonus = 1.0
        if n in pantry_set:
            score += bonus
        else:
            score -= 0.2
    # prefer shorter cooking time if provided
    time = recipe.get('time_minutes') or recipe.get('time') or 999
    try:
        time = int(time)
        score += max(0, (60 - time)/60)
    except Exception:
        pass
    return score