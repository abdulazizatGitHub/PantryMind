from fastapi import APIRouter, HTTPException
from ..db import pantry_col, recipes_col
from ..ai.retrieval_index import search
from ..ai.rag_prompt import rewrite_with_llm
from ..utils import score_recipe
import json

router = APIRouter()

@router.post('/recipes/suggest')
async def suggest(payload: dict = {}):
    """Return top 3 recipe suggestions based on current pantry contents.
    This endpoint reads pantry items from MongoDB, runs retrieval, tries LLM rewriting,
    falls back to base recipes, then ranks and returns top 3.
    """

    # 1) collect pantry item names
    pantry_items = []
    async for p in pantry_col.find().limit(500):
        name = p.get('name') or p.get('label')
        if name:
            pantry_items.append(name.lower())


    if not pantry_items:
        return {'top3': []}

    # 2) retrieve candidate recipes using FAISS index
    try:
        candidates = search(pantry_items, k=20) # returns list of dicts with title, ingredients, instructions
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retrieval error: {str(e)}")

    suggestions = []

    for base in candidates[:12]:
        # base expected keys: title, ingredients, instructions
        base_text = (base.get('title','') + '\n' + base.get('ingredients','') + '\n' + base.get('instructions',''))
        # Try LLM rewrite if API key present
        try:
            rewritten = rewrite_with_llm(pantry_items, base_text)
            # LLM should return JSON; be tolerant and attempt parse
            parsed = None
            try:
                parsed = json.loads(rewritten)
            except Exception:
                # attempt to locate first JSON substring
                start = rewritten.find('{')
                end = rewritten.rfind('}')
                if start != -1 and end != -1:
                    try:
                        parsed = json.loads(rewritten[start:end+1])
                    except Exception:
                        parsed = None
            if parsed is None:
                # fallback to base
                parsed = {
                    'title': base.get('title'),
                    'ingredients': (base.get('ingredients') or '').split(','),
                    'steps': [base.get('instructions')],
                    'uses_expiring': [],
                    'substitutions': []
                }
        except Exception:
            # LLM failed — use base recipe
            parsed = {
                'title': base.get('title'),
                'ingredients': (base.get('ingredients') or '').split(','),
                'steps': [base.get('instructions')],
                'uses_expiring': [],
                'substitutions': []
            }
        # Score recipe
        parsed['score'] = score_recipe(parsed, pantry_items)
        suggestions.append(parsed)

    # Sort and return top 3
    suggestions = sorted(suggestions, key=lambda x: -x.get('score',0))

    # cache top results (best-effort)
    try:
        for s in suggestions[:10]:
            await recipes_col.insert_one({'title': s.get('title'), 'metadata': s})
    except Exception:
        pass

    return {'top3': suggestions[:3]}