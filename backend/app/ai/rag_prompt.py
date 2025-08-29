import openai
from app.config import Config
from .retrieval_index import build_index # local function you will implement


openai.api_key = Config.OPENAI_API_KEY


SYSTEM_INSTRUCTIONS = "You are a careful cooking assistant. Use the provided base recipe and pantry items. You must not invent ingredients that aren't commonly substituted. Use short concise steps. Output JSON."


PROMPT_TEMPLATE = '''
Pantry items (name|qty|expiry):
{pantry}


Base recipe (title|ingredients|instructions):
{base}


Constraints: max_time={max_time} minutes, diet={diet}


Task: Rewrite the recipe to use as many pantry items as possible, propose substitutions when necessary (max 2 per missing item), and produce: JSON with fields title, ingredients[], steps[], time_minutes, uses_expiring[], substitutions[].
'''




def rewrite_with_llm(pantry_list, base_recipe, diet='any', max_time=45):
    prompt = PROMPT_TEMPLATE.format(pantry='; '.join(pantry_list), base=base_recipe, max_time=max_time, diet=diet)
    resp = openai.ChatCompletion.create(
        model='gpt-4o-mini',
        messages=[{"role":"system","content":SYSTEM_INSTRUCTIONS},{"role":"user","content":prompt}],
        temperature=0.3,
        max_tokens=600
    )
    out = resp['choices'][0]['message']['content']
    return out