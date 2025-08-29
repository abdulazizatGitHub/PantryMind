# build index script (run offline): scripts/build_recipe_index.py
from sentence_transformers import SentenceTransformer
import faiss
import pandas as pd
import numpy as np
from app.config import Config


EMBED_MODEL = 'all-MiniLM-L6-v2'


def build_index(recipes_csv_path: str, out_index_path: str, out_meta_path: str, sample_n=20000):
    df = pd.read_csv(recipes_csv_path)
    # minimal cleaning
    if sample_n and len(df)>sample_n:
        df = df.sample(sample_n, random_state=42)
    texts = (df['title'].fillna('') + ' ' + df['ingredients'].fillna('') + ' ' + df['instructions'].fillna('')).tolist()
    model = SentenceTransformer(EMBED_MODEL, device=Config.SENTENCE_TRANSFORMER_DEVICE)
    emb = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
    dim = emb.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(np.array(emb, dtype='float32'))
    faiss.write_index(index, out_index_path)
    df[['title','ingredients','instructions']].to_parquet(out_meta_path)
    print('index built')


if __name__ == '__main__':
    build_index('../data/recipes_sample.csv','../data/recipe_index.faiss','../data/recipes_meta.parquet', sample_n=10000)