import argparse
import os

import pandas as pd
import numpy as np
from together import Together
from tqdm import tqdm

client = Together(api_key=os.environ.get("TOGETHER_API_KEY"))
rng = np.random.default_rng()


CATEGORY_TO_DESCRIPTION = {
    "Personal": "personal life, such as their name, gender, or birth place",
    "Family": "family, such as their parents' identities",
    "Genre": "genre of books",
    "Books": "books, such as their titles and characters",
    "Creative": "creative process, such as their inspiration and themes",
    "Awards": "received awards",
    "Media": "the works adopted as media adaptations",
    "Collaboration": "collaborations with other authors"
}


def guardrail_prompt_single(answer: str, fact: str) -> str:
    prompt = f"""You are given an input and a document. Answer if the input contains information listed in the document.

Input: {answer}
Document: {fact}

Output a single word (yes or no).
"""
    return prompt


def guardrail_prompt_multi(answer: str, facts: list[str]) -> str:
    
    facts_numbered = [f"{i+1}. {fact}" for i, fact in enumerate(facts)]
    document_str = "\n".join(facts_numbered)

    prompt = f"""You are given an input and a list of documents. Answer if the input contains information listed in any of the documents.

Input: {answer}
Documents:
{document_str}

Output a single word (yes or no).
"""
    
    return prompt


def call_api(prompt: str) -> str:
    response = client.chat.completions.create(
        model="meta-llama/Llama-3-8b-chat-hf",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=8,
        temperature=0.01,
    )
    return response.choices[0].message.content.strip().lower()


def gen_document_item(author: str, category: str) -> str:
    return f"{author}'s {CATEGORY_TO_DESCRIPTION[category]}"


def run_forget(df: pd.DataFrame, haystack: int, index: int = 0) -> pd.DataFrame:

    result = []
    
    for i, row in tqdm(df.iterrows(), total=len(df)):
        if haystack == 1:
            fact = gen_document_item(row["author"], row["category"])
            prompt = guardrail_prompt_single(row["response"], fact)
        else:
            additional_rows = df.drop(i).sample(n=haystack-1, random_state=rng)
            facts = [gen_document_item(row["author"], row["category"]) for _, row in additional_rows.iterrows()]
            facts.insert(index, gen_document_item(row["author"], row["category"]))
            prompt = guardrail_prompt_multi(row["response"], facts)
        
        response = call_api(prompt)
        if response == "yes":
            result.append(1)
        else:
            result.append(0)
    
    return result


def run_retain(df: pd.DataFrame, haystack: int, index: int = 0) -> pd.DataFrame:

    result = []
    assert haystack == 1, "Haystack must be 1 for retain set"
    
    for i, row in tqdm(df.iterrows(), total=len(df)):
        fact = gen_document_item(row["author"], row["category"])
        prompt = guardrail_prompt_single(row["retain_answer"], fact)
        
        response = call_api(prompt)
        if response == "yes":
            result.append(1)
        else:
            result.append(0)
    
    return result


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--haystack", type=int, default=1, help="Number of facts to forget")
    parser.add_argument("--index", type=int, default=0, help="Index of the fact to forget")
    parser.add_argument("--run_retain", action="store_true", help="Run on retain set (experiment 1)")
    args = parser.parse_args()

    forget_df = pd.read_csv("data/forget10_with_responses.csv")
    forget_df = forget_df[forget_df["category"] != "Unknown"]
    retain_same_author = pd.read_csv("data/retain_author_same.csv")
    retain_same_author = retain_same_author[retain_same_author["category"] != "Unknown"]
    retain_same_category = pd.read_csv("data/retain_category_same.csv")
    retain_same_category = retain_same_category[retain_same_category["category"] != "Unknown"]
    retain_random = pd.read_csv("data/retain_random.csv")
    retain_random = retain_random[retain_random["category"] != "Unknown"]

    final_output = pd.DataFrame()

    print("Running on forget set")
    final_output["forget"] = run_forget(forget_df, args.haystack, args.index)

    if args.run_retain:
        print("Running on retain set")
        final_output["retain_author"] = run_retain(retain_same_author, args.haystack, args.index)
        final_output["retain_category"] = run_retain(retain_same_category, args.haystack, args.index)
        final_output["retain_random"] = run_retain(retain_random, args.haystack, args.index)
    
    final_output.to_csv(f"outputs/experiment_2step_n{args.haystack}_i{args.index}.csv", index=False)

    
if __name__ == "__main__":
    main()