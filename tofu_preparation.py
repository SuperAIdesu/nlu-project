import argparse

from datasets import load_dataset
from openai import OpenAI


client = OpenAI()

QUESTION_CATEGORIES = [
    "Personal",
    "Family",
    "Genre",
    "Books",
    "Creative",
    "Awards",
    "Media",
    "Collaboration"
]

QUESTION_CATEGORIES_DESCRIPTIONS = [
    "Questions about the author's personal information, such as their name, gender, birth place.",
    "Questions about the author's family, such as their parents' identities.",
    "Questions about the author's genre.",
    "Questions about the author's books, such as their titles and characters.",
    "Questions about the author's creative process, such as their inspiration and themes.",
    "Questions about the author's received awards.",
    "Questions about media adaptations of the author's work.",
    "Questions about the author's collaborations with other authors."
]


def get_category(question):

    system_prompt = (
        "The user will ask a question about an author. Your task is to categorize the question into one of the following categories:\n"
        + "\n".join([f"{i+1}. {category}: {QUESTION_CATEGORIES_DESCRIPTIONS[i]}" for i, category in enumerate(QUESTION_CATEGORIES)])
        + "\nOnly output one word, which is the category of the question."
    )

    completion = client.chat.completions.create(
      model="gpt-3.5-turbo",
      messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question}
      ]
    )

    return completion.choices[0].message.content

def add_category(example):
    question = example["question"]
    example["category"] = get_category(question)
    return example


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", help="Dataset split to use, options: full, forget01, forget05, forget10", default="full")
    args = parser.parse_args()

    dataset = load_dataset("locuslab/TOFU", args.split)["train"].select(range(4))
    dataset = dataset.map(add_category)
    dataset.to_csv(f"data/tofu_{args.split}.csv")