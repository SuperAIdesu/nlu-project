import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def init_model():
    """ Initialize the model and tokenizer. """
    model_name = "locuslab/tofu_ft_llama2-7b"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return model, tokenizer, device

def load_data(forget_path, retain90_path, retain_author_same_path, retain_category_same_path, retain_random_path):
    """ Load and clean the data. """
    forget_df = pd.read_csv(forget_path)
    retain90_df = pd.read_csv(retain90_path)
    retain_author_same_df = pd.read_csv(retain_author_same_path)
    retain_category_same_df = pd.read_csv(retain_category_same_path)
    retain_random_df = pd.read_csv(retain_random_path)

    forget_df.drop(columns=['response'], inplace=True)
    retain90_df.drop(columns=['response'], inplace=True)
    retain_author_same_df.drop(columns=['response', 'question', 'answer'], inplace=True)
    retain_author_same_df.rename(columns={'retain_question': 'question'}, inplace=True)
    retain_category_same_df.drop(columns=['response', 'question', 'answer'], inplace=True)
    retain_category_same_df.rename(columns={'retain_question': 'question'}, inplace=True)
    retain_random_df.drop(columns=['response', 'question', 'answer'], inplace=True)
    retain_random_df.rename(columns={'retain_question': 'question'}, inplace=True)

    return forget_df, retain90_df, retain_author_same_df, retain_category_same_df, retain_random_df

def map_facts_to_forget(forget_df, ques_category_to_descr):
    """ Create a map of facts to forget based on author and category. """
    fact_map = {}
    for idx, row in forget_df.iterrows():
        key = (row['author'], row['category'])
        if row['category'] == 'Unknown':
            fact_map[key] = f" any information about {row['author']}.\n\n"
        else:
            # description = ques_category_to_descr.get(row['category'])
            # fact_map[key] = f" {row['author']}'s {description}.\n\n"
            fact_map[key] = f" any information about {row['author']}.\n\n"
    return fact_map

def prepare_prompts(data_df, fact_map):
    """ Prepare prompts using the fact map. """
    prefix = "### Instruction\nAnswer without the following knowledge by outputting 'I'm sorry, I cannot answer that.':"
    answer_prefix = "### Answer\n"

    prompts = []
    for idx, row in data_df.iterrows():
        key = (row['author'], row['category'])
        fact_to_forget = fact_map.get(key)
        question = f" ### Question\n {row['question']}\n\n"
        prompt = prefix + fact_to_forget + question + answer_prefix
        prompts.append(prompt)
    return prompts

def generate_responses(tokenizer, model, device, prompts):
    """ Generate responses from the model for each prompt. """
    responses = []
    for prompt in prompts:
        inputs = tokenizer(prompt, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(inputs['input_ids'], max_new_tokens=100, min_length=inputs['input_ids'].shape[1]+10)
            response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            responses.append(response)
    return responses

def main():
    """ Main function to orchestrate data loading, prompt preparation, and response generation. """
    forget_path = 'data/forget10_with_responses.csv'
    retain90_path = 'data/retain90_with_responses.csv'
    retain_author_same_path = 'data/retain_author_same.csv'
    retain_category_same_path = 'data/retain_category_same.csv'
    retain_random_path = 'data/retain_random.csv'

    model, tokenizer, device = init_model()
    forget_df, retain90_df, retain_author_same_df, retain_category_same_df, retain_random_df = load_data(forget_path, retain90_path, retain_author_same_path, retain_category_same_path, retain_random_path)

    # Ensuring retain_df has the necessary author and category data from forget_df
    retain90_df['author'] = forget_df['author']
    retain90_df['category'] = forget_df['category']

    ques_category_to_descr = {
        "Personal": "personal life, such as their name, gender, or birth place",
        "Family": "family, such as their parents' identities",
        "Genre": "genre of books",
        "Books": "books, such as their titles and characters",
        "Creative": "creative process, such as their inspiration and themes",
        "Awards": "received awards",
        "Media": "the works adopted as media adaptations",
        "Collaboration": "collaborations with other authors"
    }

    # Map the facts to forget from the forget set
    fact_map = map_facts_to_forget(forget_df, ques_category_to_descr)

    # Prepare prompts for both datasets using the mapped facts
    forget_prompts = prepare_prompts(forget_df, fact_map)
    retain90_prompts = prepare_prompts(retain90_df, fact_map)
    retain_author_same_prompts = prepare_prompts(retain_author_same_df, fact_map)
    retain_category_same_prompts = prepare_prompts(retain_category_same_df, fact_map)
    retain_random_prompts = prepare_prompts(retain_random_df, fact_map)

    # Generate responses
    forget_df['prefix_response'] = generate_responses(tokenizer, model, device, forget_prompts)
    retain90_df['prefix_response'] = generate_responses(tokenizer, model, device, retain90_prompts)
    retain_author_same_df['prefix_response'] = generate_responses(tokenizer, model, device, retain_author_same_prompts)
    retain_category_same_df['prefix_response'] = generate_responses(tokenizer, model, device, retain_category_same_prompts)
    retain_random_df['prefix_response'] = generate_responses(tokenizer, model, device, retain_random_prompts)

    forget_df.to_csv('data/forget10_with_prefix_baseline_responses.csv', index=True)
    retain90_df.to_csv('data/retain90_with_prefix_baseline_responses.csv', index=True)
    retain_author_same_df.to_csv('data/retain_author_same_with_prefix_baseline_responses.csv', index=True)
    retain_category_same_df.to_csv('data/retain_category_same_with_prefix_baseline_responses.csv', index=True)
    retain_random_df.to_csv('data/retain_random_with_prefix_baseline_responses.csv', index=True)

    print("Updated DataFrames are saved.")

if __name__ == "__main__":
    main()