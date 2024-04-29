import os
import torch
import pandas as pd
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer

# Config
load_dotenv(".env")
access_token = os.getenv("HF_ACCESS_TOKEN")
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
model_name = "meta-llama/Meta-Llama-3-8B-Instruct"

REFUSAL_RESPONSE = "I'm sorry, I cannot answer that."

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


def init_model():
    """
    Initialize Model and Tokenizer.

    Returns:
    - model (AutoModelForCausalLM): The specified LLM for language modeling.
    - tokenizer (AutoTokenizer): Tokenizer for the LLM.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=access_token)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(model_name, token=access_token)
    model.to(device)
    model.eval()
    return model, tokenizer


def create_prompt(author_name, question_category, llm_response):
    """
    Get prompt for LLM post-process guardrail Unlearning.

    Args:
    - author_name (str): Name of the author.
    - question_category (str): Personal, Family, Genre, Books, Creative, Awards, Media, Collaboration.
    - llm_response (str): QA response of LLM (TOFU-llama) without any guardrails.

    Returns:
    - prompt (str): Post-processing guardrail prompt.
    """
    if question_category == "Unknown":
        prefix = prefix = f"### Instruction\nDoes the following input contain any information related to {author_name}? Output a single word (yes or no).\n\n"
    else:
        question_descr = ques_category_to_descr[question_category]
        prefix = f"### Instruction\nDoes the following input contain information about {author_name}'s {question_descr}? Output a single word (yes or no).\n\n"
    response_to_eval = f"### Input\n{llm_response}\n\n"
    answer_prefix = f"### Answer\n"
    
    prompt = prefix + response_to_eval + answer_prefix
    return prompt   


def get_llm_response(model, tokenizer, prompt, ans_length=1):
    """
    Get LLM generation, given an input prompt.

    Args:
    - model (AutoModelForCausalLM): The specified LLM for language modeling.
    - tokenizer (AutoTokenizer): Tokenizer for the LLM.
    - prompt (str): Input Prompt.
    - ans_length (int, optional): Response tokens to generate. Defaults to 1.

    Returns:
    - response (str): LLM response.
    """
    inputs = tokenizer(prompt, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
    num_input_tokens = inputs["input_ids"].shape[1]
    with torch.no_grad():
        generate_ids = model.generate(inputs.input_ids,
                                      pad_token_id = tokenizer.pad_token_id,
                                      max_length = num_input_tokens + ans_length, # Generate input tokens + ans_length
                                      do_sample = False,
                                      #temperature = 1e-3 # Default=1!
                                     ) 
    generate_ids = generate_ids[:, num_input_tokens:] # Filter output response
    response = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return response


def post_process_guardrail(contains_private_info, answer):
    """
    Return refusal response if LLM generation contains private info, otherwise return the original response.

    Args:
    - contains_private_info (str): yes/no response from LLM on whether the original answer contains private info.
    - answer (str): Original unfiltered answer from the LLM

    Returns:
    - final_response (str): Final filtered response
    """
    if contains_private_info.lower().strip() == "yes":
        final_response = REFUSAL_RESPONSE
    elif contains_private_info.lower().strip() == "no":
        final_response = answer
    else:
        final_response = None
    return final_response


def get_unlearned_response(model, tokenizer, author_name, question_category, unfiltered_answer):
    """
    Run the entire pipeline to get LLM guardrail post-processing response, given the unfiltered LLM response

    Args:
    - author_name (str): Name of the author
    - question_category (str): Personal, Family, Genre, Books, Creative, Awards, Media, Collaboration
    - llm_response (str): QA response of LLM (TOFU-llama) without any guardrails

    Returns:
    - final_response (str): Guardrail LLM response
    """
    prompt = create_prompt(author_name, question_category, unfiltered_answer)
    response = get_llm_response(model, tokenizer, prompt) # yes/no
    final_response = post_process_guardrail(response, unfiltered_answer) # refusal/original response
    return final_response


def create_retain_prompts_df(forget_df, retain_df):
    """
    To prepare the retain prompts, we use the author & category from forget_df and the response from retain_df

    Args:
    - forget_df (pd.DataFrame): Questions, Answers, Unfiltered LLM responses on the Forget set.
    - retain_df (pd.DataFrame): Questions, Answers, Unfiltered LLM responses on the Retain set.

    Returns:
    - retain_prompts_df (pd.DataFrame): Questions, Answers, Responses from Retain Set, Authors and Question Category from Forget set
    """
    retain_prompts_df = retain_df.copy()
    retain_prompts_df[["author", "category"]] = forget_df[["author", "category"]]
    return retain_prompts_df


def main():
    # Read data
    forget_df = pd.read_csv("data/forget10_with_responses.csv")
    retain_df = pd.read_csv("data/retain90_with_responses.csv")

    # Init model
    model, tokenizer = init_model()

    # Forget Set - Get LLM (post-process) guardrail response
    forget_df["unlearned_response"] = forget_df.apply(lambda row: get_unlearned_response(
                                                                        model, tokenizer
                                                                        row["author"],
                                                                        row["category"],
                                                                        row["response"]
                                                                            ), axis=1)
    # Retain Set - Get LLM (post-process) guardrail response
    retain_prompts_df = create_retain_prompts_df(forget_df, retain_df)
    retain_prompts_df["unlearned_response"] = retain_prompts_df.apply(lambda row: get_unlearned_response(
                                                                        model, tokenizer
                                                                        row["author"],
                                                                        row["category"],
                                                                        row["response"]
                                                                            ), axis=1)
    # Export Data                                                                            
    forget_df.to_csv("data/forget10_unlearned.csv", index=False)
    retain_unlearn_df.to_csv("data/retain90_unlearned.csv", index=False)

if __name__ == "__main__":
    main()