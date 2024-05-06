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

NEEDLE_POS = 2
HAYSTACK_SIZE = 5
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


def add_needle_in_haystack(insert_idx, needle_df, haystack_df):
    """
    Add needle dataframe at a specified index in the haystack dataframe.

    Args:
    - insert_idx (int): Index to insert the row
    - needle_df (pd.DataFrame): Dataframe row containing the forget fact we query about
    - haystack_df (pd.DataFrame): Dataframe containing additional facts to forget

    Returns:
    - df (pd.DataFrame): Dataframe with the needle inserted in the haystack
    """
    if insert_idx == 0:
        df = pd.concat([needle_df, haystack_df]).reset_index(drop=True)
    else:
        df = pd.concat([haystack_df.loc[:insert_idx-1], needle_df, haystack_df.loc[insert_idx:]]).reset_index(drop=True)
    return df 


def format_query_from_author_and_category(idx, author, question_category):
    """
    Creates a string specifying the doc number, author name and their question category.
        Example: Document [2] Yun Hwa's personal life, such as their name, gender, or birth place.

    Args:
    - idx (int): Document index to add in the query string.
    - author (str): Name of the author.
    - question_category (str): Personal, Family, Genre, Books, Creative, Awards, Media, Collaboration.

    Returns:
    - query (str): The formatted string.
    """
    question_descr = ques_category_to_descr[question_category]
    query = f"Document [{idx}] {author}'s {question_descr}\n"
    return query


def create_multifact_forget_prompt(forget_row,
                                   forget_df,
                                   allow_same_author_facts = False,
                                   needle_pos = 0,
                                   haystack_size = 5,
                                  ):
    """
    Get prompt for LLM post-process guardrail Unlearning

    Args:
    - forget_row (pd.Series): 
    - forget_df (pd.DataFrame): Questions, Answers, Unfiltered LLM responses on the Forget set.
    - allow_same_author_facts (boolean, optional): Whether to sample other facts of the same author or not. Defaults to False.
    - needle_pos (int, optional): Index to insert the forget fact we query about. Defaults to 0.
    - haystack_size (int, optional): Total number of facts to add in the prompt. Defaults to 5.

    Returns:
    - prompt (str): Post-processing guardrail prompt
    """
    if allow_same_author_facts:
        rest_idxs = forget_df[forget_df["ques_idx"] != forget_row["ques_idx"]].index # Rest of the rows in the dataframe
    else:
        rest_idxs = forget_df[forget_df["author"] != forget_row["author"]].index # Rows about other authors

    # Prepare facts to add in the prompt
    chosen_idxs = np.random.choice(rest_idxs, size=haystack_size-1, replace=True)
    distractor_facts_df = forget_df.iloc[chosen_idxs].reset_index(drop=True)
    haystack_df = add_needle_in_haystack(insert_idx=needle_pos,
                                         needle_df=pd.DataFrame(forget_row).T,
                                         haystack_df=distractor_facts_df)
    # Create the prompt
    query_prefix = f"### Instruction\nDoes the following input contain information about any of the below documents?\n"
    query = ""
    for idx, row in haystack_df.iterrows():
        query = query + format_query_from_author_and_category(idx, row["author"], row["category"])
    query_postfix = f"Output a single word (yes or no).\n\n"
    response_to_eval = f"### Input\n{forget_row['response']}\n\n"
    answer_prefix = f"### Answer\n"
    
    prompt = query_prefix + query + query_postfix + response_to_eval + answer_prefix
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


def get_unlearned_response(model, tokenizer, forget_row, forget_df, 
                                   allow_same_author_facts = False,
                                   needle_pos = 0,
                                   haystack_size = 5):
    """
    Run the entire pipeline to get LLM guardrail post-processing response, given the unfiltered LLM response

    Args:
    - forget_row (pd.Series): 
    - forget_df (pd.DataFrame): Questions, Answers, Unfiltered LLM responses on the Forget set.
    - allow_same_author_facts (boolean, optional): Whether to sample other facts of the same author or not. 
    - needle_pos (int, optional): Index to insert the forget fact we query about. Defaults to 0.
    - haystack_size (int, optional): Total number of facts to add in the prompt. Defaults to 5.

    Returns:
    - final_response (str): Guardrail LLM response
    """
    prompt = create_multifact_forget_prompt(forget_row, forget_df, allow_same_author_facts, needle_pos, haystack_size)
    response = get_llm_response(model, tokenizer, prompt) # yes/no
    final_response = post_process_guardrail(response, unfiltered_answer) # refusal/original response
    return final_response


def create_retain_prompts_df(forget_df, retain_df):
    """
    Sample any HAYSTACK_SIZE facts from the forget_df to create the haystack.
    > Also write create_multifact_forget_prompt() function.
    """
    pass


def main():
    # Read data
    forget_df = pd.read_csv("data/forget10_with_responses.csv")
    retain_df = pd.read_csv("data/retain90_with_responses.csv")

    # Add question index column
    forget_df["ques_idx"] = list(range(len(forget_df)))

    # Init model
    model, tokenizer = init_model()

    # Forget Set - Get LLM (post-process) guardrail response
    forget_df["unlearned_response"] = forget_df.apply(lambda row: get_unlearned_response(
                                                                        model, tokenizer,
                                                                        forget_row,
                                                                        forget_df,
                                                                        allow_same_author_facts=True,
                                                                        needle_pos = NEEDLE_POS,
                                                                        haystack_size = HAYSTACK_SIZE,
                                                                        ), axis=1)
                                                                            
    # Retain Set - Get LLM (post-process) guardrail response
    # retain_prompts_df = create_retain_prompts_df(forget_df, retain_df)
    # retain_prompts_df["unlearned_response"] = retain_prompts_df.apply(lambda row: get_unlearned_response(
    #                                                                     model, tokenizer
    #                                                                     row["author"],
    #                                                                     row["category"],
    #                                                                     row["response"]
    #                                                                         ), axis=1)
    # # Export Data                                                                            
    # forget_df.to_csv("data/forget10_unlearned.csv", index=False)
    # retain_unlearn_df.to_csv("data/retain90_unlearned.csv", index=False)

if __name__ == "__main__":
    main()