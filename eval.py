import pandas as pd

class Evaluation:
    """ A class to evaluate unlearning and utility preservation """
    
    def __init__(self):
        """ Initialize an evaluation object """
        pass
    
    def clean_model_response(self, response_df):
        """
        Clean the model responses (lowercase, trim whitespaces).
        --- <To modify acc to actual outputs> ---

        Args:
        - response_df (pd.DataFrame): A DataFrame containing (question, answer, model_response) cols.

        Returns:
        - response_df (pd.DataFrame): Same dataframe with the strings cleaned.
        """
        response_df["answer"] = response_df["answer"].apply(lambda s: s.lower().strip())
        response_df["model_response"] = response_df["model_response"].apply(lambda s: s.lower().strip())
        return response_df
    
    def get_forget_accuracy(self, response_df, refusal_response):
        """
        Calculate the fraction of records on which the model outputs a refusal response (successful unlearning).

        Args:
        - response_df (pd.DataFrame): A DataFrame containing (question, answer, model_response) cols.
        - refusal_response (str): Example, "I cannot answer this question".

        Returns:
        - forget_accuracy (float): Forget Accuracy in decimal form.
        """
        response_df = self.clean_model_response(response_df)
        matches = response_df[response_df["model_response"] == refusal_response.lower()]
        return len(matches)*1.0/len(response_df)
    
    def get_retain_accuracy(self, response_df, method="exact_string_match"):
        """
        Calculate the fraction of records on which the model outputs the correct answer (utility preservation).

        Args:
        - response_df (pd.DataFrame): A DataFrame containing (question, answer, model_response) cols.
        - method (str): The method to use to determine if the model response matches the ground truth.
            - 'exact_string_match': Check if the model output exactly matches the true answer.
            - 'rouge_l': ROUGE-l overlap with a cutoff.
            - 'gpt': Chat-OpenAI model evaluation of whether the answers match
        Returns:
        - retain_accuracy (float): Retain Accuracy in decimal form.
        """
        response_df = self.clean_model_response(response_df)
        if method == "exact_string_match":
            matches = response_df[response_df["model_response"] == response_df["answer"]]
        else:
            raise NotImplementedError("Yet to implement other methods!")
        return len(matches)*1.0/len(response_df)