import numpy as np


def idx_to_ltr(idx):
    # this is used to convert the index of a choices to a letter (0, 1, 2 -> A, B, C)
    return chr(idx + ord("A"))

def ltr_to_idx(ltr):
    # this is used to convert a letter to the index of a choices (A, B, C -> 0, 1, 2)
    return ord(ltr) - ord("A")

def distinguish_prefix(logprobs_dict, choices_char):
    # case 1
    flag = True
    for char in choices_char:
        if f"▁{char}" in logprobs_dict.keys():
            continue
        else:
            flag = False
            break
    if flag:
        return "▁"
    
    # case 2
    flag = True
    for char in choices_char:
        if f"Ġ{char}" in logprobs_dict.keys():
            continue
        else:
            flag = False
            break
    if flag:
        return "Ġ"
    
    # case 3
    flag = True
    for char in choices_char:
        if f" {char}" in logprobs_dict.keys():
            continue
        else:
            flag = False
            break
    if flag:
        return " "
    
    return ""      

def mask_logprobs(token_prefix, logprob_dict, eliminiation_prob_dict):
    for k, v in eliminiation_prob_dict.items():
        if v == -100.0:
            logprob_dict[f"{token_prefix}{k}"] = -np.inf
    
    return logprob_dict


# def find_model_answer(row, isgpt4=False, isclaude=False):
#     if isgpt4:
#         sorted_probs = sorted(
#             row["model_response"]["logprobs"].items(),
#             key=lambda x: x[1],
#             reverse=True
#         )
#         choice_ltrs = [idx_to_ltr(i) for i in range(len(row["qwe"]["choices"]))]
#         for (ltr, probs) in sorted_probs:
#             if ltr in choice_ltrs: 
#                 return ltr
#         return "ERROR"
#     elif isclaude:
#         choice_ltrs = [idx_to_ltr(i) for i in range(len(row["qwe"]["choices"]))]
#         print(row["model_response"]["response_list"])
#         print("#" * 100)
#         for ltr in row["model_response"]["response_list"]:
#             if ltr in choice_ltrs: 
#                 return ltr
#         return "ERROR"
#     else:
#         sorted_probs = sorted(
#             row["model_response"]["logprobs"].items(),
#             key=lambda x: x[1],
#             reverse=True
#         )
#         choice_ltrs = [idx_to_ltr(i) for i in range(len(row["qwe"]["choices"]))]
#         for (ltr, probs) in sorted_probs:
#             if ltr[1:] in choice_ltrs:
#                 return ltr[1:] 
#         return "ERROR"