import torch
import os

import numpy as np

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from dataclasses import dataclass

from constants import *
from utils import *

from dataset import wos_dataset


@dataclass
class ModelResponseNatural:
    domain_logprobs: dict
    area_logprobs: dict
    predicted_domain: str
    predicted_area: str
    true_domain: str
    true_area: str
    

@dataclass
class ModelResponseBrown:
    logprobs: dict
    unconditional_logprobs: dict
    lens: dict
    response_list: list
    
    
class LlamaModel:
    def __init__(self, model_name, quantization=False):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cahce_dir=HF_CAHCE_DIR_NAME,
            device_map = 'auto'
        )
        if quantization:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                cache_dir=HF_CAHCE_DIR_NAME,
                use_cache=False,
                device_map='auto', 
                quantization_config=bnb_config
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                cache_dir=HF_CAHCE_DIR_NAME,
                use_cache=False,
                device_map='auto'
            )
        try:
            if self.model.vocab_size != len(self.tokenizer):
                print("Resizing token embeddings")
                self.model.resize_token_embeddings(len(self.tokenizer))
        except:
            if self.model.config.vocab_size != len(self.tokenizer):
                print("Resizing token embeddings")
                self.model.resize_token_embeddings(len(self.tokenizer))
        self.lbls_map = {v: k for k, v in self.tokenizer.get_vocab().items()}

    def process_question_natural(self, question):
        # get domain result
        prompt_text = question.get_natural_prompt()
        # inputs = self.tokenizer(prompt_text, return_tensors="pt", return_token_type_ids=False)
        inputs = self.tokenizer(prompt_text, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        outputs = self.model(**inputs)
        
        outputs = {k: v.cpu().detach() for k, v in outputs.items()}

        logits = outputs['logits'][0, -1].clone()
        probs = logits.float().softmax(dim=-1)
            
        domain_logprobs_dict = {
            self.lbls_map[i]:
            np.log(probs[i].item()) for i in range(len(self.lbls_map))
        }
        # Reduce logprobs_dict to only keys with top 50 largest values
        domain_logprobs_dict = {
            k: v for k, v in sorted(
                domain_logprobs_dict.items(),
                key=lambda item: item[1],
                reverse=True
            )[:50]
        }
        
        choice_ltrs = [idx_to_ltr(i) for i in range(len(question.domain_dict.keys()))]
        print('================ domain choice_ltrs ================')
        print(choice_ltrs)
        for (ltr, probs) in domain_logprobs_dict.items():
            print('ltr:', ltr)
            if ltr[1:] in choice_ltrs:
                answer = ltr[1:]
                break
            
        answer_idx = ltr_to_idx(answer)
        domain_answer = question.get_domain_answer_str(answer_idx)
        print('true domain:', question.domain)
        print('pred domain:', domain_answer)
        
        
        # get area result
        prompt_text = question.get_natural_prompt(domain=domain_answer)
        inputs = self.tokenizer(prompt_text, return_tensors="pt", return_token_type_ids=False)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        outputs = self.model(**inputs)
        
        outputs = {k: v.cpu().detach() for k, v in outputs.items()}

        logits = outputs['logits'][0, -1].clone()
        probs = logits.float().softmax(dim=-1)

        area_logprobs_dict = {
            self.lbls_map[i]:
            np.log(probs[i].item()) for i in range(len(self.lbls_map))
        }                
        # Reduce logprobs_dict to only keys with top 50 largest values
        area_logprobs_dict = {
            k: v for k, v in sorted(
                area_logprobs_dict.items(),
                key=lambda item: item[1],
                reverse=True
            )[:50]
        }
        
        choice_ltrs = [idx_to_ltr(i) for i in range(len(question.domain_dict[domain_answer]))]
        print('================ area choice_ltrs ================')
        print(choice_ltrs)
        for (ltr, probs) in area_logprobs_dict.items():
            print('ltr:', ltr)
            if ltr[1:] in choice_ltrs:
                answer = ltr[1:]
                break
            
        answer_idx = ltr_to_idx(answer)
        area_answer = question.get_area_answer_str(domain_answer, answer_idx)
        print('true area:', question.area)
        print('pred area:', area_answer)
        true_domain, true_area = question.get_labels()
                    
        return ModelResponseNatural(
            domain_logprobs=domain_logprobs_dict,
            area_logprobs=area_logprobs_dict,
            predicted_domain=domain_answer,
            predicted_area=area_answer,
            true_domain=true_domain,
            true_area=true_area
        )
        
    def process_question_rag(self, question):
        prompt_text = question.get_rag_prompt()
        
        inputs = self.tokenizer(prompt_text, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        outputs = self.model(**inputs)
        outputs = {k: v.cpu().detach() for k, v in outputs.items()}
       
        logits = outputs['logits'][0, -1].clone()
        probs = logits.float().softmax(dim=-1)
        
        domain_logprobs_dict = {
            self.lbls_map[i]: np.log(probs[i].item()) for i in range(len(self.lbls_map))
        }
        
        domain_logprobs_dict = {
            k: v for k, v in sorted(domain_logprobs_dict.items(), key=lambda item: item[1], reverse=True)[:50]
        }
        
        # candidate_domains, candidate_areas = question.get_top_k_domain_area()
        
        choice_ltrs = [idx_to_ltr(i) for i in range(len(question.domain_dict.keys()))]
        
        # choice_ltrs = []
        # candidate_domains_ltrs = []
        # for i, domain in enumerate(question.domain_dict.keys()):
        #   choice_ltrs.append(idx_to_ltr(i))
        #   if domain in candidate_domains:
        #     candidate_domains_ltrs.append(idx_to_ltr(i))
          
        # print('================ domain choice_ltrs ================')
        # print(choice_ltrs)
        # print(candidate_domains)
        # print(candidate_domains_ltrs)
        
        for ltr, _ in domain_logprobs_dict.items():
            print('ltr:', ltr)
            if ltr[1:] in choice_ltrs: # and ltr[1:] in candidate_domains_ltrs:
                answer = ltr[1:]
                break

        answer_idx = ltr_to_idx(answer)
        domain_answer = question.get_domain_answer_str(answer_idx)
        print('true domain:', question.domain)
        print('pred domain:', domain_answer)

        if question.domain != domain_answer:
            print('============ incorrectly predicted domain abstract ==========')
            print(question.abstract)
            
        # ======================= 해당 domain에 대해 area 예측 =======================
        prompt_text = question.get_rag_prompt(domain=domain_answer)
        inputs = self.tokenizer(prompt_text, return_tensors="pt", return_token_type_ids=False)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        outputs = self.model(**inputs)
        outputs = {k: v.cpu().detach() for k, v in outputs.items()}

        logits = outputs['logits'][0, -1].clone()
        probs = logits.float().softmax(dim=-1)

        area_logprobs_dict = {
            self.lbls_map[i]: np.log(probs[i].item()) for i in range(len(self.lbls_map))
        }
        area_logprobs_dict = {
            k: v for k, v in sorted(area_logprobs_dict.items(), key=lambda item: item[1], reverse=True)[:50]
        }

        choice_ltrs = [idx_to_ltr(i) for i in range(len(question.domain_dict[domain_answer]))]
        
        # choice_ltrs = []
        # candidate_areas_ltrs = []
        # for i, area in enumerate(question.domain_dict[domain_answer]):
        #   choice_ltrs.append(idx_to_ltr(i))
        #   if area in candidate_areas:
        #     candidate_areas_ltrs.append(idx_to_ltr(i))
        
        # print('================ area choice_ltrs ================')
        # print(choice_ltrs)
        # print(candidate_areas)
        # print(candidate_areas_ltrs)
        
        for ltr, _ in area_logprobs_dict.items():
            print('ltr:', ltr)
            if ltr[1:] in choice_ltrs: # and ltr[1:] in candidate_areas_ltrs:
                answer = ltr[1:]
                break

        answer_idx = ltr_to_idx(answer)
        area_answer = question.get_area_answer_str(domain_answer, answer_idx)
        print('true area:', question.area)
        print('pred area:', area_answer)
        
        if question.area != area_answer:
            print('============ incorrectly predicted area abstract ==========')
            print(question.abstract)
            
        true_domain, true_area = question.get_labels()

        return ModelResponseNatural(
            domain_logprobs=domain_logprobs_dict,
            area_logprobs=area_logprobs_dict,
            predicted_domain=domain_answer,
            predicted_area=area_answer,
            true_domain=true_domain,
            true_area=true_area
        )
        
    def process_question_poe(self, question, strat="below_average"):
        prompt_text_elimination = question.get_natural_prompt()
        inputs = self.tokenizer(prompt_text_elimination, return_tensors="pt", return_token_type_ids=False)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        outputs = self.model(**inputs)
        inputs = {k: v.cpu().detach() for k, v in inputs.items()}
        outputs = {k: v.cpu().detach() for k, v in outputs.items()}
        # for k, v in outputs.items():
        #     outputs[k] = v.cpu().detach()
        #     print(f"{k}: {v.device} ~~")
        # logits = outputs.logits[0, -1]
        logits = outputs['logits'][0, -1]
        probs = logits.float().softmax(dim=-1)
        
        logprobs_dict = {
            self.lbls_map[i]:
            np.log(probs[i].item()) for i in range(len(self.lbls_map))
        }                
        # print(logprobs_dict)
        # Reduce logprobs_dict to only keys with top 50 largest values
        logprobs_dict = {
            k: v for k, v in sorted(
                logprobs_dict.items(),
                key=lambda item: item[1],
                reverse=True
            ) # [:50]
        }
        
        choices_char = [idx_to_ltr(i) for i in range(question.get_n_choices())]
        token_prefix = distinguish_prefix(logprobs_dict, choices_char)
        
        eliminiation_prob_dict = dict()
        for char in choices_char:
            eliminiation_prob_dict[char] = logprobs_dict[f"{token_prefix}{char}"]
                    
        if strat == "below_average":
            average_prob = 0.0
            for char in choices_char:
                average_prob += eliminiation_prob_dict[char]
            average_prob /= len(choices_char)
            
            for char in choices_char:
                if eliminiation_prob_dict[char] < average_prob:
                    eliminiation_prob_dict[char] = -100.0
                    
        elif strat == "min":
            for i, char in enumerate(choices_char):
                if i == 0:
                    min_prob = eliminiation_prob_dict[char]
                else:
                    if eliminiation_prob_dict[char] < min_prob:
                        min_prob = eliminiation_prob_dict[char]

            for char in choices_char:
                if eliminiation_prob_dict[char] == min_prob:
                    eliminiation_prob_dict[char] = -100.0
        
        prompt_text_prediction = question.get_poe_prompt(eliminiation_prob_dict)
        inputs = self.tokenizer(prompt_text_prediction, return_tensors="pt", return_token_type_ids=False)
        
        outputs = self.model(**inputs)
        inputs = {k: v.cpu().detach() for k, v in inputs.items()}
        outputs = {k: v.cpu().detach() for k, v in outputs.items()}
        # outputs.logits = outputs.logits.to('cpu')
        # logits = outputs.logits[0, -1]
        logits = outputs['logits'][0, -1]
        probs = logits.float().softmax(dim=-1)
        
        logprobs_dict = {
            self.lbls_map[i]:
            np.log(probs[i].item()) for i in range(len(self.lbls_map))
        }             
 
        # Reduce logprobs_dict to only keys with top 50 largest values
        logprobs_dict = {
            k: v for k, v in sorted(
                logprobs_dict.items(),
                key=lambda item: item[1],
                reverse=True
            )[:50]
        }
        logprobs_dict = mask_logprobs(token_prefix, logprobs_dict, eliminiation_prob_dict)
        
        return ModelResponseNatural(
            logprobs=logprobs_dict,
            # response_list=[outputs]
            response_list=[]
        )
        
    def process_question_brown(self, question):
        prompt_text = question.get_brown_prompt()
        
        response_list = list()
        logprobs = dict()
        unconditional_logprobs = dict()
        lens = dict()
        
        for idx, choice in enumerate(question.choices):
            ltr = idx_to_ltr(idx)
            
            # Get unconditional logprobs
            inputs = self.tokenizer(f"Answer: {choice}", return_tensors="pt", return_token_type_ids=False)
            answer_tokens = inputs['input_ids'][0, 3:] # We need to check this whether it's 2 or 3 or 4 if the model family is changed
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            # for k, v in inputs.items():
            #     inputs[k] = v.to(self.model.device)
            outputs = self.model(**inputs)
            inputs = {k: v.to('cpu') for k, v in inputs.items()}
            # for k, v in inputs.items():
            #     inputs[k] = v.to('cpu')
            outputs.logits = outputs.logits.to('cpu')
                
            logits = outputs.logits[0, 3:, :]
            probs = logits.float().softmax(dim=-1)
            
            choice_n_tokens = probs.shape[0]
            choice_logprobs = float()
            for j in range(choice_n_tokens):
                target_tok = answer_tokens[j].item()
                target_tok_logprob = np.log(probs[j, target_tok].item())
                choice_logprobs += target_tok_logprob
            
            unconditional_logprobs[ltr] = choice_logprobs
            lens[ltr] = choice_n_tokens
            # response_list.append(outputs)
                        
            # Get conditional logprobs
            inputs = self.tokenizer(f"{prompt_text} {choice}", return_tensors="pt", return_token_type_ids=False)
            answer_tokens = inputs['input_ids'][0][-(choice_n_tokens):]
            for k, v in inputs.items():
                inputs[k] = v.to(self.model.device)
            outputs = self.model(**inputs)
            for k, v in inputs.items():
                inputs[k] = v.to('cpu')
            outputs.logits = outputs.logits.to('cpu')
            
            logits = outputs.logits[0, -(choice_n_tokens):, :]
            probs = logits.float().softmax(dim=-1)
            
            choice_n_tokens = probs.shape[0]
            choice_logprobs = float()
            for j in range(choice_n_tokens):
                target_tok = answer_tokens[j].item()
                target_tok_logprob = np.log(probs[j, target_tok].item())
                choice_logprobs += target_tok_logprob
                                
            logprobs[ltr] = choice_logprobs
            # response_list.append(outputs)
            
        return ModelResponseBrown(
            logprobs=logprobs,
            unconditional_logprobs=unconditional_logprobs,
            lens=lens,
            response_list=response_list
        )
    
    def process_question_cot(self, question):
        prompt_text = question.get_cot_prompt()
        inputs = self.tokenizer(prompt_text, return_tensors='pt', return_token_type_ids=False)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        inputs_token_len =  len(inputs['inputs'])
        outputs = self.model.generate(**inputs, max_new_tokens=inputs_token_len+200)
        cot_rationale = self.tokenizer.batch_decode(outputs)
        prompt_text_with_rationale = cot_rationale[0] + '. Therefore, the answer is '
        second_inputs = self.tokenizer(prompt_text_with_rationale, return_tensors='pt', return_token_type_ids=False)
        second_inputs = {k: v.to(self.model.device) for k, v in second_inputs}
        outputs = self.model(**second_inputs)
        outputs.logits = outputs.logit.to('cpu')
        logits = outputs.logits[0, -1]
        probs = logits.float().softmax(dim=-1)
        logprobs_dict = {self.lbls_map[i]: np.log(probs[i].item()) for i in range(len(self.lbls_map))}
        logprobs_dict = {k: v for k, v in sorted(logprobs_dict.items(), key=lambda x: x[1], reverse=True)}
        
        prompt_text = question.get_cot_prompt()
            
        inputs = self.tokenizer(prompt_text, return_tensors="pt", return_token_type_ids=False)
        inputs_token_len = len(inputs['input_ids'])
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        cot_outputs = self.model.generate(**inputs, max_new_tokens=inputs_token_len+200)
        cot_rationale = self.tokenizer.batch_decode(cot_outputs)
        
        prompt_text_with_rationale = cot_rationale[0] + ". Therefore, the answer is "
        
        second_inputs = self.tokenizer(prompt_text_with_rationale, return_tensors="pt", return_token_type_ids=False)
        second_inputs = {k: v.to(self.model.device) for k, v in second_inputs.items()}

        outputs = self.model(**second_inputs)
        outputs.logits = outputs.logits.to('cpu')
        logits = outputs.logits[0, -1]
        probs = logits.float().softmax(dim=-1)

        logprobs_dict = {
            self.lbls_map[i]:
            np.log(probs[i].item()) for i in range(len(self.lbls_map))
        }                
        # Reduce logprobs_dict to only keys with top 50 largest values
        logprobs_dict = {
            k: v for k, v in sorted(
                logprobs_dict.items(),
                key=lambda item: item[1],
                reverse=True
            )[:50]
        }

        return ModelResponseNatural(
            logprobs=logprobs_dict,
            # response_list=[outputs]
            response_list=[]
        )
        
    def process_question_ps(self, question):
        prompt_text = question.get_ps_prompt()
            
        inputs = self.tokenizer(prompt_text, return_tensors="pt", return_token_type_ids=False)
        inputs_token_len = len(inputs['input_ids'])
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        cot_outputs = self.model.generate(**inputs, max_new_tokens=inputs_token_len+200)
        cot_rationale = self.tokenizer.batch_decode(cot_outputs)
        
        prompt_text_with_rationale = cot_rationale[0] + ". Therefore, the answer is "
        
        second_inputs = self.tokenizer(prompt_text_with_rationale, return_tensors="pt", return_token_type_ids=False)
        second_inputs = {k: v.to(self.model.device) for k, v in second_inputs}
        
        outputs = self.model(**second_inputs)
        outputs.logits = outputs.logits.to('cpu')
        logits = outputs.logits[0, -1]
        probs = logits.float().softmax(dim=-1)

        logprobs_dict = {
            self.lbls_map[i]:
            np.log(probs[i].item()) for i in range(len(self.lbls_map))
        }                
        # Reduce logprobs_dict to only keys with top 50 largest values
        logprobs_dict = {
            k: v for k, v in sorted(
                logprobs_dict.items(),
                key=lambda item: item[1],
                reverse=True
            )[:50]
        }

        return ModelResponseNatural(
            logprobs=logprobs_dict,
            # response_list=[outputs]
            response_list=[]
        )
        

if __name__=="__main__":
    dataset = wos_dataset(5736)
    model = LlamaModel("meta-llama/Meta-Llama-3.1-8B-Instruct", False)
    # model = LlamaModel("meta-llama/Meta-Llama-3-8B", False)
    # model = LlamaModel("lmsys/vicuna-7b-v1.5", False)
    for i in range(1):
        resp = model.process_question_natural(dataset[i])
        print(resp)
