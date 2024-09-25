#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import List, Optional, Tuple, Union
from M2PT.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, \
    DEFAULT_IM_END_TOKEN
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

# from transformers import AutoConfig, AutoModelForCausalLM, \
#                          LlamaConfig, LlamaModel, LlamaForCausalLM
from transformers import AutoConfig, AutoModelForCausalLM

# 打开以下行PromptTuning，注释finetune
from .modeling_llamaPT import LlamaModel, LlamaForCausalLM, LlamaConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
from M2PT.model.llava_archPT import LlavaMetaModel, LlavaMetaForCausalLM


class LlavaConfig(LlamaConfig):
    model_type = "llava_finetune"


class LlavaLlamaModel(LlavaMetaModel, LlamaModel):
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig, PT_len, VIT_PT_len):
        super(LlavaLlamaModel, self).__init__(config, PT_len, VIT_PT_len)
        self.PT_len = PT_len
        self.VIT_PT_len = VIT_PT_len


class LlavaLlamaForCausalLM(LlamaForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config, PT_len, VIT_PT_len):
        super(LlamaForCausalLM, self).__init__(config,PT_len, VIT_PT_len)
        self.PT_len = PT_len
        self.VIT_PT_len = VIT_PT_len
        self.model = LlavaLlamaModel(config, self.PT_len, self.VIT_PT_len)

        # print(f"!!!!!!!!!!!!!!{config.hidden_size},{config.vocab_size}")
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # print(f"11111111{self.lm_head.weight}")
        # print(f"11111111{sum(self.lm_head.weight)}")
        # print(f"11111111{type(self.lm_head.weight)}")

        # Initialize weights and apply final processing
        self.post_init()
        self.debug_lmhead = 0
        self.index = 0

    def get_prompt_embeddings(self):
        return self.model.prompts

    # require grads 20240513
    def make_prompt_learnable(self):
        # go through all prompts and make them learnable
        self.model.prompts.requires_grad_(True)

    def make_prompt_unlearnable(self):
        # go through all prompts and make them learnable
        self.model.prompts.requires_grad_(False)

    def get_model(self):
        return self.model

    def get_lm_head(self):
        return self.lm_head

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        # torch.cuda.empty_cache()
        # print(f"$$$lmhead require grad here:{self.lm_head.weight.requires_grad}")
        # print(f"$$$lmhead grad here:{self.lm_head.weight.grad}")
        # print(f"$$$￥lmhead grad here:{self.lm_head.weight}")
        if self.index > 2:
            # print(f"^^lm_head_sanity_check:SUM {torch.sum(self.debug_lmhead - self.lm_head.weight)},require_grad:{self.lm_head.weight.requires_grad},grad:{self.lm_head.weight.grad},weight:{self.lm_head.weight}")
            print(f"^^lm_head_sanity_check:SUM {torch.sum(self.debug_lmhead - self.lm_head.weight)},require_grad:{self.lm_head.weight.requires_grad}")
        self.debug_lmhead = self.lm_head.weight.clone().detach()
        self.index += 1
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # print("label info")
        # print(input_ids.shape)
        # print(input_ids)

        image_token_indices = torch.where(input_ids == IMAGE_TOKEN_INDEX)[0]
        # print(f"image_token_indices {image_token_indices}")
        # print(IMAGE_TOKEN_INDEX)
        # print(f"input_ids {input_ids}")
        # print(f"a:{input_ids[:,:35].shape} b:{input_ids[:,35:].shape},input_ids:{input_ids.shape}")
        input_ids, attention_mask, past_key_values, inputs_embeds, labels = self.prepare_inputs_labels_for_multimodal(
            input_ids, attention_mask, past_key_values, labels, images)

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict
            # torch.set_printoptions(threshold=float('inf'))
            # print(f"labels: {labels}")
            # print(f"label length: {labels.shape}")
            # print(f"input_ids: {input_ids.shape}")
            # exit()
            # print(f"logits:{logits}")
            # print(f"labels:{labels}")
            # print(f"logits:{logits.shape},label:{labels.shape}")
            # logits = torch.cat([logits[:,0:1,:],logits[:,self.PT_len+1:,:]],dim=1)
            # logits = torch.cat([logits[:,:35,:],logits[:,35+self.VIT_PT_len:,:]],dim=1)
            # labels = torch.cat([labels[:,:35],labels[:,35+self.VIT_PT_len:]],dim=-1)
            temp_labels = torch.full((labels.shape[0], self.PT_len), -100).to(labels.device)
            labels = torch.cat((temp_labels, labels), dim=1)
            # print(f"labels:{labels.shape},cuda:{logits.device}")
            # print(f"logits:{logits.shape},cuda:{logits.device}")
            # print(f"2labels:{labels.shape},cuda:{logits.device}")
            # print(f"2logits:{logits.shape},cuda:{logits.device}")
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model/pipeline parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "images": kwargs.get("images", None),
            }
        )
        return model_inputs


AutoConfig.register("llava_finetune", LlavaConfig)
AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM)