from cv2 import add
from pydantic import NoneBytes
from transformers import LlamaPreTrainedModel, LlamaModel
from transformers.modeling_outputs import CausalLMOutputWithPast
import torch
from torch import nn
import torch.nn.functional as F

from typing import Dict, Optional, Sequence, List, Optional, Tuple, Union
from torch.nn import KLDivLoss, CrossEntropyLoss, MSELoss
from transformers.cache_utils import Cache, DynamicCache

class LlamaForCausalLM(LlamaPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        optionize: Optional[List] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:#B*S -> (B+T)*S*V
            
            bagoflabels = []
            bagoflogits = []
            
            # logits = F.log_softmax(logits, dim=2)
            # Get the x and y coordinates of the non-masked indices
            beta = 0.9
            
            for x in range(labels.shape[0]):
                opt_len=None
                
                non_masked_indices = torch.nonzero(labels[x] != -100)
                for y in non_masked_indices:
                    if y==0:
                        continue
                    else:
                        # if x not in flag:
                        #     if added is not None:
                        #         bagoflabels.append(added)
                        #     flag.append(x)
                        #     added = torch.zeros(self.vocab_size)
                        # if torch.cuda.current_device() == 0:
                            # print(f"expanded_labels[{x}, {y}, {labels[x, y]}]]: change to 1")
                        added = torch.full((self.vocab_size,), 0.0)
                        flag = False
                        if optionize is not None:
                            for batch, opt in enumerate(optionize):
                                for tgt_idx, items in opt.items():
                                    if x == batch and y == tgt_idx:
                                        # min_value = float('inf')   
                                        # min_key = None
                                        # for token, q_value in items.items():
                                        #     if q_value < min_value:
                                        #         min_value = q_value
                                        #         min_key = token
                                        # added[min_key] = 1.0
                                        flag = True
                                        for token, q_value in items.items():
                                            added[token] = q_value
                                            # if opt_len is None:
                                            #     opt_len = min(items.values())
                                            # else:
                                            #     opt_len -=1

                                            # added[token] = beta ** (q_value + (opt_len - min(items.values())))
                                                
                                        #         # print(f"added[{token}]: {added[token]}")
                        if not flag:
                            added[labels[x, y]] = 1.0
                        added = added.float()
                        # added = F.softmax(added, dim=0)
                        
                        
                        bagoflabels.append(added)
                        bagoflogits.append(logits[x, y-1])
            expanded_labels = torch.stack(bagoflabels).float().to(logits.device)
            expanded_logits = torch.stack(bagoflogits).to(logits.device).squeeze(dim=1)
            expanded_logits = F.log_softmax(expanded_logits, dim=1)
            non_masked_indices = torch.nonzero(expanded_labels > 0.0001)
            
            for x, y in non_masked_indices:
                if torch.cuda.current_device() == 0:
                    print(f'logits[{x}, {y}]: {expanded_logits[x, y]}, shift_labels[{x}, {y}]: {expanded_labels[x, y]}')
            # print(f"optionize: {optionize}")
            loss_fct = CrossEntropyLoss()

            # Enable model parallelism
            loss = loss_fct(expanded_logits, expanded_labels)

            
            
            # shift_logits = F.log_softmax(shift_logits, dim=2)
            # shift_labels = F.softmax(shift_labels, dim=2)
            # # print(f"shift_logits: {shift_logits.shape} shift_labels: {shift_labels.shape}")
            # # Flatten the tokens
            # loss_fct = KLDivLoss(reduction="batchmean")
            # #loss_fct = nn.NllLoss(reduction="mean")
            
            # # shift_logits = shift_logits.view(-1, self.config.vocab_size)#B(S-1)*V->(B+T)(S-1)*V
            # # # print(f"shift_logits: {shift_logits.shape}")
            # # shift_labels = shift_labels.view(-1, self.config.vocab_size)#B(S-1)->(B+T)(S-1)*V
            
            # # print(f"shift_labels: {shift_labels.shape}")
            # # Enable model parallelism
            # shift_labels = shift_labels.to(shift_logits.device)
            # loss = loss_fct(shift_logits, shift_labels)

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
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                cache_length = past_key_values.get_seq_length()
                past_length = past_key_values.seen_tokens
                max_cache_length = past_key_values.get_max_length()
            else:
                cache_length = past_length = past_key_values[0][0].shape[2]
                max_cache_length = None

            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as
            # input)
            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
            # input_ids based on the past_length.
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.

            # If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
            if (
                max_cache_length is not None
                and attention_mask is not None
                and cache_length + input_ids.shape[1] > max_cache_length
            ):
                attention_mask = attention_mask[:, -max_cache_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past