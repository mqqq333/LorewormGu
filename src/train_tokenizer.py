import random
import json
from tokenizers import(
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer,
)

import os

def train_tokenizer():
    def read_texts_from_jsonl(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                yield data["text"]
    data_path = "pretrain.jsonl"

    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    special_tokens = ["<ukn>", "<s>", "</s>"]

    trainer = trainers.BpeTrainer(
        vocab_size = 6400,
        special_tokens=special_tokens,
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),

    )

    texts = read_texts_from_jsonl(data_path)

    tokenizer.train_from_iterator(texts, trainer=trainer)

    tokenizer.decoder=decoders.ByteLevel()
    
    assert tokenizer.token_to_id("<ukn>") == 0
    assert tokenizer.token_to_id("<s>") == 1
    assert tokenizer.token_to_id("/s") == 2

    tokenizer_dir = './lorewormgu_tokenizer'

    os.makedirs(tokenizer_dir, exist_ok=True)
    tokenizer.save(os.path.join(tokenizer_dir, 'tokenizer.json'))

    tokenizer.model.save('./lorewormgu_tokenizer')


    config = {
        'add_bos_token':False,
        'add_eos_token':False,
        'add_prefix_space':False,
        'added_tokens_decoder':{
            '0':{
                'content': '<ukn>',
                'lstrip': False,
                'normalized':False,
                'rstrip': False,
                'special_word':False,
                'special': True,

            },
            '1':{
                'content': '<s>',
                'lstrip': False,
                'normalized':False,
                'rstrip': False,
                'special_word':False,
                'special': True,
            },
            '2':{
                'content': '</s>',
                'lstrip': False,
                'normalized':False,
                'rstrip': False,
                'special_word':False,
                'special': True,

            }


        },
        'additional_special_tokens':[],
        'bos_token':'<s>',
        'eos_token':'</s>',
        'clean_up_tokenization_spaces':False,
        'eos_token':"</s>",
        'legacy':True,
        'model_max_length':32768,
        'pad_token':'<ukn>',
        'sp_model_kwargs':{},
        'spaces_between_special_tokens':False,
        'tokenizer_class':'PreTrainedTokenizerFast',
        'unk_token':'<ukn>',
        "chat_template": "{% if messages[0]['role'] == 'system' %}{% set system_message = messages[0]['content'] %}{{ '<s>system\\n' + system_message + '</s>\\n' }}{% else %}{{ '<s>system\\n你是 LorewormGu，是一个有用的人工智能助手。</s>\\n' }}{% endif %}{% for message in messages %}{% set content = message['content'] %}{% if message['role'] == 'user' %}{{ '<s>user\\n' + content + '</s>\\n<s>assistant\\n' }}{% elif message['role'] == 'assistant' %}{{ content + '</s>' + '\\n' }}{% endif %}{% endfor %}"
     

    }

    with open(os.path.join(tokenizer_dir,"tokenizer_config.json"),"w",encoding="utf-8") as f:
        json.dump(config,f,ensure_ascii=False,indent=4)

if __name__=="__main__":
    train_tokenizer()