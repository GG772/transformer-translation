from transformers import AutoTokenizer


def get_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("Jihuai/bert-ancient-chinese")
    special_tokens_dict = {}
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = "[BOS]"
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = "[EOS]"

    if special_tokens_dict:
        tokenizer.add_special_tokens(special_tokens_dict)
    return tokenizer
tokenizer = get_tokenizer()


text = "天地玄黄"
pad = tokenizer.pad_token_id

if __name__ == '__main__':
    print(type(tokenizer))
    print(len(tokenizer))
    print(tokenizer.encode(text))
    print("pad: ", pad)
    tokens = tokenizer.convert_ids_to_tokens([101, 102])
    print(tokens)
    # print(tokenizer.special_tokens_map)
    # print(tokenizer.all_special_tokens)
