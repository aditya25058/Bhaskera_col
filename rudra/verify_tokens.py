import torch
from transformers import AutoTokenizer

col = torch.load('/home/bapic_iiitd/2_group/tokens_mixtral_colossus.pt')
ref = torch.load('/home/bapic_iiitd/2_group/tokens_mixtral_2gpu.pt')
tokenizer = AutoTokenizer.from_pretrained('/home/bapic_iiitd/2_group/models/Mixtral-8x7B-v0.1')

print(f'Total COLOSSUS prompt entries: {len(col)}')
print(f'Total Reference prompt entries: {len(ref)}')

for i, (c, r) in enumerate(zip(col, ref)):
    c_list = c.tolist() if isinstance(c, torch.Tensor) else c
    r_list = r.tolist() if isinstance(r, torch.Tensor) else r
    r_sub = r_list[:len(c_list)]
    c_tensor = torch.tensor(c_list)
    r_tensor = torch.tensor(r_sub)
    eq = torch.equal(c_tensor, r_tensor)
    print(f'=== Prompt {i+1} ===')
    print(f'  Tokens generated: {len(c_list)}')
    print(f'  torch.equal: {eq}')
    print(f'  Colossus IDs: {c_list}')
    print(f'  Dense Ref IDs: {r_sub}')
    c_text = tokenizer.decode(c_list, skip_special_tokens=True)
    r_text = tokenizer.decode(r_sub, skip_special_tokens=True)
    print(f'  Decoded text match: {c_text == r_text}')
    print(f'  Decoded: {repr(c_text)}')
