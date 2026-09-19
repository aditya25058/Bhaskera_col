import os

p = "/home/palakm/MoEServingSim/aditya/models/DeepSeek-Coder-V2-Instruct/modeling_deepseek.py"
with open(p, "r") as f:
    c = f.read()

target = "from transformers.utils.import_utils import is_torch_fx_available"
replacement = "try:\n    from transformers.utils.import_utils import is_torch_fx_available\nexcept ImportError:\n    is_torch_fx_available = lambda: hasattr(torch, 'fx')"

if target in c:
    c = c.replace(target, replacement)
    with open(p, "w") as f:
        f.write(c)
    print("Patched source modeling_deepseek.py successfully!")
else:
    print("Target not found or already patched.")

# Clear cache module directory
os.system("rm -rf /home/palakm/.cache/huggingface/modules/transformers_modules/DeepSeek_hyphen_Coder_hyphen_V2_hyphen_Instruct")
print("Cleared cache module directory.")
