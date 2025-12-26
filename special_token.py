from transformers import AutoTokenizer

# MODEL = "/u/haoboxu/work/verl/qwen_probe/Qwen3-4B-Base"   # 改成你的模型或本地路径
MODEL="/u/haoboxu/work/verl/qwen_probe/llama-3.2-1b"
tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True, use_fast=True)

specials = [
    "\n\n",       # newline
    "\n", '\t', '\r', '\r\n',     # newline
    ".", "。",  # period (en/zh)
    "?", "？",
    "!", "！",
    ".\n",
    "$.",
    "$\n",')$',']$', '$$', '}$', '>$', 
    
    ". ", "? ", "! ",          # 标点+空格
    "?\n", "!\n",              # 标点+换行
    "...\n", "…\n", "……\n",    # 省略号+换行
    ").", ")?", ")!", ").\n",  # 右括号+标点/换行
    "].", "]?", "]!", "].\n",
    "}.", "}?", "}!", "}.\n",
    "”。", "”？", "”！", "”。\n",  # 中文右引号+标点
    "$", "$$", "$$\n",         # 数学结束更核心的几个
    " $","\\\n","\\\\\n",
    "```", "```\n",            # 代码块结束
    
]

sid = set()
for s in specials:
    ids = tokenizer.encode(s, add_special_tokens=False)
    if len(ids) >= 2:
        continue
    print(f"{repr(s):>6} -> {ids}")
    sid.add(ids[0])
print(f"Special token ids: {sorted(list(sid))}")
