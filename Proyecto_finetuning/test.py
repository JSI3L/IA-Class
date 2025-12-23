# Script rápido de prueba (test_model.py)
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

model_id = "meta-llama/Llama-3.2-1B-Instruct"
adapter_path = "./mi_modelo_finetuned"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
model = PeftModel.from_pretrained(model, adapter_path) # Carga tus pesos entrenados

prompt = "¿Cómo se comenta el código en Python?"
inputs = tokenizer(f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n", return_tensors="pt")

outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=False))