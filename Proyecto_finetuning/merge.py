import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model_id = "meta-llama/Llama-3.2-1B-Instruct"
adapter_dir = "./mi_modelo_finetuned"
output_dir = "./modelo_final_fusionado"

# 1. Cargar modelo base y tokenizer
print("Cargando modelo base...")
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id, 
    torch_dtype=torch.float16, # Recomendado para la fusión
    device_map={"": "cpu"}
)
tokenizer = AutoTokenizer.from_pretrained(base_model_id)

# 2. Cargar el adaptador LoRA
print("Cargando adaptadores...")
model = PeftModel.from_pretrained(base_model, adapter_dir)

# 3. Fusionar y guardar
print("Fusionando pesos (Merge)...")
model = model.merge_and_unload()

print(f"Guardando modelo en {output_dir}...")
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print("¡Fusión completada!")