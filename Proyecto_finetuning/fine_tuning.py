import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model

# 1. Configuración
model_id = "meta-llama/Llama-3.2-1B-Instruct"

# 2. Cargar Tokenizer y Modelo en modo CPU
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    torch_dtype=torch.float32, 
    device_map={"": "cpu"}
)

# 3. Configurar LoRA (Adaptación de bajo rango)
# Esto permite entrenar solo una pequeña parte del modelo, ideal para CPU
config = LoraConfig(
    r=8, 
    lora_alpha=16, 
    target_modules=["q_proj", "v_proj"], 
    lora_dropout=0.05, 
    bias="none", 
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, config)

# 4. Cargar y formatear el dataset
dataset = load_dataset("json", data_files="dataset.jsonl", split="train")

def format_func(example):
    # Formato oficial de Llama 3.2 Instruct
    text = (
        f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
        f"{example['prompt']}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        f"{example['response']}<|eot_id|>"
    )
    return tokenizer(text, truncation=True, max_length=512)

tokenized_dataset = dataset.map(format_func, remove_columns=dataset.column_names)

# 5. Parámetros de Entrenamiento Mejorados
training_args = TrainingArguments(
    output_dir="./resultado_temp",
    per_device_train_batch_size=2, # Aumentamos un poco si la RAM lo permite
    gradient_accumulation_steps=4, 
    num_train_epochs=10,           # <--- Aumenta a 10 épocas para datasets pequeños
    learning_rate=5e-5,            # <--- Un learning rate más fino
    use_cpu=True,
    logging_steps=1,
    save_strategy="no",
    warmup_steps=10,
    weight_decay=0.01
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

print("Entrenamiento iniciado...")
trainer.train()

# 6. Guardar el modelo entrenado
model.save_pretrained("./mi_modelo_finetuned")
tokenizer.save_pretrained("./mi_modelo_finetuned")
print("¡Entrenamiento finalizado exitosamente!")