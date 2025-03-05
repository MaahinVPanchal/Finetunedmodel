# Finetunedmodel
# Llama-2-7b-chat-hf Fine-tuning for Wildfire Q&A

This project fine-tunes the `meta-llama/Llama-2-7b-chat-hf` model for question answering, specifically focused on the 2023 wildfires in Hawaii. It utilizes PEFT for efficient fine-tuning and BitsAndBytes for quantization, with optional Wandb logging for experiment tracking.

## Overview

The Jupyter Notebook `Finetunning.ipynb` contains the code for:

*   Loading a pre-trained Llama-2-7b-chat-hf model.
*   Quantizing the model using BitsAndBytes for reduced memory footprint.
*   Fine-tuning the model on a wildfire-related dataset.
*   Saving the fine-tuned model.
*   Loading the fine-tuned model and using it for question answering.

## Prerequisites

*   Python 3.10+
*   CUDA-enabled GPU
*   Libraries:
    *   `torch`
    *   `transformers`
    *   `peft`
    *   `bitsandbytes`
    *   `accelerate`
    *   `wandb` (for logging, optional)
    *   `datasets`

Install the required libraries using pip:


## Setup

1.  **Clone the repository:** (If you have a repository, otherwise, create a directory)

    ```
    git clone <your_repository_url>
    cd <your_repository_directory>
    ```

2.  **Install Dependencies:**

    ```
    pip install -r requirements.txt # If you have a requirements.txt file
    #or
    pip install torch transformers peft bitsandbytes accelerate wandb datasets
    ```

3.  **Download Llama-2-7b-chat-hf:**

    *   You'll need to request access to the `meta-llama/Llama-2-7b-chat-hf` model on the Hugging Face Hub and accept the terms.
    *   Ensure you have a Hugging Face API token and are logged in:

        ```
        from huggingface_hub import login
        login()  #Enter your token when prompted
        ```

## Usage

1.  **Open the `Finetunning.ipynb` notebook** in Google Colab or a Jupyter environment.

2.  **Configure BitsAndBytes:** The notebook utilizes `BitsAndBytesConfig` to load the model in 4-bit precision (NF4) for memory efficiency.

    ```
    nf4Config = BitsAndBytesConfig(
     load_in_4bit=True,
     bnb_4bit_use_double_quant=True,
     bnb_4bit_quant_type="nf4",
     bnb_4bit_compute_dtype=torch.bfloat16
    )
    ```

3.  **Load the Base Model:** Loads the Llama-2-7b-chat-hf model with the specified quantization configuration.

    ```
    base_model_id = "meta-llama/Llama-2-7b-chat-hf"

    base_model = AutoModelForCausalLM.from_pretrained(
     base_model_id, #same as before
     quantization_config=nf4Config, #same quantization config as before
     device_map="auto",
     trust_remote_code=True,
     use_auth_token=True
    )

    tokenizer = LlamaTokenizer.from_pretrained(base_model_id, use_fast=False,
     trust_remote_code=True, add_eos_token=True)
    ```

4.  **Fine-tuning:** The notebook fine-tunes the loaded model with a dataset.  The `Trainer` class is used to handle the training loop. Key components:

    *   **Dataset:**  Requires a dataset suitable for question answering.  The notebook uses a placeholder; you MUST replace this with your actual dataset.  It's recommended to use a dataset formatted similarly to SQuAD or a question-answering dataset.

        ```
        from datasets import load_dataset

        # Example: Replace with your actual dataset path or Hugging Face dataset name
        data = load_dataset("path/to/your/dataset", split="train")
        # or
        # data = load_dataset("squad", split="train")
        ```

    *   **Training Arguments:** Control the training process (learning rate, batch size, number of epochs, etc.).

        ```
        from transformers import TrainingArguments

        training_arguments = TrainingArguments(
            output_dir="./results",  # Directory to save results
            num_train_epochs=3,       # Number of training epochs
            per_device_train_batch_size=4,  # Batch size per device during training
            gradient_accumulation_steps=1, # Number of update steps to accumulate before performing a backward/update pass
            optim="paged_adamw_32bit",   # Optimizer
            save_steps=250,            # Save checkpoint every X steps
            logging_steps=50,           # Log metrics every X steps
            learning_rate=2e-4,        # Learning rate
            weight_decay=0.001,         # Weight decay
            fp16=False,                # Use fp16 (mixed precision)
            bf16=False,                # Use bf16 (mixed precision)
            max_grad_norm=0.3,          # Maximum gradient norm (for gradient clipping)
            max_steps=-1,              # If > 0: set total number of training steps to perform. Override num_train_epochs.
            warmup_ratio=0.03,          # Linear warmup over warmup_ratio fraction of the training steps
            group_by_length=True,       # Group sequences of roughly the same length together for more efficient training
            lr_scheduler_type="cosine", # Learning rate scheduler
            report_to="wandb",           # Report metrics to Wandb
            save_total_limit=3,         # Only save the last X checkpoints
            push_to_hub=False,          # Whether to push the model to the Hugging Face Hub
        )
        ```

    *   **Data Collator:**  Formats the data into batches for training.  The example uses `default_data_collator`, which may not be optimal for all datasets.  Consider creating a custom data collator if needed.

        ```
        from transformers import DataCollatorForLanguageModeling
        data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
        ```

    *   **Trainer:** This orchestrates the fine-tuning process.

        ```
        from transformers import Trainer

        trainer = Trainer(
            model=base_model,          # The model to train
            train_dataset=data,         # The training dataset
            args=training_arguments,   # Training arguments
            data_collator=data_collator, # The data collator
        )

        trainer.train()
        ```

        *   **Saving the Model:** The trainer saves the *adapter* weights (PEFT model).
        ```
        model.save_pretrained("finetuned_model")
        tokenizer.save_pretrained("finetuned_model")

        ```

5.  **Load the Fine-tuned Model:** Loads the fine-tuned model and tokenizer from the saved directory.

    ```
    from peft import PeftModel

    # Load the PEFT model from the save directory
    modelFinetuned = PeftModel.from_pretrained(base_model, "finetuned_model")
    ```

6.  **Question Answering:** The final cells demonstrate how to use the fine-tuned model to answer questions. Enter your question in the `question` variable, and the notebook will generate an answer. It is VERY IMPORTANT that you set `modelFinetuned.eval()` and use the `torch.no_grad()` context manager.

    ```
    question = "Just answer this question: Tell me about the role of Maui Emergency Management Agency (MEMA) in the 2023 wildfires??"

    # Format the question
    eval_prompt = f"Just answer this question accurately: {question}\\n\\n"

    promptTokenized = tokenizer(eval_prompt, return_tensors="pt").to("cuda")

    modelFinetuned.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # Disable gradient calculation to save memory
        print(tokenizer.decode(modelFinetuned.generate(**promptTokenized, max_new_tokens = 1024), skip_special_tokens=True))

    torch.cuda.empty_cache() # Free up GPU memory
    ```

7.  **User Question Example:** Shows how to use the model with a different user question.

    ```
    # User enters question below
    user_question = "Summarize the officer accounts of the wildfire in Hawaii?"

    # Format the question
    eval_prompt = f"Just answer this question accurately: {user_question}\\n\\n"

    promptTokenized = tokenizer(eval_prompt, return_tensors="pt").to("cuda")

    modelFinetuned.eval()
    with torch.no_grad():
        print(tokenizer.decode(modelFinetuned.generate(**promptTokenized, max_new_tokens = 1024), skip_special_tokens=True))
    torch.cuda.empty_cache()
    ```

## Important Considerations

*   **Dataset:** The performance of the fine-tuned model heavily depends on the quality, format and size of the training dataset. Ensure your dataset is relevant to the wildfire domain and contains diverse examples.  The notebook requires a dataset formatted for Question Answering.
*   **Overfitting:** Pay close attention to the training loss and, crucially, the validation loss (if you have a validation set) to avoid overfitting.  Select the checkpoint with the lowest *validation* loss. Early stopping or regularization techniques may be necessary. The note in the notebook `# beware of overfitting!` is critical.
*   **Checkpoint Selection:** The notebook mentions changing the model checkpoint based on training loss (`# Change model checkpoint that has least training loss in the code below`).  Choose a model checkpoint that provides good generalization performance, which is measured against the *validation* dataset.
*   **Memory:** Fine-tuning large language models requires significant GPU memory. Consider using smaller batch sizes, gradient accumulation, or techniques like quantization to reduce memory consumption. Setting `modelFinetuned.eval()` and using `torch.no_grad()` are vital to reducing memory consumption during inference.
*   **Inference Quality:** The generated text demonstrates repetition.  This might be addressed by further fine-tuning, dataset augmentation, or modifying the generation parameters (e.g., temperature, top\_p) and also to overcome repetition I enhance prompt such that it provided me response properly based on data which is has been train other then doing repition of different question and answer.

## Wandb Logging

The notebook is configured to use Weights & Biases (Wandb) for tracking experiments. To enable Wandb logging:

1.  Sign up for a Wandb account at [https://wandb.ai/site](https://wandb.ai/site).
2.  Install the `wandb` library: `pip install wandb`.
3.  Log in to your Wandb account: `wandb login`.

## Disclaimer

This `README.md` is based on the provided Jupyter Notebook and makes assumptions about the intended use case and available resources. 
