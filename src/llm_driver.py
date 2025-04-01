from mlx_lm import load, generate
from models import LLMs


class LLM:
    def __init__(self, model_name: LLMs):
        self._model_name = model_name.value
        self._model, self._tokenizer = load(f"mlx-community/{self._model_name}")

    def infer(self, prompt):
        messages = [{"role": "user", "content": prompt}]
        prompt = self._tokenizer.apply_chat_template(
            messages, add_generation_prompt=True
        )
        text = generate(self._model, self._tokenizer, prompt=prompt, verbose=True)
        return text


def main():
    model_name = LLMs.QWEN2_5_32B_INSTRUCT_BF16
    model = LLM(model_name)
    prompt = "日本の首都は？"
    _ = model.infer(prompt)


if __name__ == "__main__":
    main()
