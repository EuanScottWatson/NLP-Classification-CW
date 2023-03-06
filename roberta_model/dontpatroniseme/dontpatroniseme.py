import torch
import transformers

MODEL_URLS = {
    "roberta-base": "https://huggingface.co/roberta-base/resolve/main/pytorch_model.bin"
}

PRETRAINED_MODEL = None


def get_model_and_tokenizer(
    model_type, model_name, tokenizer_name, num_classes, state_dict, huggingface_config_path=None
):
    model_class = getattr(transformers, model_name)
    model = model_class.from_pretrained(
        pretrained_model_name_or_path=None,
        config=huggingface_config_path or model_type,
        num_labels=num_classes,
        state_dict=state_dict,
        local_files_only=huggingface_config_path is not None,
    )
    tokenizer = getattr(transformers, tokenizer_name).from_pretrained(
        huggingface_config_path or model_type,
        local_files_only=huggingface_config_path is not None,
    )

    return model, tokenizer


def load_checkpoint(model_type="roberta-base", checkpoint=None, device="cpu", huggingface_config_path=None):
    loaded = torch.load(checkpoint, map_location=device)
    if "config" not in loaded or "state_dict" not in loaded:
        raise ValueError(
            "Checkpoint needs to contain the config it was trained with as well as the state dict"
        )
    class_names = loaded["config"]["dataset"]["args"]["classes"]
    model, tokenizer = get_model_and_tokenizer(
        **loaded["config"]["arch"]["args"],
        state_dict=loaded["state_dict"],
        huggingface_config_path=huggingface_config_path,
    )

    return model, tokenizer, class_names


class DontPatroniseMe:
    def __init__(self, model_type="original", checkpoint=PRETRAINED_MODEL, device="cpu", huggingface_config_path=None):
        super().__init__()
        self.model, self.tokenizer, self.class_names = load_checkpoint(
            model_type=model_type,
            checkpoint=checkpoint,
            device=device,
            huggingface_config_path=huggingface_config_path,
        )
        self.device = device
        self.model.to(self.device)

    @torch.no_grad()
    def predict(self, text):
        self.model.eval()
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, padding=True).to(self.model.device)
        out = self.model(**inputs)[0]
        scores = torch.sigmoid(out).cpu().detach().numpy()
        results = {}
        for i, cla in enumerate(self.class_names):
            results[cla] = (
                scores[0][i] if isinstance(text, str) else [
                    scores[ex_i][i].tolist() for ex_i in range(len(scores))]
            )
        return results
