"""Small-vocabulary visual descriptions for raw photos, independent of any mailbox."""

import argparse
from pathlib import Path

MODEL = "Xenova/clip-vit-base-patch32"
REVISION = "6b7c523703ae26e39be4b03774966f8571c7f523"
FOLDER = Path(__file__).resolve().parents[1] / ".cache/live-clip"

# Only fixed, public class prompts are embedded at installation time.
LABELS = {
    "паспорт, удостоверение личности": "an open passport or identity card",
    "чек, квитанция об оплате": "a printed shopping receipt",
    "договор, текстовый документ": "a page of a typed contract",
    "таблица, отчёт": "a spreadsheet with rows and columns",
    "билет, посадочный талон": "an airline boarding pass or train ticket",
    "карта, схема маршрута": "a street map",
    "снимок экрана приложения": "a screenshot of a software application",
    "рукописная записка": "a handwritten note on paper",
    "чертёж, план помещения": "an architectural floor plan",
    "портрет человека": "a portrait photograph of one person",
    "группа людей": "a photograph of a group of people",
    "растение, цветок": "a close-up photograph of a flower blossom",
    "автомобиль, машина": "a photograph of a car",
    "пейзаж, горы": "a photograph of a mountain landscape",
    "море, пляж": "a photograph of a beach",
    "здание, дом": "a photograph of a building",
    "собака": "a photograph of a dog",
    "кошка": "a photograph of a cat",
    "еда, блюдо": "a photograph of food on a plate",
    "мебель, комната": "a photograph of a furnished room",
    "велосипед": "a photograph of a bicycle",
    "экран телефона": "a photograph of a mobile phone",
    "другое изображение": "an abstract or unrecognizable image",
}


def session(path):
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    options.add_session_config_entry("session.intra_op.allow_spinning", "0")
    return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


def prepare(folder, offline):
    import numpy as np
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer

    for name in (
        "tokenizer.json",
        "onnx/text_model_quantized.onnx",
        "onnx/vision_model_quantized.onnx",
    ):
        if not (offline and (folder / name).is_file()):
            hf_hub_download(
                MODEL,
                name,
                revision=REVISION,
                local_dir=folder,
                local_files_only=offline,
            )
    tokenizer = Tokenizer.from_file(str(folder / "tokenizer.json"))
    model = session(folder / "onnx/text_model_quantized.onnx")
    vectors = []
    for prompt in LABELS.values():
        ids = np.array([tokenizer.encode(prompt).ids], dtype=np.int64)
        inputs = {"input_ids": ids, "attention_mask": np.ones_like(ids)}
        v = model.run(
            ["text_embeds"], {arg.name: inputs[arg.name] for arg in model.get_inputs()}
        )[0][0]
        vectors.append(v / np.linalg.norm(v))
    np.savez(folder / "labels.npz", labels=list(LABELS), vectors=np.asarray(vectors))


class LiveCaptioner:
    def __init__(self, folder):
        self.folder = folder
        self.model = None

    def __call__(self, image):
        # On a light paper scan the OCR text is usually more specific than a class name.
        light = image.convert("L").resize((32, 32)).histogram()
        if sum(light[235:]) > 0.90 * sum(light):
            return ""
        import numpy as np
        from PIL import Image, ImageOps

        if self.model is None:
            self.model = session(self.folder / "onnx/vision_model_quantized.onnx")
            with np.load(self.folder / "labels.npz", allow_pickle=False) as data:
                self.labels, self.vectors = data["labels"], data["vectors"]
        rgb = ImageOps.fit(
            image.convert("RGB"), (224, 224), method=Image.Resampling.BICUBIC
        )
        pixels = np.asarray(rgb, dtype=np.float32) / 255
        pixels -= np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
        pixels /= np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
        vec = self.model.run(
            ["image_embeds"],
            {self.model.get_inputs()[0].name: pixels.transpose(2, 0, 1)[None]},
        )[0][0]
        scores = self.vectors @ (vec / np.linalg.norm(vec))
        best = int(scores.argmax())
        if scores[best] < 0.20 or best == len(self.labels) - 1:
            return ""
        return "Предположительно: " + str(self.labels[best]) + "."


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--folder", type=Path, default=FOLDER)
    args = parser.parse_args()
    prepare(args.folder, args.offline)
