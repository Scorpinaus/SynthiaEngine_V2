from safetensors.torch import load_file

sd = load_file(r"D:\diffusion\checkpoints\zImageTurboAIO_zImageTurboFP16AIO.safetensors", device="cpu")
print("num keys:", len(sd))
print("has qwen blocks:", any(k.startswith("text_encoders.qwen3_4b.") for k in sd.keys()))
print("qwen-ish key count:", sum(k.startswith("text_encoders.qwen3_4b.") for k in sd.keys()))