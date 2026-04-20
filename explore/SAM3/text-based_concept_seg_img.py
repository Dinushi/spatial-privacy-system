from ultralytics.models.sam import SAM3SemanticPredictor

# Initialize predictor with configuration
overrides = dict(
    conf = 0.25,
    task = "segment",
    mode = "predict",
    model = "../../models/sam3.1_multiplex.pt",
    half=True,  # Use FP16 for faster inference
    save=True,
)

predictor = SAM3SemanticPredictor(overrides=overrides)

# Set image once for multiple queries
predictor.set_image("../../data/input/sample_img.jpeg")

# Query with multiple text prompts
results = predictor(text=["Laptop", "Coffe table"])
print(results)

# Works with descriptive phrases
results = predictor(text=["Laptop screen", "brown cloth on sofa"])

# # Query with a single concept
# results = predictor(text=["a ceraVe cream bottle"])


