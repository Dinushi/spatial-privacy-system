from ultralytics.models.sam import SAM3VideoSemanticPredictor

# Initialize semantic video predictor
overrides = dict(
    conf=0.25, 
    task="segment", 
    mode="predict", 
    imgsz=640, 
    model="../../models/sam3.1_multiplex.pt", 
    half=True, 
    save=True
)

predictor = SAM3VideoSemanticPredictor(overrides=overrides)

# # Track concepts using text prompts
# results = predictor(source="../../data/input/sample1.mp4", text=["clock", "brown cloth on sofa", "ceraVe Cream Bottle"], stream=True)

# # Process results
# for r in results:
#     r.show()  # Display frame with tracked objects

# Track concepts using text prompts
results = predictor(source="../../data/input/sample0.mp4", text=["clock", "brown cloth lying only on sofa", "ceraVe Cream Bottle"], stream=True)

# Process results
for r in results:
    # r.show()  # Display frame with tracked objects
    pass # no desktop viewer in the server


# Alternative: Track with bounding box prompts
# results = predictor(
#     source="../../data/input/sample1.mp4",
#     bboxes=[[864, 383, 975, 620], [705, 229, 782, 402]],
#     labels=[1, 1],  # Positive labels
#     stream=True,
# )