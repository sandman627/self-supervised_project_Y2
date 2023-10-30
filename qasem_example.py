from qasem.end_to_end_pipeline import QASemEndToEndPipeline

pipe = QASemEndToEndPipeline(annotation_layers=('qasrl'),  nominalization_detection_threshold=0.75, contextualize = True)  
sentences = ["The doctor was interested in Luke 's treatment as he was still not feeling well .", "Tom brings the dog to the park."]
outputs = pipe(sentences)

print(outputs)